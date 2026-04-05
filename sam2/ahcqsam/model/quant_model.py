import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer  # noqa: F401
from quantization.quantized_module import PreQuantizedLayer, QuantizedMatMul
from sam2.modeling.backbones.hieradet import MultiScaleAttention, do_pool, MultiScaleBlock
from sam2.modeling.backbones.utils import window_partition, window_unpartition
from sam2.modeling.sam2_utils import MLP
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.sam.transformer import Attention, TwoWayAttentionBlock, RoPEAttention
from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.memory_encoder import CXBlock, MaskDownSampler, Fuser, MemoryEncoder
from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention


def update_specialized_quantizer_config(base_config, quantizer_name):
    import copy
    specialized_config = copy.deepcopy(base_config)

    update_keys = {
        'softmax_agq': {'quantizer': 'AdaptiveGranularityQuantize',
                        'observer': 'LogAvgMSEFastObserver'},
        'softmax_lnq': {'quantizer': 'LogTransformQuantize',
                        'observer': 'LogTransformObserver'},
        'bimodal': {'quantizer': 'LSQSignFakeQuantize',
                    'observer': 'SignAvgMSEFastObserver'}
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config


class QuantMultiScaleAttention(QuantizedBlock):
    def __init__(self, org_module: MultiScaleAttention, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()

        self.dim = org_module.dim
        self.dim_out = org_module.dim_out
        self.num_heads = org_module.num_heads
        self.q_pool = org_module.q_pool

        if ahcqsam_config.acnr:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax_lnq')
        elif ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax_agq')
        else:
            softmax_a_config = a_qconfig

        self.qkv = PreQuantizedLayer(org_module.qkv, None, w_qconfig, a_qconfig)
        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig)
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        d_k = q.size(-1)

        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = attn_scores.softmax(dim=-1)

        attn_weights = self.softmax_post_act_fake_quantize(attn_weights, value=v)

        x = torch.matmul(attn_weights, v)

        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class QuantEncoderMLP(QuantizedBlock):
    def __init__(self, org_module: MLP, w_qconfig, a_qconfig, ahcqsam_config):
        super().__init__()

        if ahcqsam_config.hluq:
            lin2_type = 'hybrid'
        else:
            lin2_type = 'normal'

        self.lin1 = PreQuantizedLayer(org_module.layers[0], None, w_qconfig, a_qconfig)
        self.lin2 = PreQuantizedLayer(org_module.layers[1], None, w_qconfig, a_qconfig, lin2_type)
        self.sigmoid_output = org_module.sigmoid_output
        self.act = org_module.act

    def forward(self, x):
        x = self.lin2(self.act(self.lin1(x)))
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class QuantDecoderMLP(QuantizedBlock):
    def __init__(self, org_module: MLP, w_qconfig, a_qconfig, ahcqsam_config):
        super().__init__()

        if ahcqsam_config.cag:
            lin1_type = 'group'
        else:
            lin1_type = 'normal'
        if ahcqsam_config.hluq:
            lin2_type = 'hybrid'
        else:
            lin2_type = 'normal'

        self.lin1 = PreQuantizedLayer(org_module.layers[0], None, w_qconfig, a_qconfig, lin1_type)
        self.lin2 = PreQuantizedLayer(org_module.layers[1], None, w_qconfig, a_qconfig, lin2_type)
        self.sigmoid_output = org_module.sigmoid_output
        self.act = org_module.act

    def forward(self, x):
        x = self.lin2(self.act(self.lin1(x)))
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class QuantMultiScaleBlock(nn.Module):
    def __init__(self, org_module: MultiScaleBlock, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()

        self.dim = org_module.dim
        self.dim_out = org_module.dim_out
        self.norm1 = org_module.norm1
        self.window_size = org_module.window_size
        self.pool = org_module.pool
        self.q_stride = org_module.q_stride
        self.attn = QuantMultiScaleAttention(org_module.attn, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)
        self.drop_path = org_module.drop_path
        self.norm2 = org_module.norm2
        self.mlp = QuantEncoderMLP(org_module.mlp, w_qconfig, a_qconfig, ahcqsam_config)
        pool = getattr(org_module, "pool", None)
        proj = getattr(org_module, "proj", None)
        if pool is not None:
            self.pool = org_module.pool
        else:
            self.pool = None
        if proj is not None:
            self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QuantFpnNeck(QuantizedBlock):
    def __init__(self, org_module: FpnNeck, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()
        self.position_encoding = org_module.position_encoding
        self.backbone_channel_list = org_module.backbone_channel_list
        self.d_model = org_module.d_model
        self.fpn_interp_model = org_module.fpn_interp_model
        self.fuse_type = org_module.fuse_type
        self.convs = nn.ModuleList()
        self.fpn_top_down_levels = org_module.fpn_top_down_levels

        for seq in org_module.convs:
            if isinstance(seq, nn.Sequential):
                conv = seq[0]
            else:
                conv = seq
            q_conv = PreQuantizedLayer(conv, None, w_qconfig, a_qconfig)
            self.convs.append(q_conv)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


class QuantAttention(QuantizedBlock):
    def __init__(self, org_module: Attention, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()
        self.embedding_dim = org_module.embedding_dim
        self.kv_in_dim = org_module.kv_in_dim
        self.internal_dim = org_module.internal_dim
        self.num_heads = org_module.num_heads

        if ahcqsam_config.cag:
            proj_type = 'group'
        else:
            proj_type = 'normal'
        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax_agq')
        else:
            softmax_a_config = a_qconfig
        if ptq4sam_config.BIG:
            sign_a_config = update_specialized_quantizer_config(a_qconfig, 'bimodal')
        else:
            sign_a_config = a_qconfig

        self.q_proj = PreQuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, proj_type)
        self.k_proj = PreQuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, proj_type)
        self.v_proj = PreQuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, proj_type)
        self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)

        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, sign_a_config)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

        self.dropout_p = org_module.dropout_p

        if ptq4sam_config.BIG:
            self.k_post_act_fake_quantize.global_num = ptq4sam_config.global_num
            self.k_post_act_fake_quantize.peak_distance = ptq4sam_config.peak_distance
            self.k_post_act_fake_quantize.peak_height = ptq4sam_config.peak_height

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        d_k = q.size(-1)

        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = attn_scores.softmax(dim=-1)

        attn_weights = self.softmax_post_act_fake_quantize(attn_weights, value=v)

        out = torch.matmul(attn_weights, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

    def bimodal_adjust(self):
        if self.k_post_act_fake_quantize.is_bimodal:
            sign = self.k_post_act_fake_quantize.sign

            def addjust_linear(linear: torch.nn.Linear, sign):
                linear.weight.mul_(sign.unsqueeze(1))
                linear.bias.mul_(sign)

            addjust_linear(self.k_proj.module, sign)
            addjust_linear(self.q_proj.module, sign)
            self.k_post_act_fake_quantize.is_bimodal = False


class QuantRoPEAttention(QuantizedBlock):
    def __init__(self, org_module: RoPEAttention, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()

        self.compute_cis = org_module.compute_cis
        self.freqs_cis = org_module.freqs_cis
        self.rope_k_repeat = org_module.rope_k_repeat

        self.num_heads = org_module.num_heads

        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax_agq')
        else:
            softmax_a_config = a_qconfig

        self.q_proj = PreQuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig)
        self.k_proj = PreQuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig)
        self.v_proj = PreQuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig)
        self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)

        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # Attention
        d_k = q.size(-1)

        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = attn_scores.softmax(dim=-1)

        attn_weights = self.softmax_post_act_fake_quantize(attn_weights, value=v)

        out = torch.matmul(attn_weights, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class QuantMemoryAttentionLayer(nn.Module):
    def __init__(self, org_module: MemoryAttentionLayer, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()
        self.d_model = org_module.d_model
        self.dim_feedforward = org_module.dim_feedforward
        self.dropout_value = org_module.dropout_value

        self.self_attn = QuantRoPEAttention(org_module.self_attn, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)
        self.cross_attn_image = QuantRoPEAttention(org_module.cross_attn_image, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)

        # Implementation of Feedforward model
        self.linear1 = PreQuantizedLayer(org_module.linear1, None, w_qconfig, a_qconfig)
        self.dropout = org_module.dropout
        self.linear2 = PreQuantizedLayer(org_module.linear2, None, w_qconfig, a_qconfig)

        self.norm1 = org_module.norm1
        self.norm2 = org_module.norm2
        self.norm3 = org_module.norm3
        self.dropout1 = org_module.dropout1
        self.dropout2 = org_module.dropout2
        self.dropout3 = org_module.dropout3

        self.activation_str = org_module.activation_str
        self.activation = org_module.activation

        # Where to add pos enc
        self.pos_enc_at_attn = org_module.pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = org_module.pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = org_module.pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, (RoPEAttention, QuantRoPEAttention))
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class QuantMemoryAttention(nn.Module):
    def __init__(self, org_module: MemoryAttention, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()
        self.d_model = org_module.d_model
        self.layers = nn.ModuleList(
            QuantMemoryAttentionLayer(layer, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)
            for layer in org_module.layers
        )
        self.num_layers = org_module.num_layers
        self.norm = org_module.norm
        self.pos_enc_at_input = org_module.pos_enc_at_input
        self.batch_first = org_module.batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, (RoPEAttention, QuantRoPEAttention)):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output


class QuantTwoWayAttentionBlock(nn.Module):
    def __init__(self, org_module: TwoWayAttentionBlock, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()
        self.self_attn = QuantAttention(org_module.self_attn, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)
        self.norm1 = org_module.norm1

        self.cross_attn_token_to_image = QuantAttention(org_module.cross_attn_token_to_image, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)
        self.norm2 = org_module.norm2

        self.mlp = QuantDecoderMLP(org_module.mlp, w_qconfig, a_qconfig, ahcqsam_config)
        self.norm3 = org_module.norm3

        self.norm4 = org_module.norm4
        self.cross_attn_image_to_token = QuantAttention(org_module.cross_attn_image_to_token, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)

        self.skip_first_layer_pe = org_module.skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class QuantCXBlock(QuantizedBlock):
    def __init__(self, org_module: CXBlock, w_qconfig, a_qconfig):
        super().__init__()

        self.dwconv = PreQuantizedLayer(org_module.dwconv, None, w_qconfig, a_qconfig)
        self.norm = org_module.norm
        self.pwconv1 = PreQuantizedLayer(org_module.pwconv1, None, w_qconfig, a_qconfig)
        self.act = org_module.act
        self.pwconv2 = PreQuantizedLayer(org_module.pwconv2, None, w_qconfig, a_qconfig)
        self.gamma = org_module.gamma
        self.drop_path = org_module.drop_path

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class QuantMaskDownSampler(QuantizedBlock):
    def __init__(self, org_module: MaskDownSampler, w_qconfig, a_qconfig):
        super().__init__()

        self.encoder = nn.Sequential()
        for name, m in org_module.encoder.named_children():
            if isinstance(m, nn.Conv2d):
                q_conv = PreQuantizedLayer(m, None, w_qconfig, a_qconfig)
                self.encoder.add_module(name, q_conv)
            else:
                self.encoder.add_module(name, m)

    def forward(self, x):
        return self.encoder(x)


class QuantFuser(nn.Module):
    def __init__(self, org_module: Fuser, w_qconfig, a_qconfig):
        super().__init__()
        self.proj = org_module.proj
        self.layers = nn.ModuleList(
            QuantCXBlock(layer, w_qconfig, a_qconfig)
            for layer in org_module.layers
        )

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class QuantMemoryEncoder(nn.Module):
    def __init__(self, org_module: MemoryEncoder, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        super().__init__()

        self.mask_downsampler = QuantMaskDownSampler(org_module.mask_downsampler, w_qconfig, a_qconfig)

        self.pix_feat_proj = PreQuantizedLayer(org_module.pix_feat_proj, None, w_qconfig, a_qconfig)
        self.fuser = QuantFuser(org_module.fuser, w_qconfig, a_qconfig)
        self.position_encoding = org_module.position_encoding
        if isinstance(org_module.out_proj, nn.Conv2d):
            self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)
        else:
            self.out_proj = org_module.out_proj

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


specials = {
    MultiScaleBlock: QuantMultiScaleBlock,
    FpnNeck: QuantFpnNeck,
    TwoWayAttentionBlock: QuantTwoWayAttentionBlock,
    Attention: QuantAttention,
    MemoryAttention: QuantMemoryAttention,
    MemoryEncoder: QuantMemoryEncoder
}


def bimodal_adjust(model):
    print('Start to Detect Bimodal Distribution...')
    for name,m in model.named_modules():
        if isinstance(m, QuantAttention) and 'cross_attn_token_to_image' in name:
            print(name)
            print(m.k_post_act_fake_quantize.is_bimodal)
            m.bimodal_adjust()
    print('Bimodal Integration End...')
