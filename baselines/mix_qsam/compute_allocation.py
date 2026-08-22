from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from mmdet.datasets import build_dataloader, build_dataset
from tqdm import tqdm

MIX_QSAM_ROOT = Path(__file__).resolve().parent
if str(MIX_QSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MIX_QSAM_ROOT))

from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import (  # noqa: E402
    Attention as EncoderAttention,
    add_decomposed_rel_pos,
)
from projects.instance_segment_anything.models.segment_anything.modeling.transformer import (  # noqa: E402
    Attention as DecoderAttention,
)
import utils


@dataclass
class LayerUnit:
    name: str
    module: nn.Module | None
    module_type: str
    num_params: int
    macs: float = 0.0
    hook_type: str = "module"
    op_name: str | None = None


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def load_calibration_cpu(cfg, num_samples: int) -> list[Any]:
    train_loader_cfg = {
        "samples_per_gpu": 1,
        "workers_per_gpu": 2,
        "num_gpus": len(cfg.gpu_ids),
        "dist": False,
        "seed": 6769,
        "shuffle": False,
        **cfg.data.get("train_dataloader", {}),
    }
    train_loader_cfg["samples_per_gpu"] = 1

    train_data = build_dataset(cfg.data.train)
    train_loader = build_dataloader(train_data, **train_loader_cfg)

    cali_data = []
    for data_batch in train_loader:
        cali_data.append(
            {
                "img": [data_batch["img"][0].detach().cpu()],
                "img_metas": data_batch["img_metas"][0].data,
            }
        )
        if len(cali_data) == num_samples:
            break

    print(f"Loaded {len(cali_data)} calibration images on CPU.")
    return cali_data


def cap_detector_boxes(model: nn.Module, max_boxes_per_image: int | None) -> None:
    if max_boxes_per_image is None or max_boxes_per_image <= 0:
        return

    original_simple_test = model.det_model.simple_test

    def simple_test_with_box_cap(*args, **kwargs):
        results = original_simple_test(*args, **kwargs)
        for result in results:
            boxes = result.get("boxes")
            if boxes is None or boxes.shape[0] <= max_boxes_per_image:
                continue
            scores = result.get("scores")
            if scores is not None:
                indices = torch.topk(scores, k=max_boxes_per_image).indices
            else:
                indices = torch.arange(
                    max_boxes_per_image,
                    device=boxes.device,
                    dtype=torch.long,
                )
            for key in ("boxes", "scores", "labels"):
                value = result.get(key)
                if torch.is_tensor(value) and value.shape[0] == boxes.shape[0]:
                    result[key] = value[indices]
        return results

    model.det_model.simple_test = simple_test_with_box_cap


def discover_layer_units(model: nn.Module, max_units: int | None = None) -> list[LayerUnit]:
    sam = model.predictor.model
    units: list[LayerUnit] = []

    for idx, block in enumerate(sam.image_encoder.blocks):
        attn_prefix = f"image_encoder.blocks.{idx}.attn"
        units.extend(make_encoder_attention_units(attn_prefix, block.attn))
        mlp_prefix = f"image_encoder.blocks.{idx}.mlp"
        units.append(make_unit(f"{mlp_prefix}.lin1", block.mlp.lin1))
        units.append(make_unit(f"{mlp_prefix}.lin2", block.mlp.lin2))

    units.append(make_unit("image_encoder.neck.0", sam.image_encoder.neck[0]))
    units.append(make_unit("image_encoder.neck.2", sam.image_encoder.neck[2]))

    for name, module in sam.mask_decoder.transformer.named_modules():
        if isinstance(module, DecoderAttention):
            units.extend(make_decoder_attention_units(f"mask_decoder.transformer.{name}", module))
        elif type(module).__name__ == "MLPBlock":
            prefix = f"mask_decoder.transformer.{name}"
            units.append(make_unit(f"{prefix}.lin1", module.lin1))
            units.append(make_unit(f"{prefix}.lin2", module.lin2))

    if max_units is not None:
        units = units[:max_units]
    return units


def make_encoder_attention_units(prefix: str, module: EncoderAttention) -> list[LayerUnit]:
    return [
        make_unit(f"{prefix}.qkv", module.qkv),
        make_attention_op_unit(f"{prefix}.q", module, "encoder_attention", "q"),
        make_attention_op_unit(f"{prefix}.k", module, "encoder_attention", "k"),
        make_attention_op_unit(f"{prefix}.v", module, "encoder_attention", "v"),
        make_attention_op_unit(f"{prefix}.matmul1", module, "encoder_attention", "matmul1"),
        make_attention_op_unit(f"{prefix}.softmax", module, "encoder_attention", "softmax"),
        make_attention_op_unit(f"{prefix}.matmul2", module, "encoder_attention", "matmul2"),
        make_unit(f"{prefix}.proj", module.proj),
    ]


def make_decoder_attention_units(prefix: str, module: DecoderAttention) -> list[LayerUnit]:
    return [
        make_unit(f"{prefix}.q_proj", module.q_proj),
        make_unit(f"{prefix}.k_proj", module.k_proj),
        make_unit(f"{prefix}.v_proj", module.v_proj),
        make_attention_op_unit(f"{prefix}.q", module, "decoder_attention", "q"),
        make_attention_op_unit(f"{prefix}.k", module, "decoder_attention", "k"),
        make_attention_op_unit(f"{prefix}.v", module, "decoder_attention", "v"),
        make_attention_op_unit(f"{prefix}.matmul1", module, "decoder_attention", "matmul1"),
        make_attention_op_unit(f"{prefix}.softmax", module, "decoder_attention", "softmax"),
        make_attention_op_unit(f"{prefix}.matmul2", module, "decoder_attention", "matmul2"),
        make_unit(f"{prefix}.out_proj", module.out_proj),
    ]


def make_unit(name: str, module: nn.Module) -> LayerUnit:
    num_params = sum(p.numel() for p in module.parameters())
    return LayerUnit(
        name=name,
        module=module,
        module_type=type(module).__name__,
        num_params=num_params,
    )


def make_attention_op_unit(
    name: str,
    module: nn.Module,
    hook_type: str,
    op_name: str,
) -> LayerUnit:
    return LayerUnit(
        name=name,
        module=module,
        module_type=f"{type(module).__name__}.{op_name}",
        num_params=0,
        macs=0.0,
        hook_type=hook_type,
        op_name=op_name,
    )


def zero_like_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, tuple):
        return tuple(zero_like_tree(item) for item in value)
    if isinstance(value, list):
        return [zero_like_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: zero_like_tree(item) for key, item in value.items()}
    return value


class RemovableHook:
    def __init__(self, remove_fn):
        self.remove_fn = remove_fn

    def remove(self) -> None:
        self.remove_fn()


def remove_hooks(handles: list[RemovableHook]) -> None:
    for handle in handles:
        handle.remove()


def register_mac_hooks(units: list[LayerUnit], mac_sums: dict[str, float]) -> list[RemovableHook]:
    handles: list[RemovableHook] = []
    module_units: dict[int, list[LayerUnit]] = {}
    for unit in units:
        if unit.module is None:
            continue
        module_units.setdefault(id(unit.module), []).append(unit)

    registered_module_ids: set[int] = set()
    for grouped_units in module_units.values():
        module = grouped_units[0].module
        if module is None or id(module) in registered_module_ids:
            continue
        registered_module_ids.add(id(module))

        if isinstance(module, nn.Linear):
            names = [unit.name for unit in grouped_units if unit.hook_type == "module"]

            def linear_hook(mod, _inputs, output, names=names):
                macs = float(output.numel() * mod.in_features)
                for name in names:
                    mac_sums[name] = mac_sums.get(name, 0.0) + macs

            handles.append(RemovableHook(module.register_forward_hook(linear_hook).remove))
        elif isinstance(module, nn.Conv2d):
            names = [unit.name for unit in grouped_units if unit.hook_type == "module"]

            def conv_hook(mod, _inputs, output, names=names):
                kernel_ops = (
                    mod.kernel_size[0]
                    * mod.kernel_size[1]
                    * mod.in_channels
                    / mod.groups
                )
                macs = float(output.numel() * kernel_ops)
                for name in names:
                    mac_sums[name] = mac_sums.get(name, 0.0) + macs

            handles.append(RemovableHook(module.register_forward_hook(conv_hook).remove))
        elif isinstance(module, EncoderAttention):
            op_units = [unit for unit in grouped_units if unit.hook_type == "encoder_attention"]

            def encoder_attention_hook(mod, inputs, _output, op_units=op_units):
                if not inputs:
                    return
                x = inputs[0]
                B, H, W, C = x.shape
                tokens = H * W
                head_dim = C // mod.num_heads
                matmul_macs = float(B * mod.num_heads * tokens * tokens * head_dim)
                add_attention_op_macs(mac_sums, op_units, matmul_macs, matmul_macs)

            handles.append(RemovableHook(module.register_forward_hook(encoder_attention_hook).remove))
        elif isinstance(module, DecoderAttention):
            op_units = [unit for unit in grouped_units if unit.hook_type == "decoder_attention"]

            def decoder_attention_hook(mod, inputs, kwargs, _output, op_units=op_units):
                if len(inputs) >= 3:
                    q, k = inputs[:2]
                else:
                    q = kwargs.get("q")
                    k = kwargs.get("k")
                if q is None or k is None:
                    return
                batch = q.shape[0]
                query_tokens = q.shape[1]
                key_tokens = k.shape[1]
                head_dim = mod.internal_dim // mod.num_heads
                matmul_macs = float(
                    batch * mod.num_heads * query_tokens * key_tokens * head_dim
                )
                add_attention_op_macs(mac_sums, op_units, matmul_macs, matmul_macs)

            try:
                hook = module.register_forward_hook(decoder_attention_hook, with_kwargs=True)
            except TypeError:
                def positional_decoder_attention_hook(mod, inputs, output, op_units=op_units):
                    decoder_attention_hook(mod, inputs, {}, output, op_units)

                hook = module.register_forward_hook(positional_decoder_attention_hook)
            handles.append(RemovableHook(hook.remove))

    return handles


def add_attention_op_macs(
    mac_sums: dict[str, float],
    op_units: list[LayerUnit],
    matmul1_macs: float,
    matmul2_macs: float,
) -> None:
    for unit in op_units:
        if unit.op_name in {"q", "k"}:
            macs = matmul1_macs / 2.0
        elif unit.op_name == "matmul1":
            macs = matmul1_macs
        elif unit.op_name in {"v", "softmax"}:
            macs = matmul2_macs / 2.0
        elif unit.op_name == "matmul2":
            macs = matmul2_macs
        else:
            macs = 0.0
        mac_sums[unit.name] = mac_sums.get(unit.name, 0.0) + macs


def register_zero_hook(unit: LayerUnit) -> RemovableHook:
    if unit.hook_type == "module":
        if unit.module is None:
            raise RuntimeError(f"Layer unit {unit.name} has no module.")
        handle = unit.module.register_forward_hook(
            lambda _module, _inputs, output: zero_like_tree(output)
        )
        return RemovableHook(handle.remove)
    if unit.hook_type == "encoder_attention":
        return patch_encoder_attention(unit)
    if unit.hook_type == "decoder_attention":
        return patch_decoder_attention(unit)
    raise RuntimeError(f"Unsupported layer unit hook type: {unit.hook_type}")


def patch_encoder_attention(unit: LayerUnit) -> RemovableHook:
    module = unit.module
    if not isinstance(module, EncoderAttention):
        raise RuntimeError(f"{unit.name} is not an encoder attention module.")
    original_forward = module.forward
    op_name = unit.op_name

    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        if op_name == "q":
            q = torch.zeros_like(q)
        elif op_name == "k":
            k = torch.zeros_like(k)
        elif op_name == "v":
            v = torch.zeros_like(v)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        if op_name == "matmul1":
            attn = torch.zeros_like(attn)

        attn = attn.softmax(dim=-1)
        if op_name == "softmax":
            attn = torch.zeros_like(attn)

        out = attn @ v
        if op_name == "matmul2":
            out = torch.zeros_like(out)
        out = out.view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return module.proj(out)

    module.forward = patched_forward
    return RemovableHook(lambda: setattr(module, "forward", original_forward))


def patch_decoder_attention(unit: LayerUnit) -> RemovableHook:
    module = unit.module
    if not isinstance(module, DecoderAttention):
        raise RuntimeError(f"{unit.name} is not a decoder attention module.")
    original_forward = module.forward
    op_name = unit.op_name

    def patched_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = module.q_proj(q)
        k = module.k_proj(k)
        v = module.v_proj(v)
        if op_name == "q":
            q = torch.zeros_like(q)
        elif op_name == "k":
            k = torch.zeros_like(k)
        elif op_name == "v":
            v = torch.zeros_like(v)

        q = module._separate_heads(q, module.num_heads)
        k = module._separate_heads(k, module.num_heads)
        v = module._separate_heads(v, module.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        if op_name == "matmul1":
            attn = torch.zeros_like(attn)
        attn = torch.softmax(attn, dim=-1)
        if op_name == "softmax":
            attn = torch.zeros_like(attn)

        out = attn @ v
        if op_name == "matmul2":
            out = torch.zeros_like(out)
        out = module._recombine_heads(out)
        return module.out_proj(out)

    module.forward = patched_forward
    return RemovableHook(lambda: setattr(module, "forward", original_forward))


def pick_mask_tensor(output: Any) -> torch.Tensor | None:
    if (
        isinstance(output, (tuple, list))
        and len(output) >= 3
        and torch.is_tensor(output[2])
        and output[2].numel() > 0
    ):
        return output[2]

    tensors: list[torch.Tensor] = []

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(output)
    tensors = [tensor for tensor in tensors if tensor.numel() > 0]
    if not tensors:
        return None

    mask_like = [tensor for tensor in tensors if tensor.ndim >= 3]
    candidates = mask_like if mask_like else tensors
    return max(candidates, key=lambda tensor: tensor.numel())


def output_distribution(output: Any, eps: float) -> torch.Tensor | None:
    tensor = pick_mask_tensor(output)
    if tensor is None:
        return None
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    values = torch.sigmoid(tensor).flatten()
    total = values.sum()
    if not torch.isfinite(total) or total <= eps:
        return None
    return (values / total.clamp_min(eps)).clamp_min(eps)


def kl_divergence(reference: torch.Tensor, perturbed: torch.Tensor, eps: float) -> float:
    if reference.numel() != perturbed.numel():
        count = min(reference.numel(), perturbed.numel())
        reference = reference[:count]
        perturbed = perturbed[:count]
    perturbed = perturbed.clamp_min(eps)
    score = torch.sum(reference * (torch.log(reference) - torch.log(perturbed)))
    if not torch.isfinite(score):
        return 0.0
    return float(score.detach().cpu())


@torch.no_grad()
def forward_distribution(model: nn.Module, data: Any, eps: float) -> torch.Tensor | None:
    output = model.extract_feat(data)
    distribution = output_distribution(output, eps)
    del output
    if hasattr(model, "predictor") and hasattr(model.predictor, "reset_image"):
        model.predictor.reset_image()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return distribution


@torch.no_grad()
def collect_scores(
    model: nn.Module,
    cali_data: list[Any],
    units: list[LayerUnit],
    eps: float,
    device: torch.device,
) -> dict[str, Any]:
    per_image_scores: list[list[float]] = []
    skipped_images = 0
    mac_sums = {unit.name: 0.0 for unit in units}

    for cpu_data in tqdm(cali_data, desc="Collect FP MPQ scores", unit="image"):
        data = move_to_device(cpu_data, device)
        mac_handles = register_mac_hooks(units, mac_sums)
        try:
            reference = forward_distribution(model, data, eps)
        finally:
            remove_hooks(mac_handles)
        if reference is None:
            skipped_images += 1
            del data
            continue

        image_scores: list[float] = []
        for unit in tqdm(units, desc="Zero-out layers", unit="layer", leave=False):
            handle = register_zero_hook(unit)
            try:
                perturbed = forward_distribution(model, data, eps)
            finally:
                handle.remove()

            if perturbed is None:
                image_scores.append(0.0)
            else:
                image_scores.append(kl_divergence(reference, perturbed, eps))
            del perturbed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        score_sum = sum(image_scores)
        if score_sum > eps:
            image_scores = [score / score_sum for score in image_scores]
        per_image_scores.append(image_scores)
        del reference
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not per_image_scores:
        raise RuntimeError("No valid calibration image produced SAM mask distributions.")

    score_tensor = torch.tensor(per_image_scores, dtype=torch.float64)
    importance = score_tensor.mean(dim=0).tolist()
    macs = [mac_sums[unit.name] / len(per_image_scores) for unit in units]
    for unit, unit_macs in zip(units, macs):
        unit.macs = unit_macs
    if max(importance) <= eps:
        print(
            "WARNING: all collected importance scores are zero. "
            "The bit allocation is not meaningful; check the selected output distribution "
            "or the zero-out hooks."
        )
    synergy = compute_synergy(score_tensor, eps)
    return {
        "unit_names": [unit.name for unit in units],
        "num_images": len(per_image_scores),
        "skipped_images": skipped_images,
        "per_image_scores": per_image_scores,
        "importance": importance,
        "synergy": synergy,
        "macs": macs,
    }


def compute_synergy(score_tensor: torch.Tensor, eps: float) -> list[float]:
    if score_tensor.shape[1] < 2:
        return []
    values: list[float] = []
    for idx in range(score_tensor.shape[1] - 1):
        diff = torch.abs(score_tensor[:, idx] - score_tensor[:, idx + 1])
        value = torch.mean(1.0 / diff.clamp_min(eps))
        values.append(float(torch.log1p(value).cpu()))
    return values


def solve_bit_allocation(
    units: list[LayerUnit],
    importance: list[float],
    synergy: list[float],
    candidate_bits: list[int],
    target_bits: int,
    lambda_btr: float,
    solver: str = "cvxpy",
    use_bitops: bool = True,
    cvxpy_solver: str | None = None,
    cvxpy_verbose: bool = False,
    solver_time_limit: float | None = None,
    min_budget_ratio: float = 0.0,
) -> tuple[list[int], dict[str, Any]]:
    candidate_bits = sorted(set(int(bit) for bit in candidate_bits))
    min_bit = candidate_bits[0]
    total_params = sum(unit.num_params for unit in units)
    total_macs = sum(unit.macs for unit in units)
    budget = total_params * target_bits
    bitops_budget = total_macs * target_bits * target_bits if use_bitops else math.inf
    min_budget = total_params * min_bit

    metadata: dict[str, Any] = {
        "target_budget_param_bits": budget,
        "target_budget_bitops": bitops_budget,
        "min_budget_param_bits": min_budget,
        "total_params": total_params,
        "total_macs": total_macs,
        "use_bitops": use_bitops,
        "min_budget_ratio": min_budget_ratio,
        "requested_solver": solver,
        "requested_cvxpy_solver": cvxpy_solver,
        "warning": None,
    }

    if target_bits < min_bit:
        metadata["warning"] = (
            f"target_bits={target_bits} is below the minimum candidate bit {min_bit}; "
            "using the minimum candidate bit for every unit."
        )
        return [min_bit for _ in units], metadata

    if target_bits == min_bit:
        metadata["warning"] = (
            f"target_bits={target_bits} equals the minimum candidate bit {min_bit}; "
            "the parameter-bit budget only permits the all-min-bit allocation."
        )
        return [min_bit for _ in units], metadata

    if solver == "cvxpy":
        cvxpy_solution = solve_bit_allocation_cvxpy(
            units=units,
            importance=importance,
            synergy=synergy,
            candidate_bits=candidate_bits,
            budget=budget,
            bitops_budget=bitops_budget,
            lambda_btr=lambda_btr,
            use_bitops=use_bitops,
            preferred_solver=cvxpy_solver,
            verbose=cvxpy_verbose,
            time_limit=solver_time_limit,
            min_budget_ratio=min_budget_ratio,
        )
        if cvxpy_solution is None:
            raise RuntimeError("CVXPY mixed-integer solver is unavailable or failed.")
        return cvxpy_solution

    dp_solution = solve_bit_allocation_lagrangian_dp(
        units=units,
        importance=importance,
        synergy=synergy,
        candidate_bits=candidate_bits,
        budget=budget,
        lambda_btr=lambda_btr,
    )
    if dp_solution is not None:
        bits, local_metadata = dp_solution
    else:
        bits, local_metadata = solve_bit_allocation_exchange(
            units=units,
            importance=importance,
            synergy=synergy,
            candidate_bits=candidate_bits,
            target_bits=target_bits,
            budget=budget,
            lambda_btr=lambda_btr,
        )
    metadata.update(local_metadata)
    return bits, metadata


def solve_bit_allocation_lagrangian_dp(
    units: list[LayerUnit],
    importance: list[float],
    synergy: list[float],
    candidate_bits: list[int],
    budget: int,
    lambda_btr: float,
) -> tuple[list[int], dict[str, Any]] | None:
    if not units:
        return None

    mu_values = [0.0]
    mu_values.extend(step * 1e-9 for step in range(-200, 201))
    mu_values.extend(step * 5e-9 for step in range(-200, 201))
    mu_values.extend(step * 1e-8 for step in range(-200, 201))

    best_bits = None
    best_used_budget = 0
    best_objective = -math.inf
    best_mu = None

    for mu in sorted(set(mu_values)):
        bits = solve_lagrangian_chain(
            units=units,
            importance=importance,
            synergy=synergy,
            candidate_bits=candidate_bits,
            lambda_btr=lambda_btr,
            mu=mu,
        )
        used_budget = sum(unit.num_params * bit for unit, bit in zip(units, bits))
        if used_budget > budget:
            continue
        objective = allocation_objective(bits, importance, synergy, lambda_btr)
        if objective > best_objective:
            best_bits = bits
            best_used_budget = used_budget
            best_objective = objective
            best_mu = mu

    if best_bits is None:
        return None

    return best_bits, {
        "solver": "lagrangian_chain_dp",
        "used_budget_param_bits": best_used_budget,
        "max_candidate_bit": candidate_bits[-1],
        "objective": best_objective,
        "lagrangian_mu": best_mu,
    }


def solve_lagrangian_chain(
    units: list[LayerUnit],
    importance: list[float],
    synergy: list[float],
    candidate_bits: list[int],
    lambda_btr: float,
    mu: float,
) -> list[int]:
    layer_count = len(units)
    bit_count = len(candidate_bits)
    dp: list[list[tuple[float, int | None]]] = [
        [(-math.inf, None) for _ in range(bit_count)] for _ in range(layer_count)
    ]

    for bit_idx, bit in enumerate(candidate_bits):
        dp[0][bit_idx] = (
            importance[0] * bit - mu * units[0].num_params * bit,
            None,
        )

    for layer_idx in range(1, layer_count):
        for bit_idx, bit in enumerate(candidate_bits):
            best_value = -math.inf
            best_prev_idx = None
            for prev_bit_idx, prev_bit in enumerate(candidate_bits):
                value = (
                    dp[layer_idx - 1][prev_bit_idx][0]
                    + importance[layer_idx] * bit
                    - lambda_btr * synergy[layer_idx - 1] * abs(prev_bit - bit)
                    - mu * units[layer_idx].num_params * bit
                )
                if value > best_value:
                    best_value = value
                    best_prev_idx = prev_bit_idx
            dp[layer_idx][bit_idx] = (best_value, best_prev_idx)

    bit_idx = max(range(bit_count), key=lambda idx: dp[-1][idx][0])
    bits: list[int] = []
    for layer_idx in range(layer_count - 1, -1, -1):
        bits.append(candidate_bits[bit_idx])
        prev_idx = dp[layer_idx][bit_idx][1]
        if prev_idx is not None:
            bit_idx = prev_idx
    bits.reverse()
    return bits


def solve_bit_allocation_exchange(
    units: list[LayerUnit],
    importance: list[float],
    synergy: list[float],
    candidate_bits: list[int],
    target_bits: int,
    budget: int,
    lambda_btr: float,
) -> tuple[list[int], dict[str, Any]]:
    if target_bits in candidate_bits:
        initial_bit = target_bits
    else:
        initial_bit = max(bit for bit in candidate_bits if bit <= target_bits)

    bits = [initial_bit for _ in units]
    used_budget = sum(unit.num_params * bit for unit, bit in zip(units, bits))

    if used_budget > budget:
        bits = [candidate_bits[0] for _ in units]
        used_budget = sum(unit.num_params * bit for unit, bit in zip(units, bits))

    iterations = 0
    while True:
        best_bits = None
        best_gain = 0.0
        best_budget = used_budget
        current_obj = allocation_objective(bits, importance, synergy, lambda_btr)

        for idx, bit in enumerate(bits):
            for next_bit in candidate_bits:
                if next_bit <= bit:
                    continue
                trial_budget = used_budget + units[idx].num_params * (next_bit - bit)
                if trial_budget > budget:
                    continue
                trial = bits.copy()
                trial[idx] = next_bit
                gain = allocation_objective(trial, importance, synergy, lambda_btr) - current_obj
                if gain > best_gain:
                    best_bits = trial
                    best_gain = gain
                    best_budget = trial_budget

        for down_idx, down_bit in enumerate(bits):
            lower_bits = [bit for bit in candidate_bits if bit < down_bit]
            if not lower_bits:
                continue
            for lowered_bit in lower_bits:
                freed = units[down_idx].num_params * (down_bit - lowered_bit)
                for up_idx, up_bit in enumerate(bits):
                    if up_idx == down_idx:
                        continue
                    higher_bits = [bit for bit in candidate_bits if bit > up_bit]
                    for raised_bit in higher_bits:
                        cost = units[up_idx].num_params * (raised_bit - up_bit)
                        trial_budget = used_budget - freed + cost
                        if trial_budget > budget:
                            continue
                        trial = bits.copy()
                        trial[down_idx] = lowered_bit
                        trial[up_idx] = raised_bit
                        gain = (
                            allocation_objective(trial, importance, synergy, lambda_btr)
                            - current_obj
                        )
                        if gain > best_gain:
                            best_bits = trial
                            best_gain = gain
                            best_budget = trial_budget

        if best_bits is None:
            break
        bits = best_bits
        used_budget = best_budget
        iterations += 1

    return bits, {
        "solver": "exchange_local_search",
        "used_budget_param_bits": used_budget,
        "max_candidate_bit": candidate_bits[-1],
        "objective": allocation_objective(bits, importance, synergy, lambda_btr),
        "exchange_iterations": iterations,
    }


def solve_bit_allocation_cvxpy(
    units: list[LayerUnit],
    importance: list[float],
    synergy: list[float],
    candidate_bits: list[int],
    budget: int,
    bitops_budget: float,
    lambda_btr: float,
    use_bitops: bool,
    preferred_solver: str | None,
    verbose: bool,
    time_limit: float | None,
    min_budget_ratio: float,
) -> tuple[list[int], dict[str, Any]] | None:
    try:
        import cvxpy as cp
    except ImportError:
        return None

    layer_count = len(units)
    bit_count = len(candidate_bits)
    alpha = cp.Variable((layer_count, bit_count), boolean=True)
    bit_values = cp.Constant(candidate_bits)
    bit_square_values = cp.Constant([bit * bit for bit in candidate_bits])
    assigned_bits = alpha @ bit_values
    assigned_bit_squares = alpha @ bit_square_values

    importance_term = cp.sum(cp.multiply(importance, assigned_bits))
    if layer_count > 1:
        transition = cp.Variable(layer_count - 1)
        transition_constraints = [
            transition >= assigned_bits[:-1] - assigned_bits[1:],
            transition >= assigned_bits[1:] - assigned_bits[:-1],
            transition >= 0,
        ]
        transition_term = cp.sum(cp.multiply(synergy, transition))
    else:
        transition_constraints = []
        transition_term = 0

    params = [unit.num_params for unit in units]
    macs = [unit.macs for unit in units]
    param_scale = max(float(budget), 1.0)
    bitops_scale = max(float(bitops_budget), 1.0)
    scaled_params = [param / param_scale for param in params]
    scaled_macs = [mac / bitops_scale for mac in macs]
    constraints = [
        cp.sum(alpha, axis=1) == 1,
        cp.sum(cp.multiply(scaled_params, assigned_bits)) <= 1.0,
        *transition_constraints,
    ]
    if min_budget_ratio > 0:
        constraints.append(cp.sum(cp.multiply(scaled_params, assigned_bits)) >= min_budget_ratio)
    if use_bitops:
        constraints.append(cp.sum(cp.multiply(scaled_macs, assigned_bit_squares)) <= 1.0)
        if min_budget_ratio > 0:
            constraints.append(
                cp.sum(cp.multiply(scaled_macs, assigned_bit_squares)) >= min_budget_ratio
            )
    problem = cp.Problem(
        cp.Maximize(importance_term - lambda_btr * transition_term),
        constraints,
    )

    installed_solvers = cp.installed_solvers()
    print(f"Installed CVXPY solvers: {installed_solvers}")
    if preferred_solver:
        solver_order = [preferred_solver]
    else:
        solver_order = ["CBC", "GLPK_MI", "SCIP", "ECOS_BB"]

    solver_failures = []
    for solver in solver_order:
        if solver not in cp.installed_solvers():
            solver_failures.append(f"{solver}: not installed")
            continue
        print(f"Solving bit allocation with CVXPY solver {solver}...")
        try:
            solve_kwargs = {"solver": solver, "verbose": verbose}
            if solver == "ECOS_BB" and time_limit is not None:
                solve_kwargs["mi_max_iters"] = 1000000
            elif solver == "ECOS_BB":
                solve_kwargs["mi_max_iters"] = 1000000
            elif solver == "CBC" and time_limit is not None:
                solve_kwargs["maximumSeconds"] = time_limit
            elif solver == "SCIP" and time_limit is not None:
                solve_kwargs["scip_params"] = {"limits/time": time_limit}
            elif solver == "GLPK_MI" and time_limit is not None:
                print("WARNING: GLPK_MI time limit is not configured by this script.")
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as err:
            solver_failures.append(f"{solver}: SolverError: {err}")
            continue
        solver_failures.append(f"{solver}: status={problem.status}, value={problem.value}")
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and alpha.value is not None:
            bit_indices = alpha.value.argmax(axis=1)
            bits = [candidate_bits[int(idx)] for idx in bit_indices]
            used_budget = sum(unit.num_params * bit for unit, bit in zip(units, bits))
            used_bitops = sum(unit.macs * bit * bit for unit, bit in zip(units, bits))
            metadata = {
                "solver": f"cvxpy:{solver}",
                "status": problem.status,
                "cvxpy_verbose": verbose,
                "solver_time_limit": time_limit,
                "min_budget_ratio": min_budget_ratio,
                "target_budget_param_bits": budget,
                "used_budget_param_bits": used_budget,
                "target_budget_bitops": bitops_budget,
                "used_budget_bitops": used_bitops,
                "total_params": sum(params),
                "total_macs": sum(macs),
                "objective": float(problem.value),
                "warning": None,
            }
            return bits, metadata
    if solver_failures:
        print("CVXPY solver attempts failed:")
        for failure in solver_failures:
            print(f"  {failure}")
    return None

def next_candidate_bit(current: int, candidate_bits: list[int]) -> int | None:
    for bit in candidate_bits:
        if bit > current:
            return bit
    return None


def allocation_objective(
    bits: list[int],
    importance: list[float],
    synergy: list[float],
    lambda_btr: float,
) -> float:
    precision_term = sum(score * bit for score, bit in zip(importance, bits))
    transition_term = sum(
        score * abs(bits[idx] - bits[idx + 1]) for idx, score in enumerate(synergy)
    )
    return precision_term - lambda_btr * transition_term


def load_or_collect_scores(
    model: nn.Module,
    cali_data: list[Any],
    units: list[LayerUnit],
    eps: float,
    score_cache: str | None,
    device: torch.device,
) -> dict[str, Any]:
    if score_cache and os.path.exists(score_cache):
        with open(score_cache) as f:
            scores = json.load(f)
        cached_names = scores.get("unit_names")
        unit_names = [unit.name for unit in units]
        cached_macs = scores.get("macs")
        if cached_names == unit_names and cached_macs is not None:
            for unit, unit_macs in zip(units, cached_macs):
                unit.macs = unit_macs
            return scores
        print("Score cache is missing current unit names or MACs; recollect scores.")

    scores = collect_scores(model, cali_data, units, eps, device)
    if score_cache:
        Path(score_cache).parent.mkdir(parents=True, exist_ok=True)
        with open(score_cache, "w") as f:
            json.dump(scores, f, indent=2)
    return scores
