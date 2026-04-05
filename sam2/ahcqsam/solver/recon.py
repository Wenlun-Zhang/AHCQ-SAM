import numpy as np
from PIL import Image
import inspect
import torch
import torch.nn as nn
from quantization.quantized_module import QuantizedModule
from quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase, AdaptiveGranularityQuantize, GroupLSQFakeQuantize, HybridQuantize


class StopForwardException(Exception):
    pass


class DataSaverHook:
    def __init__(
        self,
        store_input: bool = True,
        store_output: bool = True,
        stop_forward: bool = False,
    ):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, inputs, *hook_args):
        # ------ analyze hook_args，get kwargs & output ------
        if len(hook_args) == 1: # w/o kwargs
            kwargs = {}
            output = hook_args[0]
        elif len(hook_args) == 2:   # w/ with_kwargs=True
            kwargs, output = hook_args
            if kwargs is None:
                kwargs = {}
        else:
            kwargs = {}
            output = hook_args[-1] if len(hook_args) > 0 else None

        # ------ handling inputs ------
        if self.store_input:
            # only process positional args if w/o kwargs
            if not kwargs:
                if len(inputs) == 1:
                    self.input_store = inputs[0]
                else:
                    self.input_store = tuple(inputs)
            else:
                # try to merge args + kwargs as a single list during forward pass if kwargs exists
                try:
                    sig = inspect.signature(module.forward)
                    params = sig.parameters
                    full_args = []
                    pos_idx = 0

                    for name, p in params.items():
                        if name == "self":
                            continue

                        if name in kwargs:
                            # use arguments in kwargs
                            full_args.append(kwargs[name])
                        elif pos_idx < len(inputs):
                            # otherwise use positional args
                            full_args.append(inputs[pos_idx])
                            pos_idx += 1
                        elif p.default is not inspect._empty:
                            # use default argument (e.g.: num_k_exclude_rope=0)
                            full_args.append(p.default)
                        else:
                            raise RuntimeError(
                                f"[DataSaverHook] Cannot reconstruct argument '{name}' for module {type(module)}"
                            )

                    if len(full_args) == 1:
                        # single input: save as Tensor
                        self.input_store = full_args[0]
                    else:
                        # multiple inputs: save as tuple
                        self.input_store = tuple(full_args)

                except Exception as e:
                    print(f"[DataSaverHook][warn] failed to merge args/kwargs for {type(module)}: {e}")
                    if len(inputs) == 1:
                        self.input_store = inputs[0]
                    else:
                        self.input_store = tuple(inputs)

        # ------ process inputs ------
        if self.store_output:
            self.output_store = output

        # ------ optional: stop forward ------
        if self.stop_forward:
            raise StopForwardException

    def clear(self):
        self.input_store = None
        self.output_store = None


def _detach_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_detach_tensor(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _detach_tensor(v) for k, v in obj.items()}
    else:
        # other types: scalar, None, etc.
        return obj


def _to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def qdrop_mix(cur_quant_inp, cur_fp_inp, drop_prob: float):
    # single Tensor: element-wise QDrop
    if isinstance(cur_quant_inp, torch.Tensor):
        mask = (torch.rand_like(cur_quant_inp) < drop_prob)
        return torch.where(mask, cur_quant_inp, cur_fp_inp)

    # tuple or list: element-wise iteration
    if isinstance(cur_quant_inp, (tuple, list)):
        assert isinstance(cur_fp_inp, (tuple, list)) and len(cur_quant_inp) == len(cur_fp_inp), \
            "Structure mismatch between quant_inp and fp_inp (tuple/list)"
        return type(cur_quant_inp)(
            qdrop_mix(q, f, drop_prob) for q, f in zip(cur_quant_inp, cur_fp_inp)
        )

    # dict: interation by keys
    if isinstance(cur_quant_inp, dict):
        assert isinstance(cur_fp_inp, dict) and cur_quant_inp.keys() == cur_fp_inp.keys(), \
            "Structure mismatch between quant_inp and fp_inp (dict)"
        return {
            k: qdrop_mix(cur_quant_inp[k], cur_fp_inp[k], drop_prob)
            for k in cur_quant_inp.keys()
        }

    # others
    return cur_quant_inp


def inp_clone(cur_quant_inp):
    if isinstance(cur_quant_inp, torch.Tensor):
        return cur_quant_inp.clone()

    if isinstance(cur_quant_inp, (tuple, list)):
        return type(cur_quant_inp)(
            inp_clone(q) for q in cur_quant_inp
        )

    if isinstance(cur_quant_inp, dict):
        return {
            k: inp_clone(cur_quant_inp[k])
            for k in cur_quant_inp.keys()
        }

    return cur_quant_inp


def collect_module_io_sam2(
    predictor,
    module,
    cali_data,          # SAVVideoDataset instance
    num_sample: int,
    num_frame: int,
    store_inp: bool = True,
    store_oup: bool = True,
):

    data_saver = DataSaverHook(
        store_input=store_inp,
        store_output=store_oup,
        stop_forward=False,
    )
    handle = module.register_forward_hook(data_saver, with_kwargs=True)

    all_inputs = []
    all_outputs = []

    num_sample = min(num_sample, len(cali_data))
    indices = list(range(num_sample))  # do not shuffle samples to ensure identical FP & Quant data

    with torch.no_grad():
        for si, idx in enumerate(indices, 1):
            index = cali_data[idx]   # SAVJPEGIndex
            print(f"[Hook Calib {si}/{num_sample}] sample {index.sample}")

            # -------- prompt construction --------
            inst_ids = sorted(index.inst_to_frames.keys())
            inst_to_oid = {inst: (i + 1) for i, inst in enumerate(inst_ids)}

            # primary prompt
            t0, inst0, m0 = index.first_prompt()

            # extra prompts
            extra = []
            for inst in inst_ids:
                if inst == inst0:
                    continue
                frames = index.inst_to_frames[inst]
                if not frames:
                    continue
                t_k = frames[0]
                mp = index.ann_dir / inst / f"{t_k:05d}.png"
                m = np.array(Image.open(mp))
                if m.ndim == 3:
                    m = m[..., 0]
                m_bin = (m > 0).astype(np.uint8)
                extra.append((inst_to_oid[inst], t_k, m_bin))

            # -------- Initialize Video State --------
            state = predictor.init_state(str(index.img_dir))
            predictor.add_new_mask(
                state,
                frame_idx=t0,
                obj_id=1,
                mask=(m0 > 0),
            )

            for obj_id, t_idx, m in [(inst_to_oid[inst0], t0, m0)] + extra:
                try:
                    predictor.add_new_mask(
                        state,
                        frame_idx=t_idx,
                        obj_id=obj_id,
                        mask=(m > 0),
                    )
                except Exception:
                    ys, xs = np.where(m > 0)
                    if len(xs) == 0:
                        continue
                    box = np.array(
                        [[xs.min(), ys.min(), xs.max(), ys.max()]],
                        dtype=np.float32,
                    )
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=t_idx,
                        obj_id=obj_id,
                        box=box,
                    )

            # -------- propagate by frames, collect data from hooks --------
            frames_collected = 0
            for frame_i, (fidx, obj_ids, masks) in enumerate(
                predictor.propagate_in_video(state), start=0
            ):
                # frame_i == 0: do not save IO
                if frame_i == 0:
                    continue

                # collect IO from second frame
                # inputs: Tensor / tuple(Tensor,...) / list[Tensor] / ...
                if store_inp and getattr(data_saver, "input_store", None) is not None:
                    inp = data_saver.input_store
                    all_inputs.append(_detach_tensor(inp))

                # outputs: Tensor / (out, pos) / list[Tensor]/ ...
                if store_oup and getattr(data_saver, "output_store", None) is not None:
                    out = data_saver.output_store
                    all_outputs.append(_detach_tensor(out))

                frames_collected += 1
                if num_frame is not None and frames_collected >= num_frame:
                    break

    handle.remove()
    torch.cuda.empty_cache()

    return all_inputs, all_outputs


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 reg_weight=None,
                 reg_weight_lamb=0.1
                 ):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.reg_weight = reg_weight
        self.reg_weight_lamb = reg_weight_lamb

    def __call__(self, pred, tgt):
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
            w_reg_loss = 0
        else:
            round_loss = 0
            w_reg_loss = 0
            layer_len = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if self.reg_weight is None:
                        round_vals = layer.weight_fake_quant.rectified_sigmoid()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        # total_loss = w_reg_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.4f} (rec:{:.4f}, round:{:.4f}, rw:{:.4f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), float(w_reg_loss), b, self.count))
        return total_loss


def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    if isinstance(pred, torch.Tensor) and isinstance(tgt, torch.Tensor):
        return (pred - tgt).abs().pow(p).sum(1).mean()
    if isinstance(pred, (tuple, list)) and isinstance(tgt, (tuple, list)):
        return (pred[0][0] - tgt[0][0]).abs().pow(p).sum(1).mean()
    raise TypeError(f"Unsupported types in lp_loss: {type(pred)} vs {type(tgt)}")


def reconstruction(predictor, fp_predictor, module, fp_module, cali_data, q_config):

    device = next(module.parameters()).device
    quant_inp, _ = collect_module_io_sam2(predictor, module, cali_data, q_config.calibrate.sample, q_config.calibrate.frame, store_inp=True, store_oup=False)
    fp_inp, fp_oup = collect_module_io_sam2(fp_predictor, fp_module, cali_data, q_config.calibrate.sample, q_config.calibrate.frame, store_inp=True, store_oup=True)

    # Replace parameters with inference_mode() labels
    for sub_name, sub_module in module.named_modules():
        for pname, p in list(sub_module.named_parameters(recurse=False)):
            new_p = torch.nn.Parameter(p.detach().clone())
            setattr(sub_module, pname, new_p)

    w_para, a_para = [], []

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, q_config.recon.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
            layer.drop_prob = q_config.recon.drop_prob
            if isinstance(layer, LSQFakeQuantize):
                a_para += [layer.scale]
            if isinstance(layer, AdaptiveGranularityQuantize):
                a_para += [layer.scale]
            if isinstance(layer, GroupLSQFakeQuantize):
                a_para += [layer.grouped_scales]
            if isinstance(layer, HybridQuantize):
                a_para += [layer.scale_log]
                a_para += [layer.scale_uni]

    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=q_config.recon.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=q_config.recon.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    print(name)
    print(type(module))
    print(len(a_para))

    if len(a_para) == 0 and len(w_para) == 0:
        print('skip opt')
        del fp_inp, fp_oup, quant_inp
        torch.cuda.empty_cache()
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(module=module, weight=q_config.recon.weight, iters=q_config.recon.iters, b_range=q_config.recon.b_range, warm_up=q_config.recon.warm_up)

    # ==== move collected inputs / outputs to target device ====
    fp_oup = [_to_device(t, device) for t in fp_oup]
    fp_inp = [_to_device(t, device) for t in fp_inp]
    quant_inp = [_to_device(t, device) for t in quant_inp]

    # ====== QDrop Reconstruction ======
    sz = len(quant_inp)
    for it in range(q_config.recon.iters):
        idx = torch.randint(0, sz, ()).item()

        if q_config.recon.drop_prob < 1.0:
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]

            cur_inp = qdrop_mix(cur_quant_inp, cur_fp_inp, q_config.recon.drop_prob)
        else:
            cur_inp = inp_clone(quant_inp[idx])

        cur_fp_oup = fp_oup[idx]

        if a_opt:
            a_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()

        if cur_inp is None:
            continue

        if isinstance(cur_inp, list):
            if len(cur_inp) == 0:
                continue
            cur_quant_oup = module(cur_inp)

        elif isinstance(cur_inp, tuple):
            if len(cur_inp) == 0:
                continue
            elif len(cur_inp) == 1:
                cur_quant_oup = module(cur_inp[0])
            else:
                cur_quant_oup = module(*cur_inp)

        else:
            cur_quant_oup = module(cur_inp)
        # ===================================

        err = loss_func(cur_quant_oup, cur_fp_oup)
        cur_inp = None
        cur_quant_oup = None
        torch.cuda.empty_cache()
        err.backward()
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

        if q_config.ahcqsam.cag:
            contains_group_lsq = any(isinstance(sub_module, GroupLSQFakeQuantize) for sub_module in module.modules())
            if contains_group_lsq:
                if it in [int(q_config.recon.iters * 0.2), int(q_config.recon.iters * 0.4), int(q_config.recon.iters * 0.6), int(q_config.recon.iters * 0.8)]:
                    if it == int(q_config.recon.iters * 0.2):
                        try:
                            a_opt, a_scheduler = group_channel(module, a_para, q_config.recon, num_channel=q_config.ahcqsam.group * 8)
                        except Exception:
                            pass
                    if it == int(q_config.recon.iters * 0.4):
                        try:
                            a_opt, a_scheduler = group_channel(module, a_para, q_config.recon, num_channel=q_config.ahcqsam.group * 4)
                        except Exception:
                            pass
                    if it == int(q_config.recon.iters * 0.6):
                        try:
                            a_opt, a_scheduler = group_channel(module, a_para, q_config.recon, num_channel=q_config.ahcqsam.group * 2)
                        except Exception:
                            pass
                    if it == int(q_config.recon.iters * 0.8):
                        try:
                            a_opt, a_scheduler = group_channel(module, a_para, q_config.recon, num_channel=q_config.ahcqsam.group)
                        except Exception:
                            pass

    del fp_inp, fp_oup, quant_inp, cur_fp_oup
    torch.cuda.empty_cache()

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
            layer.drop_prob = 1.0


def group_channel(module, a_para, config, num_channel):
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, GroupLSQFakeQuantize):
            sub_module.group_channel(num_channel)
            print(f'Group number of activation channel into {num_channel}')
            a_para += [sub_module.grouped_scales]
            if len(a_para) != 0:
                a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
                a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
            else:
                a_opt, a_scheduler = None, None
    return a_opt, a_scheduler


def condiBasedAct_reconstruction(model, fp_model, module, fp_module, cali_data, config):
    lambda_ = 0.003
    beta = 0.001
    t_mode = 'mean'
    cond_num_before_optimization = cond_num_after_optimization = 0
    with torch.no_grad():
        try:
            U, S, Vh = torch.linalg.svd(module.module.weight.data.clone().T, full_matrices=False)
            s_max = torch.max(S)
            s_min = torch.min(S)
            cond_num_before_optimization = s_max / s_min
            print(f"Before Optimization \
                        Weight's Max: {s_max}, Min: {s_min}, Condi Number: {cond_num_before_optimization}")
        except torch.linalg.LinAlgError:
            print("SVD does not converge.")

    if cond_num_before_optimization < 100:
        # if True:
        print(f"Skip this layer due to the Condi Number {cond_num_before_optimization} < 100")
        return 0, 0

    if not module.layer_pre_act_fake_quantize.fake_quant_enabled or len(module.module.weight.shape) != 2:
        return 0, 0

    # return 0, 0
    device = next(module.parameters()).device
    # get data first
    quant_inp, _ = collect_module_io_sam2(model, module, cali_data, config.calibrate.sample,
                                          config.calibrate.frame, store_inp=True, store_oup=False)
    fp_inp, fp_oup = collect_module_io_sam2(fp_model, fp_module, cali_data, config.calibrate.sample,
                                            config.calibrate.frame, store_inp=True, store_oup=True)

    fp_oup = [_to_device(t, device) for t in fp_oup]
    fp_inp = [_to_device(t, device) for t in fp_inp]
    quant_inp = [_to_device(t, device) for t in quant_inp]

    def _proximal_step_inline_stable(W_in, C_i_stable):

        with torch.no_grad():
            try:
                U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
            except torch.linalg.LinAlgError:
                print("SVD does not converge, use prvious W_k.")
                return W_in

            if t_mode == 'mean':
                t = torch.mean(S)
            elif t_mode == 'median':
                t = torch.median(S)
            else:
                t = (torch.max(S) + torch.min(S)) / 2

            sigma_i = S
            numerator = sigma_i + 2 * lambda_ * t

            denominator = 1 + (2 * lambda_) + (2 * beta * C_i_stable)

            sigma_i_formula = numerator / denominator

            energies = S ** 2
            total_energy = torch.sum(energies)

            if total_energy == 0:
                return W_in

            c_energies = torch.cumsum(energies, dim=0)

            energy_threshold = 0.8 * total_energy

            indices_over_threshold = torch.where(c_energies > energy_threshold)[0]

            if len(indices_over_threshold) == 0:
                cutoff_index = len(S) - 1
            else:
                cutoff_index = indices_over_threshold[0]

            sigma_i_final_output = S.clone()

            minor_targets = torch.maximum(
                S[cutoff_index + 1:],
                sigma_i_formula[cutoff_index + 1:]
            )

            sigma_i_final_output[cutoff_index + 1:] = minor_targets

            W_out = U @ torch.diag(sigma_i_final_output) @ Vh
            return W_out

    accumulation_steps = 32
    K = config.ahcqsam.k
    print("Use Single Sample Per-Time")
    print(f"Use Gradient Accumulation: Batch Size = {accumulation_steps}.")

    for i in range(K):

        total_loss = 0.0

        with torch.no_grad():

            W_in = module.module.weight.data.clone().T
            try:
                U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
            except torch.linalg.LinAlgError:
                print("SVD does not converge, skip.")
                continue

            C_i_sum = 0.0
            total_samples_processed = 0
            for prox_step in range(accumulation_steps):
                idx = prox_step
                cur_inp_p = quant_inp[idx].to(device)
                cur_fp_inp_p = fp_inp[idx].to(device)

                delta_X_batch = cur_fp_inp_p - module.layer_pre_act_fake_quantize(cur_inp_p)

                assert delta_X_batch.shape[-1] == module.module.in_features
                D_in = module.module.in_features
                delta_X_r = delta_X_batch.reshape(-1, D_in).to(module.module.weight.device)

                delta_X_tilde = delta_X_r @ U
                C_i_current_sum = torch.sum(delta_X_tilde ** 2, dim=0)
                C_i_sum += C_i_current_sum
                num_samples_in_batch = delta_X_r.shape[0]
                total_samples_processed += num_samples_in_batch

                C_i_sum += C_i_current_sum

            C_i_stable = C_i_sum / total_samples_processed

            W_plus = _proximal_step_inline_stable(W_in, C_i_stable)

            module.module.weight.data = W_plus.T.data

        if (i + 1) % 10 == 0:
            avg_loss = total_loss / accumulation_steps
            print(f"PGD Iteration [{i + 1}/{K}] Avg Loss (Eff. BS={accumulation_steps}): {avg_loss:.4f}")

    with torch.no_grad():
        try:
            U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
            s_max = torch.max(S)
            s_min = torch.min(S)
            cond_num_after_optimization = s_max / s_min
            print(f"After Optimization \
                        Weight's Max: {s_max}, Min: {s_min}, Condi Number: {cond_num_after_optimization}.")
        except torch.linalg.LinAlgError:
            print("SVD does not converge.")
    if K > 0:
        del fp_inp, fp_oup, quant_inp
    torch.cuda.empty_cache()
    return cond_num_before_optimization, cond_num_after_optimization
