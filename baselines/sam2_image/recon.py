from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

BASELINE_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = BASELINE_ROOT.parent
REPO_ROOT = BASELINES_ROOT.parent
ROOT = REPO_ROOT / "sam2"
SAM2_REPO = ROOT / "sam2"
AHCQSAM_ROOT = ROOT / "ahcqsam"
for path in (BASELINES_ROOT, ROOT, SAM2_REPO, AHCQSAM_ROOT, BASELINE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from quantization.fake_quant import (
    AdaptiveGranularityQuantize,
    GroupLSQFakeQuantize,
    HybridQuantize,
    LSQFakeQuantize,
    QuantizeBase,
)
from quantization.quantized_module import (
    PreQuantizedLayer,
    QuantizedBlock,
    QuantizedLayer,
    QuantizedMatMul,
)
from quantization.state import enable_quantization
from solver.recon import (
    DataSaverHook,
    LossFunction,
    _detach_tensor,
    _to_device,
    group_channel,
    inp_clone,
    qdrop_mix,
)

from coco_dataset import CocoBoxPromptDataset


RECON_MODULE_TYPES = (
    QuantizedBlock,
    QuantizedLayer,
    PreQuantizedLayer,
    QuantizedMatMul,
)
SKIP_IMAGE_RECON_ROOTS = {"memory_attention", "memory_encoder"}


def _call_module(module: nn.Module, cur_inp: Any) -> Any:
    if isinstance(cur_inp, list):
        if len(cur_inp) == 0:
            return None
        return module(cur_inp)
    if isinstance(cur_inp, tuple):
        if len(cur_inp) == 0:
            return None
        if len(cur_inp) == 1:
            return module(cur_inp[0])
        return module(*cur_inp)
    return module(cur_inp)


def _clone_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, list):
        return [_clone_tree(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tree(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone_tree(v) for k, v in obj.items()}
    return obj


def _append_hook_io(
    hook: DataSaverHook,
    all_inputs: list[Any],
    all_outputs: list[Any],
    max_samples: int,
    store_inp: bool,
    store_oup: bool,
    store_on_cpu: bool,
) -> bool:
    has_input = getattr(hook, "input_store", None) is not None
    has_output = getattr(hook, "output_store", None) is not None
    if store_inp and not has_input:
        return False
    if store_oup and not has_output:
        return False

    if store_inp:
        inp = _detach_tensor(hook.input_store)
        all_inputs.append(_to_device(inp, torch.device("cpu")) if store_on_cpu else inp)
    if store_oup:
        out = _detach_tensor(hook.output_store)
        all_outputs.append(_to_device(out, torch.device("cpu")) if store_on_cpu else out)
    hook.clear()
    return len(all_inputs) >= max_samples or len(all_outputs) >= max_samples


@torch.inference_mode()
def collect_module_io_sam2_image(
    predictor,
    module: nn.Module,
    dataset: CocoBoxPromptDataset,
    image_ids: list[int],
    max_samples: int,
    multimask_output: bool,
    store_inp: bool = True,
    store_oup: bool = True,
    store_on_cpu: bool = False,
) -> tuple[list[Any], list[Any]]:
    data_saver = DataSaverHook(
        store_input=store_inp,
        store_output=store_oup,
        stop_forward=False,
    )
    handle = module.register_forward_hook(data_saver, with_kwargs=True)

    all_inputs: list[Any] = []
    all_outputs: list[Any] = []

    try:
        progress = tqdm(
            dataset.iter_image_ids(image_ids),
            total=len(image_ids),
            desc="Collect recon IO",
            unit="image",
            leave=False,
        )
        for record in progress:
            if not record.image_path.exists():
                raise FileNotFoundError(f"Image file not found: {record.image_path}")

            image = np.array(Image.open(record.image_path).convert("RGB"))
            predictor.set_image(image)
            if _append_hook_io(
                data_saver,
                all_inputs,
                all_outputs,
                max_samples,
                store_inp,
                store_oup,
                store_on_cpu,
            ):
                break

            for instance in record.instances:
                predictor.predict(
                    box=instance.bbox_xyxy,
                    multimask_output=multimask_output,
                    return_logits=True,
                )
                if _append_hook_io(
                    data_saver,
                    all_inputs,
                    all_outputs,
                    max_samples,
                    store_inp,
                    store_oup,
                    store_on_cpu,
                ):
                    break

            progress.set_postfix(
                inputs=len(all_inputs),
                outputs=len(all_outputs),
            )
            if len(all_inputs) >= max_samples or len(all_outputs) >= max_samples:
                break
    finally:
        handle.remove()
        torch.cuda.empty_cache()

    return all_inputs[:max_samples], all_outputs[:max_samples]


def _init_recon_params(module: nn.Module, q_config) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    w_para: list[torch.nn.Parameter] = []
    a_para: list[torch.nn.Parameter] = []

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)) and hasattr(layer, "weight_fake_quant"):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, q_config.recon.round_mode)
            w_para.append(weight_quantizer.alpha)

        if isinstance(layer, QuantizeBase) and "act_fake_quantize" in name:
            layer.drop_prob = q_config.recon.drop_prob
            if isinstance(layer, LSQFakeQuantize):
                a_para.append(layer.scale)
            if isinstance(layer, AdaptiveGranularityQuantize):
                a_para.append(layer.scale)
            if isinstance(layer, GroupLSQFakeQuantize):
                a_para.append(layer.grouped_scales)
            if isinstance(layer, HybridQuantize):
                a_para.append(layer.scale_log)
                a_para.append(layer.scale_uni)

    return w_para, a_para


def _finalize_recon(module: nn.Module) -> None:
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)) and hasattr(layer, "weight_fake_quant"):
            weight_quantizer = layer.weight_fake_quant
            if getattr(weight_quantizer, "adaround", False):
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and "act_fake_quantize" in name:
            layer.drop_prob = 1.0


def _supports_acnr(module: nn.Module) -> bool:
    linear = getattr(module, "module", None)
    return (
        isinstance(module, PreQuantizedLayer)
        and isinstance(linear, nn.Linear)
        and getattr(module, "qinput", False)
        and hasattr(module, "layer_pre_act_fake_quantize")
    )


def reconstruction_image(
    predictor,
    fp_predictor,
    module: nn.Module,
    fp_module: nn.Module,
    dataset: CocoBoxPromptDataset,
    image_ids: list[int],
    q_config,
    multimask_output: bool,
) -> None:
    device = next(module.parameters()).device
    max_samples = int(q_config.recon.sample)
    store_on_cpu = not q_config.recon.keep_gpu

    quant_inp, _ = collect_module_io_sam2_image(
        predictor=predictor,
        module=module,
        dataset=dataset,
        image_ids=image_ids,
        max_samples=max_samples,
        multimask_output=multimask_output,
        store_inp=True,
        store_oup=False,
        store_on_cpu=store_on_cpu,
    )
    fp_inp, fp_oup = collect_module_io_sam2_image(
        predictor=fp_predictor,
        module=fp_module,
        dataset=dataset,
        image_ids=image_ids,
        max_samples=max_samples,
        multimask_output=multimask_output,
        store_inp=True,
        store_oup=True,
        store_on_cpu=store_on_cpu,
    )

    num_samples = min(len(quant_inp), len(fp_inp), len(fp_oup))
    if num_samples == 0:
        print("skip recon: no IO samples collected")
        return

    quant_inp = quant_inp[:num_samples]
    fp_inp = fp_inp[:num_samples]
    fp_oup = fp_oup[:num_samples]
    quant_inp = [_clone_tree(t) for t in quant_inp]
    fp_inp = [_clone_tree(t) for t in fp_inp]
    fp_oup = [_clone_tree(t) for t in fp_oup]

    for sub_module in module.modules():
        for pname, p in list(sub_module.named_parameters(recurse=False)):
            setattr(sub_module, pname, torch.nn.Parameter(p.detach().clone()))

    w_para, a_para = _init_recon_params(module, q_config)
    if len(a_para) == 0 and len(w_para) == 0:
        print("skip recon: no learnable quantization parameters")
        _finalize_recon(module)
        return

    if q_config.recon.keep_gpu:
        fp_oup = [_to_device(t, device) for t in fp_oup]
        fp_inp = [_to_device(t, device) for t in fp_inp]
        quant_inp = [_to_device(t, device) for t in quant_inp]

    a_opt = torch.optim.Adam(a_para, lr=q_config.recon.scale_lr) if a_para else None
    a_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            a_opt,
            T_max=q_config.recon.iters,
            eta_min=0.0,
        )
        if a_opt
        else None
    )
    w_opt = torch.optim.Adam(w_para) if w_para else None
    loss_func = LossFunction(
        module=module,
        weight=q_config.recon.weight,
        iters=q_config.recon.iters,
        b_range=q_config.recon.b_range,
        warm_up=q_config.recon.warm_up,
    )

    batch_size = max(1, int(q_config.recon.batch_size))
    print(
        f"Recon samples={num_samples}, batch_size={batch_size}, "
        f"act_params={len(a_para)}, weight_params={len(w_para)}"
    )

    for it in range(q_config.recon.iters):
        indices = torch.randint(0, num_samples, (min(batch_size, num_samples),)).tolist()

        if a_opt:
            a_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()

        total_err = None
        valid = 0
        for idx in indices:
            cur_quant_inp = _to_device(quant_inp[idx], device)
            cur_fp_inp = _to_device(fp_inp[idx], device)
            cur_fp_oup = _to_device(fp_oup[idx], device)

            if q_config.recon.drop_prob < 1.0:
                cur_inp = qdrop_mix(cur_quant_inp, cur_fp_inp, q_config.recon.drop_prob)
            else:
                cur_inp = inp_clone(cur_quant_inp)

            cur_quant_oup = _call_module(module, cur_inp)
            if cur_quant_oup is None:
                continue

            err = loss_func(cur_quant_oup, cur_fp_oup)
            total_err = err if total_err is None else total_err + err
            valid += 1

        if valid == 0 or total_err is None:
            continue

        total_err = total_err / valid
        total_err.backward()

        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

        if q_config.ahcqsam.cag:
            contains_group_lsq = any(
                isinstance(sub_module, GroupLSQFakeQuantize)
                for sub_module in module.modules()
            )
            if contains_group_lsq:
                milestones = [
                    int(q_config.recon.iters * 0.2),
                    int(q_config.recon.iters * 0.4),
                    int(q_config.recon.iters * 0.6),
                    int(q_config.recon.iters * 0.8),
                ]
                if it in milestones:
                    factors = {milestones[0]: 8, milestones[1]: 4, milestones[2]: 2, milestones[3]: 1}
                    try:
                        a_opt, a_scheduler = group_channel(
                            module,
                            a_para,
                            q_config.recon,
                            num_channel=q_config.ahcqsam.group * factors[it],
                        )
                    except Exception:
                        pass

    _finalize_recon(module)
    torch.cuda.empty_cache()


def _proximal_step_stable(W_in: torch.Tensor, C_i_stable: torch.Tensor) -> torch.Tensor:
    lambda_ = 0.003
    beta = 0.001
    with torch.no_grad():
        try:
            U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
        except torch.linalg.LinAlgError:
            print("SVD does not converge, keep previous weight.")
            return W_in

        t = torch.mean(S)
        sigma_i = S
        numerator = sigma_i + 2 * lambda_ * t
        denominator = 1 + (2 * lambda_) + (2 * beta * C_i_stable)
        sigma_i_formula = numerator / denominator

        energies = S**2
        total_energy = torch.sum(energies)
        if total_energy == 0:
            return W_in

        c_energies = torch.cumsum(energies, dim=0)
        indices_over_threshold = torch.where(c_energies > 0.8 * total_energy)[0]
        cutoff_index = (
            len(S) - 1
            if len(indices_over_threshold) == 0
            else indices_over_threshold[0]
        )

        sigma_i_final = S.clone()
        if cutoff_index + 1 < len(S):
            sigma_i_final[cutoff_index + 1:] = torch.maximum(
                S[cutoff_index + 1:],
                sigma_i_formula[cutoff_index + 1:],
            )

        return U @ torch.diag(sigma_i_final) @ Vh


def condi_based_act_reconstruction_image(
    predictor,
    fp_predictor,
    module: PreQuantizedLayer,
    fp_module: nn.Module,
    dataset: CocoBoxPromptDataset,
    image_ids: list[int],
    q_config,
    multimask_output: bool,
) -> tuple[float, float]:
    if not _supports_acnr(module):
        return 0.0, 0.0

    device = next(module.parameters()).device
    weight = module.module.weight.data
    if weight.ndim != 2:
        return 0.0, 0.0

    with torch.no_grad():
        try:
            _, S_before, _ = torch.linalg.svd(weight.clone().T, full_matrices=False)
            cond_before = float(torch.max(S_before) / torch.min(S_before))
        except torch.linalg.LinAlgError:
            print("SVD does not converge, skip ACNR.")
            return 0.0, 0.0

    if cond_before < 100:
        print(f"Skip ACNR due to condition number {cond_before:.4f} < 100")
        return 0.0, 0.0

    max_samples = int(q_config.recon.sample)
    store_on_cpu = not q_config.recon.keep_gpu
    quant_inp, _ = collect_module_io_sam2_image(
        predictor=predictor,
        module=module,
        dataset=dataset,
        image_ids=image_ids,
        max_samples=max_samples,
        multimask_output=multimask_output,
        store_inp=True,
        store_oup=False,
        store_on_cpu=store_on_cpu,
    )
    fp_inp, _ = collect_module_io_sam2_image(
        predictor=fp_predictor,
        module=fp_module,
        dataset=dataset,
        image_ids=image_ids,
        max_samples=max_samples,
        multimask_output=multimask_output,
        store_inp=True,
        store_oup=False,
        store_on_cpu=store_on_cpu,
    )

    num_samples = min(len(quant_inp), len(fp_inp))
    if num_samples == 0:
        print("skip ACNR: no IO samples collected")
        return cond_before, cond_before

    quant_inp = [_clone_tree(t) for t in quant_inp[:num_samples]]
    fp_inp = [_clone_tree(t) for t in fp_inp[:num_samples]]
    batch_size = max(1, int(q_config.recon.batch_size))
    K = int(q_config.ahcqsam.k)
    print(
        f"ACNR samples={num_samples}, batch_size={batch_size}, "
        f"iterations={K}, cond_before={cond_before:.4f}"
    )

    for i in range(K):
        with torch.no_grad():
            W_in = module.module.weight.data.clone().T
            try:
                U, _, _ = torch.linalg.svd(W_in, full_matrices=False)
            except torch.linalg.LinAlgError:
                print("SVD does not converge, skip this ACNR step.")
                continue

            C_i_sum = None
            total_rows = 0
            indices = torch.randint(0, num_samples, (min(batch_size, num_samples),)).tolist()
            for idx in indices:
                cur_quant_inp = _to_device(quant_inp[idx], device)
                cur_fp_inp = _to_device(fp_inp[idx], device)
                if not isinstance(cur_quant_inp, torch.Tensor) or not isinstance(cur_fp_inp, torch.Tensor):
                    continue

                quantized_inp = module.layer_pre_act_fake_quantize(cur_quant_inp)
                delta_x = cur_fp_inp - quantized_inp
                d_in = module.module.in_features
                if delta_x.shape[-1] != d_in:
                    continue

                delta_x = delta_x.reshape(-1, d_in).to(module.module.weight.device)
                delta_x_tilde = delta_x @ U
                cur_sum = torch.sum(delta_x_tilde**2, dim=0)
                C_i_sum = cur_sum if C_i_sum is None else C_i_sum + cur_sum
                total_rows += delta_x.shape[0]

            if C_i_sum is None or total_rows == 0:
                continue

            C_i_stable = C_i_sum / total_rows
            W_plus = _proximal_step_stable(W_in, C_i_stable)
            module.module.weight.data = W_plus.T.data

        if (i + 1) % 10 == 0:
            print(f"ACNR iteration [{i + 1}/{K}]")

    with torch.no_grad():
        try:
            _, S_after, _ = torch.linalg.svd(module.module.weight.data.clone().T, full_matrices=False)
            cond_after = float(torch.max(S_after) / torch.min(S_after))
        except torch.linalg.LinAlgError:
            print("SVD does not converge after ACNR.")
            cond_after = cond_before

    torch.cuda.empty_cache()
    return cond_before, cond_after


def select_recon_image_ids(
    dataset: CocoBoxPromptDataset,
    seed: int,
) -> list[int]:
    image_ids = list(dataset.image_ids)
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return image_ids


def condi_based_act_recon_model_image(
    predictor,
    fp_predictor,
    dataset: CocoBoxPromptDataset,
    q_config,
    multimask_output: bool,
    seed: int,
) -> None:
    if len(dataset) == 0:
        raise RuntimeError("No images are available for ACNR reconstruction.")

    image_ids = select_recon_image_ids(dataset, seed)
    enable_quantization(predictor.model, "act_fake_quantize")

    def _condi_recon_model(module: nn.Module, fp_module: nn.Module, prefix: str = "") -> None:
        for name, child_module in module.named_children():
            global_name = f"{prefix}.{name}" if prefix else name
            if prefix == "" and name in SKIP_IMAGE_RECON_ROOTS:
                continue
            child_fp_module = getattr(fp_module, name)
            if isinstance(child_module, PreQuantizedLayer):
                print(f"Begin ACNR reconstruction for module [{global_name}]")
                cond_before, cond_after = condi_based_act_reconstruction_image(
                    predictor=predictor,
                    fp_predictor=fp_predictor,
                    module=child_module,
                    fp_module=child_fp_module,
                    dataset=dataset,
                    image_ids=image_ids,
                    q_config=q_config,
                    multimask_output=multimask_output,
                )
                if cond_before and cond_before < cond_after:
                    print(
                        f"Module [{global_name}], "
                        f"Cond_Num_Before_Optimization: {cond_before}, "
                        f"Cond_Num_After_Optimization: {cond_after}"
                    )
            else:
                _condi_recon_model(child_module, child_fp_module, global_name)

    _condi_recon_model(predictor.model, fp_predictor.model)


def recon_model_image(
    predictor,
    fp_predictor,
    dataset: CocoBoxPromptDataset,
    q_config,
    multimask_output: bool,
    seed: int,
) -> None:
    if len(dataset) == 0:
        raise RuntimeError("No images are available for reconstruction.")

    image_ids = select_recon_image_ids(dataset, seed)
    enable_quantization(predictor.model)

    def _recon_model(module: nn.Module, fp_module: nn.Module, prefix: str = "") -> None:
        for name, child_module in module.named_children():
            global_name = f"{prefix}.{name}" if prefix else name
            if prefix == "" and name in SKIP_IMAGE_RECON_ROOTS:
                continue
            child_fp_module = getattr(fp_module, name)
            if isinstance(child_module, RECON_MODULE_TYPES):
                print(f"Begin reconstruction for module [{global_name}]: {type(child_module)}")
                reconstruction_image(
                    predictor=predictor,
                    fp_predictor=fp_predictor,
                    module=child_module,
                    fp_module=child_fp_module,
                    dataset=dataset,
                    image_ids=image_ids,
                    q_config=q_config,
                    multimask_output=multimask_output,
                )
            else:
                _recon_model(child_module, child_fp_module, global_name)

    _recon_model(predictor.model, fp_predictor.model)
