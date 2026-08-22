from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import trange

from ahcqsam.quantization.fake_quant import LSQFakeQuantize
from ahcqsam.quantization.state import enable_quantization
from pq_sam.quantizer import PQSAMGADTFakeQuantize


def _mse_loss(pred, target):
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        return F.mse_loss(pred, target)
    if isinstance(pred, (tuple, list)) and isinstance(target, (tuple, list)):
        losses = []
        for pred_item, target_item in zip(pred, target):
            loss = _mse_loss(pred_item, target_item)
            if loss is not None:
                losses.append(loss)
        if losses:
            return sum(losses)
    if isinstance(pred, dict) and isinstance(target, dict):
        losses = []
        for key, pred_item in pred.items():
            if key not in target:
                continue
            loss = _mse_loss(pred_item, target[key])
            if loss is not None:
                losses.append(loss)
        if losses:
            return sum(losses)
    return None


def _collect_trainable_quant_params(model, train_encoder=True):
    params = []
    seen = set()
    for name, module in model.named_modules():
        if not train_encoder and ".image_encoder." in name:
            continue
        if isinstance(module, PQSAMGADTFakeQuantize):
            module.drop_prob = 1.0
            for param in (module.scale, module.gadt_v, module.gadt_s):
                if id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        elif isinstance(module, LSQFakeQuantize) and "weight_fake_quant" in name:
            if id(module.scale) not in seen:
                params.append(module.scale)
                seen.add(id(module.scale))
    return params


def _freeze_except_quant_params(model, params):
    trainable_ids = {id(param) for param in params}
    for param in model.parameters():
        param.requires_grad = id(param) in trainable_ids


def _detach_output(output):
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _detach_output(value) for key, value in output.items()}
    return output


def _move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, tuple):
        return tuple(_move_to_device(item, device) for item in data)
    if isinstance(data, list):
        return [_move_to_device(item, device) for item in data]
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    return data


def optimize_quant_params_pq(model, fp_model, cali_data, pq_config):
    enable_quantization(model)
    fp_model.eval()
    model.eval()
    quant_device = next(model.parameters()).device
    teacher_device = next(fp_model.parameters()).device
    print(f"PQ-SAM optimization devices: quant={quant_device}, teacher={teacher_device}")

    train_encoder = bool(pq_config.get("train_encoder", False))
    params = _collect_trainable_quant_params(model, train_encoder=train_encoder)
    if not params:
        print("Skip PQ-SAM parameter optimization: no trainable quantization parameters.")
        return
    print(f"PQ-SAM train_encoder: {train_encoder}")
    _freeze_except_quant_params(model, params)

    opt = torch.optim.Adam(params, lr=pq_config.lr)
    epochs = int(pq_config.epochs)
    batch_size = int(pq_config.batch_size)
    max_calib_prompts = pq_config.get("max_calib_prompts", None)
    total_steps = max(epochs * ((len(cali_data) + batch_size - 1) // batch_size), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=0.0)
    step = 0

    progress = trange(total_steps, desc="PQ-SAM global optimization")
    for _ in range(epochs):
        indices = torch.randperm(len(cali_data))
        for start in range(0, len(cali_data), batch_size):
            batch_indices = indices[start:start + batch_size]
            opt.zero_grad()
            total_loss = 0.0
            valid_losses = 0
            for idx in batch_indices.tolist():
                data = cali_data[idx]
                with torch.no_grad():
                    teacher_data = _move_to_device(data, teacher_device)
                    target = _detach_output(
                        fp_model.extract_feat(teacher_data, max_calib_prompts=max_calib_prompts)
                    )
                    target = _move_to_device(target, quant_device)
                quant_data = _move_to_device(data, quant_device)
                pred = model.extract_feat(quant_data, max_calib_prompts=max_calib_prompts)
                loss = _mse_loss(pred, target)
                if loss is not None:
                    total_loss = total_loss + loss
                    valid_losses += 1
            if valid_losses == 0:
                progress.update(1)
                step += 1
                continue
            total_loss = total_loss / valid_losses
            total_loss.backward()
            opt.step()
            scheduler.step()

            progress.set_postfix(loss=f"{float(total_loss.detach()):.4e}")
            progress.update(1)
            step += 1
    progress.close()

    model.eval()
    for module in model.modules():
        if hasattr(module, "drop_prob"):
            module.drop_prob = 1.0
