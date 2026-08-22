from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from quantization.fake_quant import QuantizeBase
from quantization.quantized_module import PreQuantizedLayer, QuantizedLayer, QuantizedMatMul


@dataclass
class LayerUnit:
    name: str
    module: nn.Module
    module_type: str
    num_params: int
    quantizer_targets: list[str]
    macs: float = 0.0


ForwardDistributionFn = Callable[[nn.Module, Any, float, int | None, Any], torch.Tensor | None]


def discover_layer_units(model: nn.Module, max_units: int | None = None) -> list[LayerUnit]:
    units: list[LayerUnit] = []
    for name, module in model.named_modules():
        if isinstance(module, PreQuantizedLayer):
            targets = [f"{name}.module.weight_fake_quant"]
            if getattr(module, "qinput", False):
                targets.append(f"{name}.layer_pre_act_fake_quantize")
            units.append(make_unit(name, module, targets))
        elif isinstance(module, QuantizedLayer):
            targets = [f"{name}.module.weight_fake_quant"]
            if getattr(module, "qoutput", False):
                targets.append(f"{name}.layer_post_act_fake_quantize")
            units.append(make_unit(name, module, targets))
        elif isinstance(module, QuantizedMatMul):
            targets = []
            if getattr(module, "qinput", False):
                targets.extend(
                    [
                        f"{name}.a_layer_pre_act_fake_quantize",
                        f"{name}.b_layer_pre_act_fake_quantize",
                    ]
                )
            units.append(make_unit(name, module, targets))

    if max_units is not None:
        units = units[:max_units]
    return units


def exclude_first_last_units(units: list[LayerUnit]) -> list[LayerUnit]:
    if len(units) <= 2:
        return units
    return units[1:-1]


def make_unit(name: str, module: nn.Module, targets: list[str]) -> LayerUnit:
    return LayerUnit(
        name=name,
        module=module,
        module_type=type(module).__name__,
        num_params=count_compute_params(module),
        quantizer_targets=targets,
    )


def count_compute_params(module: nn.Module) -> int:
    if isinstance(module, (PreQuantizedLayer, QuantizedLayer)):
        wrapped = getattr(module, "module", None)
        if wrapped is not None:
            return sum(param.numel() for param in wrapped.parameters(recurse=False))
    return sum(param.numel() for param in module.parameters(recurse=False))


def unit_weight_matrix(unit: LayerUnit) -> torch.Tensor | None:
    if not isinstance(unit.module, (PreQuantizedLayer, QuantizedLayer)):
        return None
    wrapped = getattr(unit.module, "module", None)
    weight = getattr(wrapped, "weight", None)
    if not torch.is_tensor(weight) or weight.numel() == 0:
        return None
    weight = weight.detach().float().cpu()
    if weight.ndim == 2:
        return weight.abs()
    if weight.ndim >= 3:
        return weight.abs().flatten(start_dim=2).mean(dim=2)
    return None


def get_nested_module(root: nn.Module, path: str) -> nn.Module | None:
    module: Any = root
    for part in path.split("."):
        if part.isdigit():
            try:
                module = module[int(part)]
            except (TypeError, IndexError):
                return None
        elif hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module if isinstance(module, nn.Module) else None


def register_mac_hooks(units: list[LayerUnit], mac_sums: dict[str, float]) -> list[Any]:
    handles = []
    for unit in units:
        handles.append(unit.module.register_forward_hook(make_mac_hook(unit, mac_sums)))
    return handles


def make_mac_hook(unit: LayerUnit, mac_sums: dict[str, float]):
    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        mac_sums[unit.name] = mac_sums.get(unit.name, 0.0) + estimate_macs(module, inputs, output)

    return hook


def estimate_macs(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> float:
    if isinstance(module, (PreQuantizedLayer, QuantizedLayer)):
        wrapped = getattr(module, "module", None)
        if isinstance(wrapped, nn.Linear):
            return float(tensor_numel(output) * wrapped.in_features)
        if isinstance(wrapped, nn.Conv2d):
            kernel_ops = (
                wrapped.kernel_size[0]
                * wrapped.kernel_size[1]
                * wrapped.in_channels
                / wrapped.groups
            )
            return float(tensor_numel(output) * kernel_ops)
        if isinstance(wrapped, nn.Embedding):
            return float(tensor_numel(output))
    if isinstance(module, QuantizedMatMul) and inputs:
        matmul_inputs = inputs[0]
        if isinstance(matmul_inputs, (tuple, list)) and len(matmul_inputs) == 2:
            lhs, rhs = matmul_inputs
            if torch.is_tensor(lhs) and torch.is_tensor(rhs) and lhs.ndim >= 2 and rhs.ndim >= 2:
                return float(lhs.numel() * rhs.shape[-1])
    return float(max(tensor_numel(output), 0))


def tensor_numel(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel()
    if isinstance(value, (tuple, list)):
        return sum(tensor_numel(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_numel(item) for item in value.values())
    return 0


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


class ActivationStats:
    def __init__(self) -> None:
        self.sum_sq: torch.Tensor | None = None
        self.count = 0

    def update(self, value: Any) -> None:
        tensor = pick_first_tensor(value)
        if tensor is None or tensor.numel() == 0:
            return
        tensor = tensor.detach().float().cpu()
        channel_values = channel_sum_sq_and_count(tensor)
        if channel_values is None:
            return
        sum_sq, count = channel_values
        if self.sum_sq is None:
            self.sum_sq = sum_sq
        elif self.sum_sq.numel() == sum_sq.numel():
            self.sum_sq += sum_sq
        else:
            size = min(self.sum_sq.numel(), sum_sq.numel())
            self.sum_sq[:size] += sum_sq[:size]
        self.count += count

    def rms(self) -> torch.Tensor | None:
        if self.sum_sq is None or self.count <= 0:
            return None
        return torch.sqrt(self.sum_sq / max(self.count, 1))


def pick_first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = pick_first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = pick_first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def channel_sum_sq_and_count(tensor: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    if tensor.ndim == 0:
        return None
    if tensor.ndim == 1:
        return tensor.pow(2), 1
    if tensor.ndim == 2:
        return tensor.pow(2).sum(dim=0), tensor.shape[0]
    if tensor.ndim == 3:
        return tensor.pow(2).sum(dim=(0, 1)), tensor.shape[0] * tensor.shape[1]
    if tensor.ndim == 4:
        return tensor.pow(2).sum(dim=(0, 2, 3)), tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
    return tensor.flatten(0, -2).pow(2).sum(dim=0), math.prod(tensor.shape[:-1])


@torch.no_grad()
def collect_scores(
    model: nn.Module,
    samples: list[Any],
    units: list[LayerUnit],
    forward_distribution: ForwardDistributionFn,
    eps: float,
    score_frames: int | None,
    sis_config=None,
) -> dict[str, Any]:
    skipped_samples = 0
    mac_sums = {unit.name: 0.0 for unit in units}
    activation_stats = {unit.name: ActivationStats() for unit in units}

    for sample in tqdm(samples, desc="Collect Mix-QSAM2 WAIS scores", unit="video"):
        mac_handles = register_mac_hooks(units, mac_sums)
        activation_handles = register_activation_hooks(units, activation_stats)
        try:
            reference = forward_distribution(model, sample, eps, score_frames, sis_config)
        finally:
            remove_hooks(mac_handles)
            remove_hooks(activation_handles)
        if reference is None:
            skipped_samples += 1
            continue

        del reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    valid_samples = len(samples) - skipped_samples
    if valid_samples <= 0:
        raise RuntimeError("No valid video sample produced output distributions for MPQ scoring.")

    raw_importance = [
        layer_wais(unit, activation_stats[unit.name].rms())
        for unit in units
    ]
    score_sum = sum(raw_importance)
    importance = [score / score_sum for score in raw_importance] if score_sum > eps else raw_importance
    macs = [mac_sums[unit.name] / valid_samples for unit in units]
    for unit, mac in zip(units, macs):
        unit.macs = mac

    return {
        "unit_names": [unit.name for unit in units],
        "num_samples": valid_samples,
        "skipped_samples": skipped_samples,
        "raw_importance": raw_importance,
        "importance": importance,
        "macs": macs,
    }


def layer_wais(unit: LayerUnit, activation_rms: torch.Tensor | None) -> float:
    weight = unit_weight_matrix(unit)
    if weight is None or activation_rms is None:
        return 0.0
    channels = min(weight.shape[1], activation_rms.numel())
    if channels <= 0:
        return 0.0
    score = weight[:, :channels] * activation_rms[:channels].reshape(1, -1)
    return float(score.mean().cpu())


def register_activation_hooks(
    units: list[LayerUnit],
    activation_stats: dict[str, ActivationStats],
) -> list[Any]:
    handles = []
    for unit in units:
        handles.append(unit.module.register_forward_hook(make_activation_hook(unit, activation_stats)))
    return handles


def make_activation_hook(
    unit: LayerUnit,
    activation_stats: dict[str, ActivationStats],
):
    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        activation_stats[unit.name].update(inputs[0] if inputs else None)

    return hook


def load_or_collect_scores(
    model: nn.Module,
    samples: list[Any],
    units: list[LayerUnit],
    forward_distribution: ForwardDistributionFn,
    eps: float,
    score_frames: int | None,
    score_cache: str | None,
    sis_config=None,
) -> dict[str, Any]:
    if score_cache and os.path.exists(score_cache):
        with open(score_cache) as f:
            scores = json.load(f)
        if (
            scores.get("unit_names") == [unit.name for unit in units]
            and scores.get("macs") is not None
            and scores.get("raw_importance") is not None
        ):
            for unit, mac in zip(units, scores["macs"]):
                unit.macs = mac
            return scores
        print("Score cache does not match current units; recollecting scores.")

    scores = collect_scores(
        model,
        samples,
        units,
        forward_distribution,
        eps,
        score_frames,
        sis_config=sis_config,
    )
    if score_cache:
        Path(score_cache).parent.mkdir(parents=True, exist_ok=True)
        with open(score_cache, "w") as f:
            json.dump(scores, f, indent=2)
    return scores


def solve_bit_allocation(
    units: list[LayerUnit],
    importance: list[float],
    candidate_bits: list[int],
    target_bits: int,
) -> tuple[list[int], dict[str, Any]]:
    candidate_bits = sorted(set(int(bit) for bit in candidate_bits))
    min_bit = candidate_bits[0]
    max_bit = candidate_bits[-1]
    total_params = sum(unit.num_params for unit in units)
    total_macs = sum(unit.macs for unit in units)
    budget = total_params * target_bits
    if target_bits <= min_bit:
        bits = [min_bit for _ in units]
    elif target_bits >= max_bit:
        bits = [max_bit for _ in units]
    else:
        bits = initialize_bits_by_importance(units, importance, candidate_bits, target_bits)
        bits = adjust_bits_to_budget(units, importance, bits, candidate_bits, budget)

    used_budget = sum(unit.num_params * bit for unit, bit in zip(units, bits))
    return bits, {
        "solver": "mix_qsam2_importance_heuristic",
        "target_budget_param_bits": budget,
        "used_budget_param_bits": used_budget,
        "weighted_average_bit": used_budget / max(total_params, 1),
        "total_params": total_params,
        "total_macs": total_macs,
        "objective": allocation_objective(bits, importance),
        "warning": None,
    }


def initialize_bits_by_importance(
    units: list[LayerUnit],
    importance: list[float],
    candidate_bits: list[int],
    target_bits: int,
) -> list[int]:
    bits = [target_bits for _ in units]
    ranked = sorted(range(len(units)), key=lambda i: importance[i], reverse=True)
    bit_count = len(candidate_bits)
    for rank, idx in enumerate(ranked):
        bucket = min(rank * bit_count // max(len(ranked), 1), bit_count - 1)
        bits[idx] = candidate_bits[-(bucket + 1)]
    return bits


def adjust_bits_to_budget(
    units: list[LayerUnit],
    importance: list[float],
    bits: list[int],
    candidate_bits: list[int],
    budget: float,
) -> list[int]:
    bits = bits.copy()
    while weighted_bit_budget(units, bits) > budget:
        candidates = [
            idx for idx, bit in enumerate(bits)
            if previous_bit(bit, candidate_bits) is not None
        ]
        if not candidates:
            break
        idx = min(candidates, key=lambda i: importance[i])
        bits[idx] = previous_bit(bits[idx], candidate_bits)

    while True:
        current_budget = weighted_bit_budget(units, bits)
        candidates = [
            idx for idx, bit in enumerate(bits)
            if next_bit(bit, candidate_bits) is not None
            and current_budget + units[idx].num_params * (next_bit(bit, candidate_bits) - bit) <= budget
        ]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: importance[i] * units[i].num_params)
        bits[idx] = next_bit(bits[idx], candidate_bits)
    return bits


def weighted_bit_budget(units: list[LayerUnit], bits: list[int]) -> float:
    return sum(unit.num_params * bit for unit, bit in zip(units, bits))


def previous_bit(bit: int, candidate_bits: list[int]) -> int | None:
    lower = [candidate for candidate in candidate_bits if candidate < bit]
    return lower[-1] if lower else None


def next_bit(bit: int, candidate_bits: list[int]) -> int | None:
    higher = [candidate for candidate in candidate_bits if candidate > bit]
    return higher[0] if higher else None


def allocation_objective(bits: list[int], importance: list[float]) -> float:
    return sum(score * (2.0 ** (-bit)) for score, bit in zip(importance, bits))


def select_score_samples(dataset, num_samples: int, seed: int) -> list[Any]:
    indices = list(range(len(dataset)))
    rng = __import__("random").Random(seed)
    rng.shuffle(indices)
    return [dataset[idx] for idx in indices[: min(num_samples, len(indices))]]


@torch.no_grad()
def search_mixed_precision_bits(
    model: nn.Module,
    train_dataset,
    q_config,
    forward_distribution: ForwardDistributionFn,
    seed: int,
    sis_config=None,
) -> dict[str, Any] | None:
    mix_cfg = getattr(q_config, "mix_qsam2", None)
    if mix_cfg is None or not mix_cfg.enabled:
        return None

    score_samples = int(mix_cfg.score_samples)
    score_frames = int(mix_cfg.score_frames) if mix_cfg.score_frames else None
    eps = float(mix_cfg.eps)
    target_bits = int(mix_cfg.target_bits)
    units = discover_layer_units(model, getattr(mix_cfg, "max_units", None))
    if getattr(mix_cfg, "exclude_first_last", True):
        units = exclude_first_last_units(units)
    if not units:
        raise RuntimeError("No quantized units were discovered for Mix-QSAM2 MPQ.")

    print("Search Mix-QSAM2 bit allocation")
    print(f"  units: {len(units)}")
    print(f"  target_bits: {target_bits}")
    print(f"  candidate_bits: {list(mix_cfg.candidate_bits)}")
    print(f"  score_samples: {score_samples}")
    print(f"  score_frames: {score_frames}")
    print(f"  exclude_first_last: {getattr(mix_cfg, 'exclude_first_last', True)}")
    print("  solver: mix_qsam2_importance_heuristic")

    samples = select_score_samples(train_dataset, score_samples, seed)
    scores = load_or_collect_scores(
        model=model,
        samples=samples,
        units=units,
        forward_distribution=forward_distribution,
        eps=eps,
        score_frames=score_frames,
        score_cache=mix_cfg.score_cache,
        sis_config=sis_config,
    )
    bits, metadata = solve_bit_allocation(
        units=units,
        importance=scores["importance"],
        candidate_bits=list(mix_cfg.candidate_bits),
        target_bits=target_bits,
    )
    return {
        "config": {
            "score_samples": score_samples,
            "score_frames": score_frames,
            "target_bits": target_bits,
            "candidate_bits": list(mix_cfg.candidate_bits),
            "solver": "mix_qsam2_importance_heuristic",
            "exclude_first_last": bool(getattr(mix_cfg, "exclude_first_last", True)),
            "eps": eps,
        },
        "score_summary": {
            "num_samples": scores["num_samples"],
            "skipped_samples": scores["skipped_samples"],
        },
        "allocation": metadata,
        "units": [
            {
                "name": unit.name,
                "type": unit.module_type,
                "num_params": unit.num_params,
                "macs": unit.macs,
                "raw_importance": scores["raw_importance"][idx],
                "importance": scores["importance"][idx],
                "bit": bits[idx],
                "quantizer_targets": unit.quantizer_targets,
            }
            for idx, unit in enumerate(units)
        ],
    }


def apply_mixed_precision_bits(model: nn.Module, allocation: dict[str, Any] | None) -> None:
    if allocation is None:
        return
    applied = 0
    skipped: list[tuple[str, str]] = []
    for unit in allocation.get("units", []):
        bit = int(unit["bit"])
        for target in unit.get("quantizer_targets", []):
            quantizer = get_nested_module(model, target)
            if not isinstance(quantizer, QuantizeBase):
                skipped.append((unit["name"], target))
                continue
            quantizer.set_bit(bit)
            applied += 1

    print("Applied Mix-QSAM2 bits")
    print(f"  quantizers updated: {applied}")
    print(f"  quantizer targets skipped: {len(skipped)}")
    if skipped:
        print("  skipped examples:")
        for unit_name, target in skipped[:20]:
            print(f"    {unit_name} -> {target}")
