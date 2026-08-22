from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from pq_sam.quantizer import PQSAMGADTFakeQuantize


def _flatten_channels(x: torch.Tensor, ch_axis: int) -> torch.Tensor:
    if ch_axis < 0:
        ch_axis += x.ndim
    return x.detach().movedim(ch_axis, -1).reshape(-1, x.shape[ch_axis]).float().cpu()


class OHCActivationCollector:
    def __init__(self, max_tokens_per_layer: int, max_elements_per_layer: int):
        self.max_tokens_per_layer = max_tokens_per_layer
        self.max_elements_per_layer = max_elements_per_layer
        self.samples = {}
        self.counts = {}
        self.element_counts = {}
        self.handles = []

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, PQSAMGADTFakeQuantize):
                self.samples[name] = []
                self.counts[name] = 0
                self.element_counts[name] = 0
                self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, name: str, module: PQSAMGADTFakeQuantize):
        def hook(_, inputs, __):
            if self.counts[name] >= self.max_tokens_per_layer:
                return

            x = inputs[0]
            ch_axis = module._resolve_channel_axis(x)
            flat = _flatten_channels(x, ch_axis)
            channel_count = flat.shape[1]
            remaining_tokens = self.max_tokens_per_layer - self.counts[name]
            remaining_elements = self.max_elements_per_layer - self.element_counts[name]
            remaining = min(remaining_tokens, max(remaining_elements // channel_count, 0))
            if remaining <= 0:
                return
            if flat.shape[0] > remaining:
                index = torch.randperm(flat.shape[0])[:remaining]
                flat = flat[index]
            self.samples[name].append(flat)
            self.counts[name] += flat.shape[0]
            self.element_counts[name] += flat.numel()

        return hook

    def tensors(self):
        output = {}
        for name, chunks in self.samples.items():
            if chunks:
                output[name] = torch.cat(chunks, dim=0)
        return output


def _sample_flattened(x: torch.Tensor, max_values: int) -> torch.Tensor:
    x = x.reshape(-1)
    if x.numel() <= max_values:
        return x
    index = torch.randperm(x.numel())[:max_values]
    return x[index]


def _quantile(x: torch.Tensor, q, max_values: int):
    sample = _sample_flattened(x, max_values)
    q = torch.as_tensor(q, dtype=sample.dtype, device=sample.device)
    return torch.quantile(sample, q)


def _sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    if x.shape[0] <= max_rows:
        return x
    index = torch.randperm(x.shape[0])[:max_rows]
    return x[index]


def _tensorwise_truncate(x: torch.Tensor, outlier_lambda: float, max_quantile_values: int) -> torch.Tensor:
    q1, q3 = _quantile(x, [0.25, 0.75], max_quantile_values)
    iqr = q3 - q1
    lower = q1 - outlier_lambda * iqr
    upper = q3 + outlier_lambda * iqr
    return x.clamp(lower, upper)


def _central_group_mask(
    x: torch.Tensor,
    remaining: torch.Tensor,
    alpha: float,
    max_quantile_values: int,
    max_channel_quantile_tokens: int,
) -> torch.Tensor:
    remaining_x = x[:, remaining]
    lower_q = min(alpha, 1.0 - alpha)
    upper_q = max(alpha, 1.0 - alpha)
    lower, upper = _quantile(remaining_x, [lower_q, upper_q], max_quantile_values)
    channel_sample = _sample_rows(remaining_x, max_channel_quantile_tokens)
    channel_lower = torch.quantile(channel_sample, lower_q, dim=0)
    channel_upper = torch.quantile(channel_sample, upper_q, dim=0)
    mask = (channel_lower >= lower) & (channel_upper <= upper)
    if mask.any():
        return mask

    center = (lower + upper) * 0.5
    channel_center = remaining_x.median(dim=0).values
    closest = torch.argmin((channel_center - center).abs())
    mask = torch.zeros_like(channel_center, dtype=torch.bool)
    mask[closest] = True
    return mask


def _build_ohc_groups(
    x: torch.Tensor,
    outlier_lambda: float,
    alpha: float,
    max_quantile_values: int,
    max_channel_quantile_tokens: int,
):
    x = _tensorwise_truncate(x, outlier_lambda, max_quantile_values)
    num_channels = x.shape[1]
    labels = torch.full((num_channels,), -1, dtype=torch.long)
    init_v = []
    init_s = []
    remaining = torch.arange(num_channels)
    group_id = 0

    while remaining.numel() > 0:
        mask = _central_group_mask(
            x,
            remaining,
            alpha,
            max_quantile_values,
            max_channel_quantile_tokens,
        )
        group_channels = remaining[mask]
        group_x = x[:, group_channels]
        init_v.append(group_x.mean())
        init_s.append(group_x.std(unbiased=False).clamp(min=torch.finfo(group_x.dtype).eps))
        labels[group_channels] = group_id
        remaining = remaining[~mask]
        group_id += 1

    return labels, torch.stack(init_v), torch.stack(init_s)


@torch.no_grad()
def initialize_ohc(model: nn.Module, cali_data, pq_config) -> None:
    collector = OHCActivationCollector(
        max_tokens_per_layer=pq_config.max_tokens_per_layer,
        max_elements_per_layer=pq_config.max_elements_per_layer,
    )
    collector.register(model)
    try:
        for data in tqdm(cali_data, desc="Collect OHC activations", unit="image"):
            model.extract_feat(data)
    finally:
        collector.remove()

    activations = collector.tensors()
    for name, module in model.named_modules():
        if not isinstance(module, PQSAMGADTFakeQuantize):
            continue
        if name not in activations:
            continue
        labels, init_v, init_s = _build_ohc_groups(
            activations[name],
            outlier_lambda=pq_config.outlier_lambda,
            alpha=pq_config.grouping_alpha,
            max_quantile_values=pq_config.max_quantile_values,
            max_channel_quantile_tokens=pq_config.max_channel_quantile_tokens,
        )
        module.set_channel_groups(labels, init_v=init_v, init_s=init_s)
        print(f"OHC init [{name}]: channels={labels.numel()}, groups={init_v.numel()}")
