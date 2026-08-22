from __future__ import annotations

import torch
import torch.nn as nn

from ahcqsam.quantization.fake_quant import QuantizeBase
from ahcqsam.quantization.util_quant import fake_quantize_learnable_per_tensor_affine_training


class PQSAMGADTFakeQuantize(QuantizeBase):
    """PQ-SAM activation quantizer with learnable grouped shift and scale."""

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))
        self.gadt_v = nn.Parameter(torch.tensor([0.0], dtype=torch.float))
        self.gadt_s = nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("labels", torch.empty(0, dtype=torch.long))
        self.register_buffer("eps", torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0
        self.channel_axis = ch_axis

    def set_channel_groups(self, labels, init_v=None, init_s=None):
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.scale.device)
        if labels.ndim != 1:
            raise ValueError("PQSAMGADTFakeQuantize expects 1-D channel group labels.")
        num_groups = int(labels.max().item()) + 1 if labels.numel() > 0 else 1
        self.labels = labels

        if init_v is None:
            init_v = torch.zeros(num_groups, dtype=self.scale.dtype, device=self.scale.device)
        else:
            init_v = torch.as_tensor(init_v, dtype=self.scale.dtype, device=self.scale.device)
        if init_s is None:
            init_s = torch.ones(num_groups, dtype=self.scale.dtype, device=self.scale.device)
        else:
            init_s = torch.as_tensor(init_s, dtype=self.scale.dtype, device=self.scale.device)

        if init_v.numel() != num_groups or init_s.numel() != num_groups:
            raise ValueError("GADT init parameter size must match the number of channel groups.")
        self.gadt_v = nn.Parameter(init_v.reshape(num_groups))
        self.gadt_s = nn.Parameter(init_s.reshape(num_groups))

    def _resolve_channel_axis(self, x):
        if self.channel_axis == "det":
            return x.ndim - 1
        if self.channel_axis == -1:
            return x.ndim - 1
        return self.channel_axis

    def _mapped_gadt_params(self, x):
        ch_axis = self._resolve_channel_axis(x)
        if self.labels.numel() == x.shape[ch_axis]:
            labels = self.labels.to(self.gadt_v.device)
            v = self.gadt_v[labels]
            s = self.gadt_s[labels]
        else:
            v = self.gadt_v
            s = self.gadt_s

        v = v.to(device=x.device, dtype=x.dtype)
        s = s.to(device=x.device, dtype=x.dtype)
        if v.numel() == 1:
            return v, s

        view_shape = [1] * x.ndim
        view_shape[ch_axis] = x.shape[ch_axis]
        return v.reshape(view_shape), s.reshape(view_shape)

    def gadt_transform(self, x):
        v, s = self._mapped_gadt_params(x)
        s = s.abs().clamp(min=self.eps.item())
        return (x - v) / s

    def gadt_inverse_transform(self, x):
        v, s = self._mapped_gadt_params(x)
        s = s.abs().clamp(min=self.eps.item())
        return x * s + v

    def forward(self, x, value=None):
        if self.observer_enabled == 1:
            self.observer(self.gadt_transform(x).detach())
            scale, zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            scale = scale.to(self.scale.device)
            zero_point = zero_point.to(self.zero_point.device)
            self.scale.data = torch.ones_like(scale)
            self.scale.data.copy_(scale)
            self.zero_point.resize_(zero_point.shape)
            self.zero_point.copy_(zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = x
            x = self.gadt_transform(x)
            if self.use_grad_scaling:
                grad_factor = 1.0 / (x.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            x = fake_quantize_learnable_per_tensor_affine_training(
                x, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor
            )
            x = self.gadt_inverse_transform(x)
            if self.drop_prob < 1.0:
                return torch.where(torch.rand_like(x) < self.drop_prob, x, x_orig)
        return x
