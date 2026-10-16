"""Read-only Local3 diagnostics; keep activations out of persistent model state."""

from contextlib import contextmanager
import math

import torch
from torch import nn


@contextmanager
def local3_feature_statistics(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    result = {}
    if not hasattr(core, "point_context"):
        yield result
        return

    masks, parts, gate_parts = {}, {}, []

    def record(name, tensor):
        value = tensor.detach().float()
        if not value.numel():
            parts.setdefault(name, [])
            return
        variance, mean = torch.var_mean(value, unbiased=False)
        parts.setdefault(name, []).append((
            value.numel(), mean.item(), variance.item(),
            torch.count_nonzero(value).item(), torch.isfinite(value).all().item(),
        ))

    def record_feature(name, tensor):
        record(name, tensor)
        mask = masks[tensor.device]
        unobserved = tensor.detach().permute(0, 2, 3, 1)[~mask[:, 0]]
        record(name + "_unobserved", unobserved)

    def capture_mask(module, inputs):
        x = inputs[0].detach()
        masks[x.device] = x[:, 23:24] > 0
        record_feature("point_input", x[:, 7:23])

    def capture_base(module, inputs):
        record_feature("point_base", inputs[0])

    def capture_residual(module, inputs, output):
        record_feature("local_residual", output)

    def capture_local(module, inputs, output):
        record_feature("point_local", output)

    def capture_gate(module, inputs, output):
        value = output.detach()
        gate_parts.append((value.numel(), (value <= 0.01).sum().item(),
                           (value >= 0.99).sum().item(), torch.isfinite(value).all().item()))

    handles = []
    try:
        handles.append(core.register_forward_pre_hook(capture_mask))
        handles.append(core.point_context.register_forward_pre_hook(capture_base))
        handles.append(core.point_context.local.register_forward_hook(capture_residual))
        handles.append(core.point_context.register_forward_hook(capture_local))
        handles.append(core.point_gate.register_forward_hook(capture_gate))
        yield result
    finally:
        for handle in handles:
            handle.remove()
        masks.clear()
        for name, shards in parts.items():
            count = sum(n for n, _, _, _, _ in shards)
            if count:
                mean = sum(n * m for n, m, _, _, _ in shards) / count
                variance = sum(n * (v + (m - mean) ** 2) for n, m, v, _, _ in shards) / count
                result[name] = dict(mean=mean, std=math.sqrt(max(variance, 0.0)),
                                    nonzero_fraction=sum(nz for _, _, _, nz, _ in shards) / count,
                                    finite=all(finite for _, _, _, _, finite in shards))
            else:
                result[name] = dict(mean=None, std=None, nonzero_fraction=0.0, finite=True)
        if gate_parts:
            count = sum(n for n, _, _, _ in gate_parts)
            result["gate_saturation"] = dict(
                le_001_fraction=sum(low for _, low, _, _ in gate_parts) / count,
                ge_099_fraction=sum(high for _, _, high, _ in gate_parts) / count,
                finite=all(finite for _, _, _, finite in gate_parts),
            )
