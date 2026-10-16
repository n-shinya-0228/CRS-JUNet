"""Step C.3 optimizer grouping and read-only scalar diagnostics."""

from contextlib import contextmanager
import math

import torch
from torch import nn


def local_alpha_parameter(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    return getattr(core, "local_alpha", None)


def build_alpha_optimizer(model, train):
    alpha = local_alpha_parameter(model)
    if alpha is None or alpha.ndim != 0 or not alpha.requires_grad:
        raise ValueError("Step C.3 requires one trainable scalar local_alpha")
    parameters = list(model.parameters())
    if sum(parameter is alpha for parameter in parameters) != 1:
        raise ValueError("local_alpha must be registered exactly once")
    return torch.optim.AdamW(
        [{"params": [p for p in parameters if p is not alpha]},
         {"params": [alpha], "weight_decay": 0.0}],
        lr=train["lr"], weight_decay=train["w_decay"],
    )


@contextmanager
def local_alpha_statistics(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    result = {}
    if local_alpha_parameter(core) is None:
        yield result
        return
    parts = {}
    alphas = []
    device_alphas = {}

    def record(name, tensor):
        value = tensor.detach().float()
        variance, mean = torch.var_mean(value, unbiased=False)
        parts.setdefault(name, []).append((
            value.numel(), mean.item(), variance.item(), value.abs().mean().item(),
            torch.isfinite(value).all().item(),
        ))

    def capture_base(module, inputs):
        with torch.no_grad():
            # Hooks execute on each replica; use its own alpha and activations.
            device_alphas[inputs[0].device] = module.local_alpha.detach()
            alphas.append(module.local_alpha.detach().item())
            record("point_base", inputs[0])

    def capture_local(module, inputs, output):
        record("point_local", output)

    def residual(module, inputs, output):
        record("local_residual", output)
        record("scaled_local_residual", device_alphas[output.device] * output.detach())

    handles = []
    try:
        handles.append(core.point_context.local.register_forward_hook(residual))
        handles.append(core.point_context.register_forward_pre_hook(capture_base))
        handles.append(core.point_context.register_forward_hook(capture_local))
        yield result
    finally:
        for handle in handles:
            handle.remove()
        device_alphas.clear()
        for name, shards in parts.items():
            count = sum(n for n, _, _, _, _ in shards)
            mean = sum(n * m for n, m, _, _, _ in shards) / count
            variance = sum(n * (v + (m - mean) ** 2) for n, m, v, _, _ in shards) / count
            result[name] = dict(mean=mean, std=math.sqrt(max(variance, 0.0)),
                                abs_mean=sum(n * a for n, _, _, a, _ in shards) / count,
                                finite=all(f for _, _, _, _, f in shards))
        if alphas and "local_residual" in result:
            alpha = alphas[0]
            if any(value != alpha for value in alphas):
                raise RuntimeError("DataParallel replicas have different local_alpha values")
            result["alpha"] = alpha
            result["local_contribution_ratio"] = (
                result["scaled_local_residual"]["abs_mean"] /
                (result["point_base"]["abs_mean"] + 1e-8)
            )


def log_best_comparison(logger, reference, class_iou, miou):
    def value(number):
        return "N/A" if number is None else f"{number:.6f}"

    logger.info(
        "Best mIoU comparison: "
        f"Baseline={value(reference.get('baseline_8ch_miou'))} | "
        f"C.1b={value(reference.get('step_c1b_miou'))} | "
        f"C.2={value(reference.get('step_c2_miou'))} | C.3={miou:.6f}"
    )
    for name in reference.get("step_c2_class_iou", {}):
        logger.info(
            f"Best class comparison [{name}]: "
            f"Baseline={value(reference.get('baseline_8ch_class_iou', {}).get(name))} | "
            f"C.1b={value(reference.get('step_c1b_class_iou', {}).get(name))} | "
            f"C.2={value(reference['step_c2_class_iou'][name])} | "
            f"C.3={value(class_iou.get(name))}"
        )
