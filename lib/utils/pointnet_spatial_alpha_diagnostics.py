"""Detached C.4 diagnostics: one batch of maps, online full-validation moments."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from threading import Lock

import torch
from torch import nn


def spatial_alpha_module(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    return getattr(getattr(core, "point_context", None), "alpha", None)


def build_spatial_alpha_optimizer(model, train):
    alpha = spatial_alpha_module(model)
    if alpha is None or alpha.alpha_global.ndim != 0:
        raise ValueError("C.4 requires a scalar global alpha and a 16->1 predictor")
    scale_params = [alpha.alpha_global, alpha.alpha_predictor.weight]
    if sum(p.numel() for p in scale_params) != 17 or not all(p.requires_grad for p in scale_params):
        raise ValueError("Expected exactly 17 trainable alpha parameters")
    ids = {id(p) for p in scale_params}
    params = list(model.parameters())
    if sum(id(p) in ids for p in params) != 2:
        raise ValueError("Alpha parameters must each be registered exactly once")
    return torch.optim.AdamW(
        [{"params": [p for p in params if id(p) not in ids]},
         {"params": scale_params, "weight_decay": 0.0}],
        lr=train["lr"], weight_decay=train["w_decay"],
    )


class RunningMoments:
    """Merge batch population moments without retaining any tensor or map."""

    def __init__(self):
        self.count = self.finite_count = self.negative = self.gt3 = 0
        self.mean = self.m2 = self.abs_sum = 0.0
        self.minimum, self.maximum = math.inf, -math.inf
        self._lock = Lock()

    def update(self, tensor):
        values = tensor.detach().reshape(-1).double()
        count = values.numel()
        finite = torch.isfinite(values)
        values = values[finite]
        n = values.numel()
        if not n:
            with self._lock:
                self.count += count
            return
        variance, mean = torch.var_mean(values, unbiased=False)
        mean, variance = mean.item(), variance.item()
        abs_sum, minimum, maximum = values.abs().sum().item(), values.min().item(), values.max().item()
        negative, gt3 = (values < 0).sum().item(), (values > 3).sum().item()
        with self._lock:
            self.count += count
            combined = self.finite_count + n
            delta = mean - self.mean
            self.m2 += n * variance + delta * delta * self.finite_count * n / combined
            self.mean += delta * n / combined
            self.finite_count = combined
            self.abs_sum += abs_sum
            self.minimum = min(self.minimum, minimum)
            self.maximum = max(self.maximum, maximum)
            self.negative += negative
            self.gt3 += gt3

    def summary(self, min_count=1):
        finite = self.finite_count == self.count
        usable = finite and self.count >= min_count
        return dict(
            count=self.count, finite_count=self.finite_count, finite=finite,
            mean=self.mean if usable else None,
            std=math.sqrt(max(self.m2 / self.finite_count, 0.0)) if usable else None,
            min=self.minimum if usable else None, max=self.maximum if usable else None,
            abs_mean=self.abs_sum / self.finite_count if usable else None,
            negative_fraction=self.negative / self.count if usable else None,
            gt3_fraction=self.gt3 / self.count if usable else None,
        )


def tensor_statistics(tensor, quantiles=False):
    moments = RunningMoments()
    moments.update(tensor)
    result = moments.summary()
    if quantiles:
        values = tensor.detach().float().reshape(-1).cpu()
        percentiles = (torch.quantile(values, torch.tensor([.05, .50, .95])).tolist()
                       if values.numel() and result["finite"] else [None] * 3)
        result.update(zip(("p05", "p50", "p95"), percentiles))
    return result


@contextmanager
def spatial_alpha_capture(model, detailed=False):
    """Capture predictions only; labels are never available to model hooks."""
    alpha = spatial_alpha_module(model)
    result = {}
    if alpha is None:
        yield result
        return
    core = model.module if isinstance(model, nn.DataParallel) else model
    maps, delta_maps, residuals = {}, {}, {}
    features = {name: RunningMoments() for name in
                ("point_base", "local_residual", "scaled_local_residual", "point_local")}

    def local_hook(module, inputs, output):
        features["point_base"].update(inputs[0])
        features["local_residual"].update(output)
        residuals[output.device] = output.detach()

    def delta_hook(module, inputs, output):
        delta_maps[output.device] = output.detach().float().cpu()

    def alpha_hook(module, inputs, output):
        maps[output.device] = output.detach().float().cpu()
        if detailed:
            scaled = output.detach() * residuals.pop(output.device)
            features["scaled_local_residual"].update(scaled)

    def context_hook(module, inputs, output):
        features["point_local"].update(output)

    def gather(shards):
        # Match DataParallel's scatter/gather order, including non-sorted device_ids.
        devices = ([torch.device("cuda", index) for index in model.device_ids]
                   if isinstance(model, nn.DataParallel) and model.device_ids else list(shards))
        return torch.cat([shards[device] for device in devices if device in shards], dim=0)

    handles = []
    try:
        handles.append(alpha.register_forward_hook(alpha_hook))
        if detailed:
            handles.append(core.point_context.local.register_forward_hook(local_hook))
            handles.append(alpha.alpha_predictor.register_forward_hook(delta_hook))
            handles.append(core.point_context.register_forward_hook(context_hook))
        yield result
    finally:
        for handle in handles:
            handle.remove()
        residuals.clear()
        if maps:
            result["alpha_map"] = gather(maps)
            if detailed:
                result["alpha_delta"] = tensor_statistics(gather(delta_maps))
                result["features"] = {name: value.summary() for name, value in features.items()}
                base = result["features"]["point_base"]["abs_mean"]
                scaled = result["features"]["scaled_local_residual"]["abs_mean"]
                result["local_contribution_ratio"] = (scaled / (base + 1e-8)
                                                      if base is not None and scaled is not None else None)
        maps.clear()
        delta_maps.clear()


class SpatialAlphaValidation:
    def __init__(self, model, class_names, min_class_cells=5):
        if min_class_cells < 1:
            raise ValueError("min_class_cells must be positive")
        self.class_names = dict(class_names)
        self.min_class_cells = int(min_class_cells)
        self.batches = 0
        self.maps, self.observed, self.unobserved = (RunningMoments() for _ in range(3))
        self.classes = {cid: RunningMoments() for cid in class_names}
        alpha = spatial_alpha_module(model)
        value = alpha.alpha_global.detach().item()
        norm = alpha.alpha_predictor.weight.detach().float().norm().item()
        self.global_value = value if math.isfinite(value) else None
        self.delta_norm = norm if math.isfinite(norm) else None
        self.batch0 = None

    def update(self, capture, labels, mask):
        # This runs AFTER forward; consume the map so only scalar moments survive.
        alpha = capture.pop("alpha_map").squeeze(1)
        labels = labels.detach().cpu()
        observed = mask.detach().cpu()
        if observed.ndim == 4:
            observed = observed.squeeze(1)
        observed = observed > 0
        if alpha.shape != labels.shape or alpha.shape != observed.shape:
            raise ValueError("Alpha, validation labels and observation mask must align")
        self.maps.update(alpha)
        observed_alpha, observed_labels = alpha[observed], labels[observed]
        self.observed.update(observed_alpha)
        self.unobserved.update(alpha[~observed])
        for cid, moments in self.classes.items():
            moments.update(observed_alpha[observed_labels == cid])
        if self.batches == 0:
            self.batch0 = dict(capture)
            self.batch0.update(alpha_map=tensor_statistics(alpha, quantiles=True),
                               observed=tensor_statistics(alpha[observed]),
                               unobserved=tensor_statistics(alpha[~observed]))
        self.batches += 1

    def summary(self):
        return dict(
            scope="full_validation", batches=self.batches,
            alpha_global=self.global_value, alpha_delta_norm=self.delta_norm,
            alpha_map=self.maps.summary(), observed=self.observed.summary(),
            unobserved=self.unobserved.summary(), min_class_cells=self.min_class_cells,
            class_alpha={self.class_names[cid]: moments.summary(self.min_class_cells)
                         for cid, moments in self.classes.items()},
            batch0=self.batch0,
        )


def _number(value):
    return "N/A" if value is None else f"{value:.6f}"


def log_spatial_validation(logger, writer, summary, epoch):
    logger.info(f"Spatial alpha (evaluated model): global={_number(summary['alpha_global'])} | "
                f"delta weight norm={_number(summary['alpha_delta_norm'])}")
    for scope in ("alpha_map", "observed", "unobserved"):
        logger.info(f"Spatial alpha full validation [{scope}]: {summary[scope]}")
        for name, value in summary[scope].items():
            if value is not None:
                writer.add_scalar(f"validating/spatial_alpha/{scope}/{name}", value, epoch)
    logger.info(f"Spatial alpha validation batch 0: {summary['batch0']}")
    for name in ("alpha_global", "alpha_delta_norm"):
        if summary[name] is not None:
            writer.add_scalar(f"validating/spatial_alpha/{name}", summary[name], epoch)
    for name, stats in summary["class_alpha"].items():
        logger.info(f"Class alpha [{name}]: mean={_number(stats['mean'])} | "
                    f"std={_number(stats['std'])} | cells={stats['count']} | finite={stats['finite']}")
        for key in ("mean", "std", "count"):
            if stats[key] is not None:
                writer.add_scalar(f"validating/class_alpha/{name}/{key}", stats[key], epoch)
    stats = summary["alpha_map"]
    if (not stats["finite"] or summary["alpha_global"] is None or summary["alpha_delta_norm"] is None
            or (stats["min"] is not None and stats["min"] <= -5)
            or (stats["max"] is not None and stats["max"] >= 10)):
        logger.warning("Extreme/non-finite spatial alpha detected; no clamp or mask applied")


def save_spatial_best_summary(logdir, summary, reference, logger):
    payload = dict(summary, references=reference)
    path = Path(logdir) / "best_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    alpha = summary["spatial_alpha"]
    logger.info(f"Best C.4: epoch={summary['epoch']} | mIoU={summary['miou']:.6f} | "
                f"acc={summary['acc']:.6f} | loss={summary['loss']:.6f} | "
                f"alpha_global={_number(alpha['alpha_global'])} | "
                f"alpha_map mean/std={_number(alpha['alpha_map']['mean'])}/{_number(alpha['alpha_map']['std'])} | "
                f"delta norm={_number(alpha['alpha_delta_norm'])}")
    logger.info("Best mIoU comparison: " + " | ".join(
        f"{label}={_number(reference.get(key))}" for label, key in
        (("Baseline", "baseline_8ch_miou"), ("C.1b", "step_c1b_miou"),
         ("C.2", "step_c2_miou"), ("C.3", "step_c3_miou"))) + f" | C.4={summary['miou']:.6f}")
    for name, stats in alpha["class_alpha"].items():
        columns = " | ".join(
            f"{label}={_number(reference.get(key, {}).get(name))}" for label, key in
            (("Baseline", "baseline_8ch_class_iou"), ("C.1b", "step_c1b_class_iou"),
             ("C.2", "step_c2_class_iou"), ("C.3", "step_c3_class_iou")))
        logger.info(f"Best class comparison [{name}]: {columns} | "
                    f"C.4={_number(summary['class_iou'].get(name))} | "
                    f"alpha mean={_number(stats['mean'])} | std={_number(stats['std'])} | cells={stats['count']}")
