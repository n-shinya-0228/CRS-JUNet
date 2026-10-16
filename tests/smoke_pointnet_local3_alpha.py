"""Matched C.2/C.3 real-data preflight; never starts full training."""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import torch
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_pointnet_hybrid_gated_local3_alpha import Trainer
from trainer_PointNet_Gated_StableLR import scheduler_preview
from lib.utils.ioueval import iouEval
from lib.utils.pointnet_alpha_diagnostics import local_alpha_parameter
from smoke_pointnet_stablelr import SmokeBatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-c2", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-data smoke test")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger.add(output / "smoke.log")
    torch.set_num_threads(4)
    name = ("sjnet_pointnet_gated_local3_stablelr.yaml" if args.control_c2
            else "sjnet_pointnet_gated_local3_alpha_stablelr.yaml")
    with (ROOT / "config/arch" / name).open() as f:
        arch = yaml.safe_load(f)
    with (ROOT / "config/labels/semantic-kitti.yaml").open() as f:
        data = yaml.safe_load(f)

    with tempfile.TemporaryDirectory(prefix="stepc3_smoke_") as temporary:
        trainer = Trainer(arch, data, args.dataset, temporary, logger, resume=None, pretrained=None)
        model = trainer.model
        alpha = local_alpha_parameter(model)
        assert not trainer.multi_gpu, "This matched memory test expects a single GPU"
        assert trainer.parser.get_train_set().batch_size == 3
        assert trainer.w_paste == 0 and trainer.parser.copy_paste_prob == 0
        assert trainer.start_epoch == 0 and not trainer.optimizer.state
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable == 28876586 + int(not args.control_c2)
        if alpha is not None:
            assert alpha.ndim == 0 and alpha.item() == 1.0
            assert trainer.optimizer.param_groups[1]["params"][0] is alpha
            assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0
            assert trainer.optimizer.param_groups[0]["weight_decay"] == arch["train"]["w_decay"]
        evaluator = iouEval(trainer.parser.get_n_classes(), trainer.device,
                            [i for i, weight in enumerate(trainer.loss_w) if weight < 1e-10])
        shapes, first, checks, alpha_steps, losses = {}, {}, [], [], []

        def record(name, value, channels):
            if name not in shapes:
                assert value.shape == (3, channels, 512, 512), (name, value.shape)
                assert torch.isfinite(value).all()
                shapes[name] = list(value.shape)

        def output_hook(name, channels):
            def hook(module, inputs, value):
                record(name, value, channels)
            return hook

        def input_hook(name, channels):
            def hook(module, inputs):
                record(name, inputs[0], channels)
            return hook

        def input_parts(module, inputs):
            x = inputs[0]
            for name, value, channels in (("input", x, 24), ("hand", x[:, :7], 7),
                                           ("point", x[:, 7:23], 16), ("mask", x[:, 23:24], 1)):
                record(name, value, channels)

        def residual(module, inputs, value):
            if "local_residual" not in shapes:
                record("local_residual", value, 16)
                first["base"], first["residual"] = inputs[0].detach(), value.detach()

        def local(module, inputs, value):
            if "point_local" not in shapes:
                record("point_local", value, 16)
                with torch.no_grad():
                    scale = 1.0 if alpha is None else alpha.detach()
                    if alpha is not None:
                        assert scale.item() == 1.0
                        shapes["alpha"] = list(scale.shape)
                    scaled = scale * first["residual"]
                    record("scaled_local_residual", scaled, 16)
                    expected = first["base"] + first["residual"]
                    assert torch.allclose(first["base"] + scaled, expected, atol=1e-6)
                    assert torch.allclose(value, expected, atol=1e-6)
                    first.clear()

        def outputs(module, inputs, value):
            record("logits", value["logits"], 20)

        handles = [model.register_forward_pre_hook(input_parts),
                   model.hand_stem.register_forward_hook(output_hook("hand_feat", 16)),
                   model.point_stem.register_forward_hook(output_hook("point_base", 16)),
                   model.point_context.local.register_forward_hook(residual),
                   model.point_context.register_forward_hook(local),
                   model.point_gate.register_forward_hook(output_hook("gate", 16)),
                   model.stem_feat.register_forward_pre_hook(input_hook("fusion_input", 32)),
                   model.stem_feat.register_forward_hook(output_hook("fusion_output", 48)),
                   model.register_forward_hook(outputs)]
        original_mix = trainer._mix_losses

        def finite_loss(*args, **kwargs):
            loss = original_mix(*args, **kwargs)
            assert torch.isfinite(loss).all()
            losses.append(loss.detach().item())
            return loss

        trainer._mix_losses = finite_loss

        def gradient_check(optimizer, args, kwargs):
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
            assert all(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in model.point_context.parameters())
            checks.append(len(gradients))
            if alpha is not None:
                assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
                alpha_steps.append(dict(alpha=alpha.item(), grad=alpha.grad.item()))
                logger.info(f"Smoke local_alpha={alpha.item():.9f} | local_alpha.grad={alpha.grad.item():.9f} "
                            "(unscaled, after existing gradient clipping)")

        gradient_handle = trainer.optimizer.register_step_pre_hook(gradient_check)
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer.train_epoch(SmokeBatches(trainer.parser.train_dataset, 3, 3), model,
                                trainer.optimizer, 0, evaluator, color_fn=None, report=1)
            torch.cuda.synchronize()
            assert len(checks) == 3
            train_peak = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            for handle in handles:
                handle.remove()
            gradient_handle.remove()
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            trainer.validate(SmokeBatches(trainer.parser.valid_dataset, 3, 2), model,
                             evaluator, trainer.parser.get_xentropy_class_string)
            torch.cuda.synchronize()
            assert all(item["finite"] for item in trainer.last_local_statistics.values())
            assert all(math.isfinite(v) for v in trainer.last_gate_statistics.values())
            alpha_stats = getattr(trainer, "last_alpha_statistics", {})
            for stats in alpha_stats.values():
                assert stats["finite"] if isinstance(stats, dict) else math.isfinite(stats)
            if alpha is not None:
                assert alpha.item() != 1.0
            report = dict(
                status="PASS", model=arch["model"]["name"], gpu=torch.cuda.get_device_name(), shapes=shapes,
                total_parameters=total, trainable_parameters=trainable,
                added_parameters_vs_c2=total - 28876586, batch_size=3,
                copy_paste_prob=0.0, w_paste=0.0, paste_mask_sum=0.0,
                train_batches=3, validation_batches=2, gradient_checks=checks,
                alpha_initial=1.0 if alpha is not None else None, alpha_steps=alpha_steps,
                alpha_after_updates=alpha.item() if alpha is not None else None,
                alpha_diagnostics=alpha_stats,
                optimizer_groups=[dict(parameters=sum(p.numel() for p in group["params"]),
                                       weight_decay=group["weight_decay"], lr=group["lr"])
                                  for group in trainer.optimizer.param_groups],
                losses_train_then_valid=losses, gate=trainer.last_gate_statistics,
                local_features=trainer.last_local_statistics,
                train_peak_allocated_GiB=train_peak, train_peak_reserved_GiB=reserved,
                validation_peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                lr_preview=scheduler_preview(arch["train"]), resume=None, pretrained=None,
            )
        except torch.cuda.OutOfMemoryError:
            (output / "report.json").write_text(json.dumps(dict(status="OOM", batch_size=3)))
            raise
        finally:
            trainer.writer.close()
            for handle in handles:
                handle.remove()
            gradient_handle.remove()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
