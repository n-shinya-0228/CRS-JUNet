"""Matched C.1b/C.2 real-data smoke tests; never starts full training."""

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
from train_pointnet_hybrid_gated_local3 import Trainer
from trainer_PointNet_Gated_StableLR import scheduler_preview
from lib.utils.ioueval import iouEval
from smoke_pointnet_stablelr import SmokeBatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", action="store_true", help="Measure unchanged C.1b as control")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger.add(output / "smoke.log")
    torch.set_num_threads(4)
    config_name = ("sjnet_pointnet_gated70_stablelr.yaml" if args.baseline
                   else "sjnet_pointnet_gated_local3_stablelr.yaml")
    with (ROOT / "config/arch" / config_name).open() as f:
        arch = yaml.safe_load(f)
    with (ROOT / "config/labels/semantic-kitti.yaml").open() as f:
        data = yaml.safe_load(f)
    with tempfile.TemporaryDirectory(prefix="stepc2_smoke_") as temporary:
        trainer = Trainer(arch, data, args.dataset, temporary, logger, resume=None, pretrained=None)
        model = trainer.model
        assert trainer.parser.get_train_set().batch_size == 3
        assert len(trainer.parser.train_dataset) == 19130
        assert trainer.parser.get_train_size() == 6376
        assert trainer.w_paste == 0 and trainer.parser.copy_paste_prob == 0
        assert trainer.start_epoch == 0 and not trainer.optimizer.state
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected_total = 28874250 + (0 if args.baseline else 2336)
        assert total == trainable == expected_total
        evaluator = iouEval(trainer.parser.get_n_classes(), trainer.device,
                            [i for i, weight in enumerate(trainer.loss_w) if weight < 1e-10])
        shapes, checks, losses = {}, [], []

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
            record("input", x, 24)
            record("hand", x[:, 0:7], 7)
            record("point", x[:, 7:23], 16)
            record("mask", x[:, 23:24], 1)

        def outputs(module, inputs, value):
            record("logits", value["logits"], 20)

        handles = [model.register_forward_pre_hook(input_parts),
                   model.hand_stem.register_forward_hook(output_hook("hand_feat", 16)),
                   model.point_stem.register_forward_hook(output_hook("point_base", 16)),
                   model.point_gate.register_forward_hook(output_hook("gate", 16)),
                   model.stem_feat.register_forward_pre_hook(input_hook("fusion_input", 32)),
                   model.stem_feat.register_forward_hook(output_hook("fusion_output", 48)),
                   model.register_forward_hook(outputs)]
        if not args.baseline:
            handles.extend([model.point_context.local.register_forward_hook(output_hook("local", 16)),
                            model.point_context.register_forward_hook(output_hook("point_local", 16))])
        original_mix = trainer._mix_losses

        def finite_loss(*args, **kwargs):
            loss = original_mix(*args, **kwargs)
            assert torch.isfinite(loss).all()
            losses.append(loss.detach().item())
            return loss

        trainer._mix_losses = finite_loss

        def gradient_check(optimizer, args, kwargs):
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(grad).all().item() for grad in gradients)
            if not args_baseline:
                assert all(p.grad is not None and p.grad.abs().sum().item() > 0
                           for p in model.point_context.parameters())
            checks.append(len(gradients))

        args_baseline = args.baseline
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
            local_stats = getattr(trainer, "last_local_statistics", {})
            assert all(item["finite"] for item in local_stats.values())
            assert all(math.isfinite(v) for v in trainer.last_gate_statistics.values())
            if not args.baseline:
                assert local_stats["point_input_unobserved"]["nonzero_fraction"] == 0
            report = dict(
                status="PASS", model=arch["model"]["name"], shapes=shapes,
                total_parameters=total, trainable_parameters=trainable,
                added_parameters=total - 28874250, batch_size=3,
                train_scans=19130, train_batches_per_epoch=6376,
                copy_paste_prob=0.0, w_paste=0.0, paste_mask_sum=0.0,
                train_batches=3, validation_batches=2, gradient_checks=checks,
                losses_train_then_valid=losses, gate=trainer.last_gate_statistics,
                local_features=local_stats, train_peak_allocated_GiB=train_peak,
                train_peak_reserved_GiB=reserved,
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
