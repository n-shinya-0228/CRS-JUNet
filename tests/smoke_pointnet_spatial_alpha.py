"""C.4 preflight on real data, with optional matched C.2 control. No full training."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import torch
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_pointnet_hybrid_gated_local3_spatial_alpha import Trainer
from trainer_PointNet_Gated_StableLR import scheduler_preview, validate_resume_checkpoint
from lib.utils.ioueval import iouEval
from lib.utils.pointnet_spatial_alpha_diagnostics import spatial_alpha_module
from smoke_pointnet_stablelr import SmokeBatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-c2", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-data preflight")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger.add(output / "smoke.log")
    torch.set_num_threads(4)
    name = ("sjnet_pointnet_gated_local3_stablelr.yaml" if args.control_c2
            else "sjnet_pointnet_gated_local3_spatial_alpha_stablelr.yaml")
    with (ROOT / "config/arch" / name).open() as f:
        arch = yaml.safe_load(f)
    with (ROOT / "config/labels/semantic-kitti.yaml").open() as f:
        data = yaml.safe_load(f)
    with tempfile.TemporaryDirectory(prefix="stepc4_smoke_") as temporary:
        trainer = Trainer(arch, data, args.dataset, temporary, logger, resume=None, pretrained=None)
        model = trainer.model
        alpha = spatial_alpha_module(model)
        assert not trainer.multi_gpu, "The GPU-memory smoke test expects one GPU"
        assert trainer.parser.get_train_set().batch_size == 3
        assert trainer.w_paste == trainer.parser.copy_paste_prob == 0
        assert trainer.start_epoch == 0 and not trainer.optimizer.state
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable == 28876586 + (0 if args.control_c2 else 17)
        if alpha is not None:
            assert alpha.alpha_global.shape == torch.Size([]) and alpha.alpha_global.item() == 1
            assert alpha.alpha_predictor.weight.count_nonzero().item() == 0
        evaluator = iouEval(trainer.parser.get_n_classes(), trainer.device,
                            [i for i, weight in enumerate(trainer.loss_w) if weight < 1e-10])
        shapes, first, checks, alpha_steps, losses = {}, {}, [], [], []

        def record(name, value, channels):
            if name not in shapes:
                assert value.shape == (3, channels, 512, 512), (name, value.shape)
                assert torch.isfinite(value).all()
                shapes[name] = list(value.shape)

        def out_hook(name, channels):
            def hook(module, inputs, value):
                record(name, value, channels)
            return hook

        def in_hook(name, channels):
            def hook(module, inputs):
                record(name, inputs[0], channels)
            return hook

        def input_parts(module, inputs):
            x = inputs[0]
            record("input", x, 24)
            record("hand", x[:, :7], 7)
            record("point", x[:, 7:23], 16)
            record("mask", x[:, 23:24], 1)

        def residual(module, inputs, value):
            if "local_residual" not in shapes:
                record("local_residual", value, 16)
                first.update(base=inputs[0].detach(), residual=value.detach())

        def alpha_hook(module, inputs, value):
            assert torch.isfinite(value).all()
            if "alpha_map" not in shapes:
                record("alpha_map", value, 1)
                assert torch.equal(value, torch.ones_like(value))
                first["alpha"] = value.detach()

        def local(module, inputs, value):
            if "point_local" not in shapes:
                record("point_local", value, 16)
                with torch.no_grad():
                    scale = first.get("alpha", 1.0)
                    scaled = scale * first["residual"]
                    record("scaled_local_residual", scaled, 16)
                    assert torch.allclose(value, first["base"] + first["residual"], atol=1e-6)
                    assert torch.allclose(value, first["base"] + scaled, atol=1e-6)
                first.clear()

        def outputs(module, inputs, value):
            record("logits", value["logits"], 20)

        handles = [model.register_forward_pre_hook(input_parts),
                   model.hand_stem.register_forward_hook(out_hook("hand_feat", 16)),
                   model.point_stem.register_forward_hook(out_hook("point_base", 16)),
                   model.point_context.local.register_forward_hook(residual),
                   model.point_context.register_forward_hook(local),
                   model.point_gate.register_forward_hook(out_hook("gate", 16)),
                   model.stem_feat.register_forward_pre_hook(in_hook("fusion_input", 32)),
                   model.stem_feat.register_forward_hook(out_hook("fusion_output", 48)),
                   model.register_forward_hook(outputs)]
        if alpha is not None:
            shapes["alpha_global"] = list(alpha.alpha_global.shape)
            handles += [alpha.register_forward_hook(alpha_hook),
                        alpha.alpha_predictor.register_forward_hook(out_hook("alpha_delta", 1))]
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
            checks.append(len(gradients))
            if alpha is not None:
                global_grad, predictor_grad = alpha.alpha_global.grad, alpha.alpha_predictor.weight.grad
                assert global_grad is not None and predictor_grad is not None
                assert torch.isfinite(global_grad).all() and torch.isfinite(predictor_grad).all()
                assert global_grad.abs().item() > 0 and predictor_grad.norm().item() > 0
                check = dict(alpha_global=alpha.alpha_global.item(), global_grad=global_grad.item(),
                             predictor_grad_norm=predictor_grad.norm().item(),
                             predictor_grad=predictor_grad.flatten().tolist())
                alpha_steps.append(check)
                logger.info(f"Alpha gradient check (unscaled, after existing clip): {check}")

        expected_counts = {cid: 0 for cid in range(20) if not data["learning_ignore"][cid]}

        class AuditedValidationBatches(SmokeBatches):
            def __iter__(self):
                for batch in super().__iter__():
                    for cid in expected_counts:
                        expected_counts[cid] += ((batch[2] == cid) & (batch[1][:, 0] > 0)).sum().item()
                    yield batch

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
            if alpha is not None:
                assert alpha.alpha_global.item() != 1.0
                assert alpha.alpha_predictor.weight.count_nonzero().item() > 0
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            acc, miou, loss = trainer.validate(AuditedValidationBatches(trainer.parser.valid_dataset, 3, 2),
                                              model, evaluator, trainer.parser.get_xentropy_class_string)
            torch.cuda.synchronize()
            validation_peak = torch.cuda.max_memory_allocated() / 2**30
            assert all(item["finite"] for item in trainer.last_local_statistics.values())
            assert all(math.isfinite(v) for v in trainer.last_gate_statistics.values())
            spatial = getattr(trainer, "last_spatial_statistics", {})
            if alpha is not None:
                assert spatial["alpha_map"]["finite"] and spatial["alpha_map"]["std"] > 0
                assert len(spatial["class_alpha"]) == 19 and spatial["batches"] == 2
                for cid, count in expected_counts.items():
                    name = trainer.parser.get_xentropy_class_string(cid)
                    assert spatial["class_alpha"][name]["count"] == count
                trainer.best_val_epoch = 0
                trainer.early_stopping.update(miou, 0)
                trainer._save_checkpoint(0, 0.0, miou, is_best=True)
                trainer._save_checkpoint(0, 0.0, miou)
                for filename in ("best_val.path", "latest.path"):
                    state = torch.load(Path(temporary) / filename, map_location="cpu", weights_only=False)
                    validate_resume_checkpoint(state, arch, data)
                    for key in ("point_context.alpha.alpha_global", "point_context.alpha.alpha_predictor.weight"):
                        torch.testing.assert_close(state.get("model_train", state["model"])[key],
                                                   model.state_dict()[key].cpu(), rtol=0, atol=0)
                        torch.testing.assert_close(state["model_ema"][key],
                                                   trainer.ema_model.state_dict()[key].cpu(), rtol=0, atol=0)
                    assert state["best_spatial_summary"]["spatial_alpha"] == spatial
                    del state
                shutil.copyfile(Path(temporary) / "best_summary.json", output / "smoke_best_summary.json")
            report = dict(
                status="PASS", model=arch["model"]["name"], gpu=torch.cuda.get_device_name(),
                shapes=shapes, total_parameters=total, trainable_parameters=trainable,
                added_parameters_vs_c2=total - 28876586, batch_size=3, train_batches=3, validation_batches=2,
                gradient_checks=checks, alpha_gradient_steps=alpha_steps,
                alpha_global_after_updates=alpha.alpha_global.item() if alpha is not None else None,
                alpha_predictor_norm_after_updates=alpha.alpha_predictor.weight.norm().item() if alpha is not None else None,
                spatial_alpha=spatial, gate=trainer.last_gate_statistics,
                optimizer_groups=[dict(parameters=sum(p.numel() for p in g["params"]), weight_decay=g["weight_decay"], lr=g["lr"])
                                  for g in trainer.optimizer.param_groups],
                copy_paste_prob=0, w_paste=0, paste_mask_sum=0, losses_train_then_valid=losses,
                train_peak_allocated_GiB=train_peak, train_peak_reserved_GiB=reserved,
                validation_peak_allocated_GiB=validation_peak,
                lr_preview=scheduler_preview(arch["train"]), resume=None, pretrained=None,
                checkpoint_alpha_roundtrip=alpha is not None,
            )
        except torch.cuda.OutOfMemoryError:
            (output / "report.json").write_text(json.dumps(dict(status="OOM", batch_size=3)))
            raise
        finally:
            trainer.writer.close()
            for handle in handles:
                handle.remove()
            gradient_handle.remove()
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
