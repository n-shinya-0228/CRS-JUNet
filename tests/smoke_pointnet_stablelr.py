"""Step C.1b preflight: three real training batches and two validation batches."""

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
from train_pointnet_hybrid_gated_stablelr import Trainer
from trainer_PointNet_Gated_StableLR import scheduler_preview
from lib.dataset.Parser_PointNet import bev_collate_fn
from lib.utils.ioueval import iouEval


class SmokeBatches:
    def __init__(self, dataset, batch_size, batches):
        self.dataset, self.batch_size, self.batches = dataset, batch_size, batches

    def __len__(self):
        return self.batches

    def __iter__(self):
        for batch_index in range(self.batches):
            start = batch_index * self.batch_size
            batch = bev_collate_fn([self.dataset[i] for i in range(start, start + self.batch_size)])
            assert batch[0].shape == (3, 23, 512, 512)
            assert batch[1].shape == (3, 1, 512, 512)
            assert batch[14].count_nonzero().item() == 0
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--output", required=True, help="New preflight directory, NOT the training log")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-data smoke test")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger.add(output / "smoke.log")
    torch.set_num_threads(4)
    with (ROOT / "config/arch/sjnet_pointnet_gated70_stablelr.yaml").open() as f:
        arch = yaml.safe_load(f)
    with (ROOT / "config/labels/semantic-kitti.yaml").open() as f:
        data = yaml.safe_load(f)
    with tempfile.TemporaryDirectory(prefix="stepc1b_smoke_") as temporary:
        trainer = Trainer(arch, data, args.dataset, temporary, logger, pretrained=None, resume=None)
        assert len(trainer.parser.train_dataset) == 19130
        assert trainer.parser.get_train_size() == 6376
        assert trainer.parser.get_train_set().batch_size == 3
        assert trainer.w_paste == 0 and trainer.parser.copy_paste_prob == 0
        assert trainer.start_epoch == 0 and not trainer.optimizer.state
        assert trainer.optimizer.param_groups[0]["lr"] == 2e-4
        evaluator = iouEval(trainer.parser.get_n_classes(), trainer.device,
                            [i for i, weight in enumerate(trainer.loss_w) if weight < 1e-10])
        gradient_checks = []
        step_losses = []
        original_mix = trainer._mix_losses

        def finite_loss(*args, **kwargs):
            loss = original_mix(*args, **kwargs)
            assert torch.isfinite(loss).all()
            step_losses.append(float(loss.detach()))
            return loss

        trainer._mix_losses = finite_loss

        def check_gradients(optimizer, args, kwargs):
            grads = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
            assert grads and all(torch.isfinite(gradient).all().item() for gradient in grads)
            for weight in (trainer.model.hand_stem[0].weight, trainer.model.point_stem[0].weight,
                           trainer.model.point_gate[0].weight, trainer.model.stem_feat[0].conv.weight):
                assert weight.grad is not None and weight.grad.abs().sum().item() > 0
            gradient_checks.append(len(grads))

        handle = trainer.optimizer.register_step_pre_hook(check_gradients)
        torch.cuda.reset_peak_memory_stats()
        trainer.train_epoch(SmokeBatches(trainer.parser.train_dataset, 3, 3), trainer.model,
                            trainer.optimizer, 0, evaluator, color_fn=None, report=1)
        torch.cuda.synchronize()
        handle.remove()
        assert len(gradient_checks) == 3  # No AMP-skipped optimizer updates.
        train_peak = torch.cuda.max_memory_allocated() / 2**30
        train_reserved = torch.cuda.max_memory_reserved() / 2**30
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trainer.validate(SmokeBatches(trainer.parser.valid_dataset, 3, 2), trainer.model,
                         evaluator, trainer.parser.get_xentropy_class_string)
        assert all(math.isfinite(value) for value in trainer.last_gate_statistics.values())
        assert not trainer.ema_model.point_gate._forward_hooks
        trainer.writer.close()
        report = {
            "status": "PASS", "device": torch.cuda.get_device_name(),
            "train_batches": 3, "validation_batches": 2, "batch_size": 3,
            "in_vol_shape": [3, 23, 512, 512], "model_input_shape": [3, 24, 512, 512],
            "hand_shape": [3, 7, 512, 512], "point_shape": [3, 16, 512, 512],
            "mask_shape": [3, 1, 512, 512], "train_scans": 19130,
            "train_batches_per_epoch": 6376, "copy_paste_prob": 0.0, "w_paste": 0.0,
            "gradient_checks": gradient_checks, "losses_train_then_valid": step_losses,
            "validation_first_batch_gate": trainer.last_gate_statistics,
            "train_peak_allocated_GiB": train_peak, "train_peak_reserved_GiB": train_reserved,
            "validation_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "lr_preview_epochs_0_to_10": scheduler_preview(arch["train"]),
            "comparison_references": arch["experiment"],
            "resume": None, "pretrained": None,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
