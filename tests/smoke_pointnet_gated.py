"""One real-data CUDA train batch and validation batch, without a full run."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import torch
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_pointnet_hybrid_gated import Trainer
from lib.dataset.Parser_PointNet import bev_collate_fn
from lib.utils.ioueval import iouEval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")
    torch.set_num_threads(4)
    with (ROOT / "config/arch/sjnet_pointnet_gated70.yaml").open() as f:
        arch = yaml.safe_load(f)
    with (ROOT / "config/labels/semantic-kitti.yaml").open() as f:
        data = yaml.safe_load(f)
    arch["train"]["workers"] = 0
    batch_size = arch["train"]["batch_size"]

    with tempfile.TemporaryDirectory(prefix="stepc1_smoke_") as logdir:
        trainer = Trainer(arch, data, args.dataset, logdir, logger, use_mps=False)
        ignore = [i for i, weight in enumerate(trainer.loss_w) if weight < 1e-10]
        evaluator = iouEval(trainer.parser.get_n_classes(), trainer.device, ignore)
        batch = bev_collate_fn([trainer.parser.train_dataset[i] for i in range(batch_size)])
        assert batch[0].shape == (batch_size, 23, 512, 512)
        assert batch[14].count_nonzero() == 0
        gate_stats = {}

        def gate_hook(module, args, output):
            gate_stats.update(mean=output.float().mean().item(),
                              minimum=output.min().item(), maximum=output.max().item())

        handle = trainer.model.point_gate.register_forward_hook(gate_hook)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        acc, iou, loss, _ = trainer.train_epoch(
            [batch], trainer.model, trainer.optimizer, epoch=0,
            evaluator=evaluator, color_fn=None, report=1,
        )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        handle.remove()
        assert torch.isfinite(torch.tensor(loss))
        for parameter in (trainer.model.hand_stem[0].weight, trainer.model.point_stem[0].weight,
                          trainer.model.point_gate[0].weight, trainer.model.stem_feat[0].conv.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        train_peak = torch.cuda.max_memory_allocated()
        trainer.optimizer.zero_grad(set_to_none=True)
        del batch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        validation = bev_collate_fn([trainer.parser.valid_dataset[i] for i in range(batch_size)])
        acc_v, iou_v, loss_v = trainer.validate(
            [validation], trainer.model, evaluator, trainer.parser.get_xentropy_class_string,
        )
        assert torch.isfinite(torch.tensor(loss_v))
        trainer.writer.close()
        print(json.dumps({
            "batch_size": batch_size, "input_shape": [batch_size, 24, 512, 512],
            "train_loss": loss, "validation_loss": loss_v,
            "initial_gate": gate_stats, "train_seconds": train_seconds,
            "train_peak_allocated_GiB": train_peak / 2**30,
            "validation_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "status": "PASS",
        }, indent=2))


if __name__ == "__main__":
    main()
