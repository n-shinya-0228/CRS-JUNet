"""D.1 inference invariants, EMA selection, metric parity and partial restart."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

import evaluate_pointnet_fixed_alpha_sensitivity as evaluation
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_FixedAlpha import SJNet3_PointNet_Gated_Local3_FixedAlpha
from lib.utils.ioueval import iouEval

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class FixedAlphaTests(unittest.TestCase):
    def test_strict_compatible_state_all_outputs_masks_and_reordered_alphas(self):
        options = dict(nclasses=4, base_ch=8, aspp_out=16, drop=0)
        torch.manual_seed(0)
        original = SJNet3_PointNet_Gated_Local3(**options).eval().requires_grad_(False)
        model = SJNet3_PointNet_Gated_Local3_FixedAlpha(**options).eval().requires_grad_(False)
        model.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(original.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in original.parameters()))
        self.assertIs(model.forward.__func__, original.forward.__func__)
        digest = evaluation.state_digest(model.state_dict())
        x = torch.randn(2, 24, 32, 320)
        for mask in (0., 1., None):
            x[:, 23:24] = torch.rand(2, 1, 32, 320).round() if mask is None else mask
            before = x.clone()
            model.set_alpha(1)
            with torch.no_grad():
                reference, actual = original(x), model(x)
            for name in reference:
                torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)
            for alpha in (1.2, .4, 1.):
                model.set_alpha(alpha)
                with torch.no_grad():
                    output = model(x)
                self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
                self.assertEqual(evaluation.state_digest(model.state_dict()), digest)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
                if alpha == 1:
                    for name in output:
                        torch.testing.assert_close(output[name], reference[name], atol=0, rtol=0)
            torch.testing.assert_close(x, before, atol=0, rtol=0)
        incomplete = dict(original.state_dict())
        incomplete.pop("point_stem.0.weight")
        with self.assertRaises(RuntimeError):
            model.load_state_dict(incomplete, strict=True)
        with self.assertRaises(ValueError):
            model.set_alpha(float("nan"))

    def test_local_equation_changes_only_plain_float_coefficient(self):
        model = SJNet3_PointNet_Gated_Local3_FixedAlpha(nclasses=4, base_ch=8, aspp_out=16).eval()
        base = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            residual = model.point_context.local(base)
            model.set_alpha(1)
            first = model.point_context(base)
            torch.testing.assert_close(first, base + residual, atol=0, rtol=0)
            model.set_alpha(.4)
            second = model.point_context(base)
            torch.testing.assert_close(second, base + .4 * residual, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(any("alpha" in name for name, _ in model.named_parameters()))
        self.assertFalse(any("alpha" in name for name, _ in model.named_buffers()))

    def test_ema_selection_and_wrong_checkpoint_rejection(self):
        arch = {"train": {"use_ema": True}}
        data = {"split": {"valid": [8]}}
        checkpoint = dict(experiment="Step C.2", model_name="SJNet3_PointNet_Gated_Local3", epoch=18,
                          best_val_epoch=18, arch_cfg=arch, data_cfg=data, best_val_iou=.508252934833837,
                          model={"weight": torch.tensor([2.])}, model_ema={"weight": torch.tensor([2.])},
                          model_train={"weight": torch.tensor([1.])})
        state, source = evaluation.select_evaluation_state(checkpoint, arch, data)
        self.assertIn("EMA", source)
        self.assertEqual(state["weight"].item(), 2.)
        for changes in (dict(experiment="Step C.3"), dict(epoch=30), dict(model={"weight": torch.tensor([1.])})):
            with self.assertRaises(ValueError):
                evaluation.select_evaluation_state(dict(checkpoint, **changes), arch, data)
        absent = dict(checkpoint)
        absent.pop("model_ema")
        with self.assertRaises(ValueError):
            evaluation.select_evaluation_state(absent, arch, data)
        with self.assertRaises(ValueError):
            evaluation.select_evaluation_state(checkpoint, arch, {"split": {"valid": [0]}})

    def test_original_height_correction_and_accumulated_confusion_metrics(self):
        logits = torch.zeros(1, 20, 1, 4)
        logits[0, 7, 0, :2] = 1
        logits[0, 8, 0, 2:] = 1
        features = torch.zeros(1, 23, 1, 4)
        features[0, 0, 0] = torch.tensor([-.4, -.3, -.4, -.3])
        self.assertEqual(evaluation.original_predictions(logits, features).tolist(), [[[2, 7, 3, 8]]])
        evaluator = iouEval(3, torch.device("cpu"), [0])
        predictions, targets = torch.tensor([1, 1, 2, 0, 1, 2]), torch.tensor([1, 2, 2, 1, 0, 1])
        evaluator.addBatch(predictions[:2], targets[:2])
        evaluator.addBatch(predictions[2:], targets[2:])
        matrix = torch.zeros(3, 3).long()
        for p, y in zip(predictions, targets):
            matrix[p, y] += 1
        torch.testing.assert_close(evaluator.conf_matrix, matrix, atol=0, rtol=0)
        miou, per_class = evaluator.getIoU()
        self.assertAlmostEqual(miou.item(), 1/3)
        self.assertAlmostEqual(per_class[1].item(), 1/3)
        self.assertAlmostEqual(per_class[2].item(), 1/3)
        self.assertAlmostEqual(evaluator.getacc().item(), .5)

    def test_reproduction_gate_and_zero_or_tied_best_alphas(self):
        self.assertTrue(evaluation.reproduction_passed({"miou": .508253}, .508252934833837))
        self.assertFalse(evaluation.reproduction_passed({"miou": .507}, .508252934833837))
        winners, _, status = evaluation.best_alphas({.4: 0., .6: 0., .8: 0., 1.: 0., 1.2: 0.})
        self.assertEqual(len(winners), 5)
        self.assertEqual(status, "indistinguishable_all_zero")
        self.assertEqual(evaluation.best_alphas({.4: .3, 1.: .3, 1.2: .2})[2], "tie")

    def test_partial_restart_has_no_duplicate_scans_or_confusion_counts(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
                self.calls, self.fail = 0, True

            def set_alpha(self, alpha):
                self.fixed_alpha = alpha

            def forward(self, x):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RuntimeError("simulated interruption")
                logits = torch.zeros(x.shape[0], 20, 2, 2)
                logits[:, 1] = self.weight
                return {"logits": logits}

        class Loader:
            batch_size = 1

            def __len__(self):
                return 2

            def __iter__(self):
                for index in range(2):
                    batch = [None] * 16
                    batch[:3] = [torch.zeros(1, 23, 2, 2), torch.ones(1, 1, 2, 2), torch.ones(1, 2, 2).long()]
                    batch[4:6] = [["08"], [f"{index:06d}.label"]]
                    batch[14] = torch.zeros(1, 1, 2, 2)
                    yield batch

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 5.

        model, loader = Model().eval(), Loader()
        classes = {index: f"class_{index}" for index in range(20)}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(torch.cuda, "Event", Event), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=2**30):
            options = dict(model=model, loader=loader, alpha=1., device=torch.device("cpu"), ignore=[0],
                           class_names=classes, scan_ids=["08/000000.label", "08/000001.label"],
                           output=Path(tmp), metadata_id="same-data-and-model", report_every=1)
            with self.assertRaises(RuntimeError):
                evaluation.evaluate_alpha(**options)
            saved = json.loads((Path(tmp) / "progress_alpha_1.json").read_text())
            self.assertEqual(saved["next_batch"], 1)
            self.assertEqual(saved["seen_scans"], 1)
            self.assertEqual(sum(map(sum, saved["confusion_matrix"])), 4)
            model.fail = False
            result = evaluation.evaluate_alpha(**options)
            self.assertEqual(result["scans"], 2)
            self.assertEqual(sum(map(sum, result["confusion_matrix"])), 8)
            self.assertAlmostEqual(result["miou"], 1/19)
            self.assertAlmostEqual(result["accuracy"], 1.)
            self.assertAlmostEqual(result["forward_cuda_seconds"], .01)

    def test_exports_all_nineteen_classes_deltas_ties_and_three_plots(self):
        data = yaml.safe_load((ROOT / "config/labels/semantic-kitti.yaml").read_text())
        names = [data["labels"][data["learning_map_inv"][cid]] for cid in range(1, 20)]
        alphas = [.4, .6, .8, 1., 1.2]
        run = dict(status="complete", baseline_passed=True,
                   metadata=dict(alphas=alphas, checkpoint="C.2/best_val.path", weight_source="EMA"), results={})
        for alpha in alphas:
            values = {name: .4 + alpha * .01 for name in names}
            values["motorcyclist"] = 0.
            run["results"][str(alpha)] = dict(miou=sum(values.values()) / 19, accuracy=.89,
                per_class_iou=values, wall_seconds=1., forward_cuda_seconds=.5, peak_allocated_GiB=1.)
        with tempfile.TemporaryDirectory() as tmp:
            evaluation.export_results(Path(tmp), run)
            with (Path(tmp) / "best_alpha_by_class.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 19)
            self.assertEqual(next(r for r in rows if r["Class"] == "motorcyclist")["status"], "indistinguishable_all_zero")
            self.assertIn("indistinguishable (all zero)", (Path(tmp) / "comparison.md").read_text())
            with (Path(tmp) / "per_class_delta.csv").open() as stream:
                deltas = list(csv.DictReader(stream))
            self.assertTrue(all(float(row["alpha_1"]) == 0 for row in deltas))
            for name in ("miou_vs_alpha.png", "selected_class_iou_vs_alpha.png", "per_class_delta_heatmap.png"):
                self.assertGreater((Path(tmp) / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
