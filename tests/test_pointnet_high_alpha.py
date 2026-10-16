import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import evaluate_pointnet_high_alpha_sensitivity as high

torch.set_num_threads(2)


class HighAlphaTests(unittest.TestCase):
    def setUp(self):
        data = yaml.safe_load((high.ROOT / "config/labels/semantic-kitti.yaml").read_text())
        self.classes = {cid: data["labels"][data["learning_map_inv"][cid]] for cid in range(20)}
        self.metadata = dict(experiment="Step D.1b", alphas=high.ALL_ALPHAS,
            validation_scans=4071, validation_batches=1357, model_state_sha256="same-frozen-state",
            checkpoint_sha256="same-checkpoint", weight_source="EMA", checkpoint_best_miou=.508252934833837,
            arch_cfg={"train": {"batch_size": 3}}, data_cfg=data, gpu="same-gpu",
            evaluation_script_sha256="same-code", validation_manifest_sha256="same-features")
        matrix = np.zeros((20, 20), dtype=np.int64)
        matrix[0, 0] = 4071 * 512 * 512 - 1900
        for cid in range(1, 20):
            matrix[cid, cid] = 100
        evaluator = high.d1.iouEval(20, torch.device("cpu"), [0])
        evaluator.conf_matrix.copy_(torch.from_numpy(matrix))
        miou, per_class = evaluator.getIoU()
        self.results = {str(a): dict(alpha=a, scans=4071, batches=1357, model_state_sha256="same-frozen-state",
            miou=miou.item(), accuracy=evaluator.getacc().item(),
            per_class_iou={self.classes[cid]: per_class[cid].item() for cid in range(1, 20)},
            confusion_matrix=matrix.tolist(), wall_seconds=1., forward_cuda_seconds=.5, peak_allocated_GiB=1.)
            for a in high.OLD_ALPHAS}
        self.old = dict(status="complete", baseline_passed=True, preflight={"status": "PASS"},
                        metadata=dict(self.metadata, experiment="Step D.1", alphas=high.OLD_ALPHAS), results=self.results)
        self.manifest = dict(entries=[[{"path": "08/frame.pt", "size": 10, "mtime_ns": 1}]], scan_ids=["08/frame.label"])

    def validate(self, old=None, current=None, manifest=None):
        return high.validate_source(old or self.old, current or self.metadata, manifest or self.manifest,
                                    self.manifest["entries"], self.manifest["scan_ids"], self.classes)

    def test_import_exact_copy_and_json_integer_key_compatibility(self):
        old = high.json_normalized(self.old)
        with patch.object(high.d1, "reproduction_passed", return_value=True):
            imported = self.validate(old=old)
        self.assertEqual(imported, self.results)
        imported["1.0"]["miou"] = 0
        self.assertEqual(old["results"]["1.0"]["miou"], 1.)
        self.assertEqual(high.NEW_ALPHAS, [1.1, 1.3, 1.4, 1.6])
        self.assertFalse(set(high.NEW_ALPHAS) & set(high.OLD_ALPHAS))

    def test_reject_changed_checkpoint_environment_code_mapping_and_features(self):
        for key in ("checkpoint_sha256", "weight_source", "gpu", "evaluation_script_sha256", "validation_manifest_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(current=dict(self.metadata, **{key: "changed"}))
        changed = copy.deepcopy(self.metadata)
        changed["data_cfg"]["learning_map"][10] = 2
        with self.assertRaisesRegex(ValueError, "data_cfg"):
            self.validate(current=changed)
        changed = copy.deepcopy(self.manifest)
        changed["entries"][0][0]["mtime_ns"] = 2
        with self.assertRaisesRegex(ValueError, "scan order"):
            self.validate(manifest=changed)

    def test_reject_incomplete_results_invalid_metrics_classes_and_failed_gate(self):
        with self.assertRaisesRegex(ValueError, "reproduction gate"):
            self.validate()
        for key, value in (("scans", 4070), ("model_state_sha256", "changed"), ("confusion_matrix", [[0]])):
            result = dict(self.results["1.0"], **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                high.validate_result(result, 1., self.metadata, self.classes)
        with self.assertRaises(AssertionError):
            high.validate_result(dict(self.results["1.0"], miou=.5), 1., self.metadata, self.classes)
        wrong = copy.deepcopy(self.results["1.0"])
        wrong["per_class_iou"]["car_changed"] = wrong["per_class_iou"].pop("car")
        with self.assertRaisesRegex(ValueError, "Class names"):
            high.validate_result(wrong, 1., self.metadata, self.classes)
        wrong = copy.deepcopy(self.old)
        wrong["results"].pop("0.4")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate(old=wrong)

    def test_export_all_nine_conditions_both_references_na_and_plots(self):
        results = {}
        for a in high.ALL_ALPHAS:
            r = copy.deepcopy(self.results["1.0"])
            r.update(alpha=a, miou=.49+a*.01)
            r["per_class_iou"] = {self.classes[cid]: .3+a*.01 for cid in range(1, 20)}
            r["per_class_iou"]["motorcyclist"] = 0.
            r["per_class_iou"]["car"] = .9  # Tied nonzero values must remain explicit.
            results[str(a)] = r
        run = dict(status="complete", baseline_passed=True,
                   metadata=dict(self.metadata, checkpoint="C.2/best_val.path"), results=results)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            high.export_results(output, run)
            for filename, zero_column in (("per_class_delta_vs_alpha1.csv", "alpha_1"),
                                          ("per_class_delta_vs_alpha12.csv", "alpha_1.2")):
                with (output / filename).open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 19)
                self.assertTrue(all(float(row[zero_column]) == 0 for row in rows))
                self.assertEqual(len(rows[0]), 10)
            with (output / "best_alpha_by_class.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(next(r for r in rows if r["Class"] == "motorcyclist")["best_alphas"], "N/A")
            self.assertEqual(next(r for r in rows if r["Class"] == "car")["status"], "tie")
            with (output / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(r["alpha"]) for r in rows], high.ALL_ALPHAS)
            self.assertEqual(sum(r["source"] == "D.1b measured" for r in rows), 4)
            self.assertAlmostEqual(float(rows[-1]["delta_vs_alpha12"]), .004)
            report = (output / "comparison.md").read_text()
            self.assertIn("N/A (all zero)", report)
            self.assertIn("upper tested boundary 1.6", report)
            self.assertIn("percentage points", report)
            for filename in ("miou_curve.png", "class_iou_curves.png", "class_delta_heatmap.png", "class_delta_heatmap_vs_alpha12.png"):
                self.assertGreater((output / filename).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
