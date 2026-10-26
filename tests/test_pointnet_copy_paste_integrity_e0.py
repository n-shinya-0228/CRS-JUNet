"""CPU-only E.0 diagnostics; synthetic cases deliberately expose legacy limits."""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lib.dataset.SemanticKitti_PointNet import SemanticKitti
from lib.utils.laserscan_Pointnet import build_polar_point_features
from lib.utils.pointnet_copy_paste_e0 import (
    CandidateTrace, ambiguous_priority_cells, cell_ids_numpy, differences, normalize_handcrafted,
    project_handcrafted, transplant_cell_points,
)


class CopyPasteIntegrityE0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def dataset(self):
        dataset = object.__new__(SemanticKitti)
        dataset.bev_files = ["synthetic_train_donor.pt"]
        return dataset

    def inputs(self, label=9):
        return (torch.zeros(7, 512, 512), torch.zeros(16, 512, 512),
                torch.ones(1, 512, 512), torch.full((512, 512), label, dtype=torch.long))

    def donor(self, cells=16, cls=18):
        label = torch.zeros(512, 512, dtype=torch.long)
        label[100:104, 100:104] = cls
        label.reshape(-1)[torch.where(label.reshape(-1) > 0)[0][cells:]] = 0
        return {"proj_tensor": torch.ones(7, 512, 512),
                "mask_t": (label > 0).unsqueeze(0), "labels_t": label}

    def run_synthetic(self, inputs, donor):
        trace = CandidateTrace()
        with patch("lib.dataset.SemanticKitti_PointNet.gzip.open"), \
             patch("lib.dataset.SemanticKitti_PointNet.torch.load", return_value=donor):
            output, _ = trace.call(self.dataset(), [v.clone() for v in inputs])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, output)))
        self.assertEqual(int(output[-1].sum()), 0)
        self.assertEqual(len(trace.rows), 12)
        return trace

    def test_no_donor_is_unchanged(self):
        dataset = self.dataset()
        dataset.bev_files = []
        inputs = self.inputs()
        result = dataset.apply_bev_copy_paste(*[v.clone() for v in inputs])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, result)))
        self.assertFalse(result[-1].any())

    def test_no_eligible_object_skips(self):
        trace = self.run_synthetic(self.inputs(), self.donor(cls=1))
        self.assertTrue(all(r["reason"] == "ys.numel() == 0" for r in trace.rows))

    def test_tiny_object_skips(self):
        trace = self.run_synthetic(self.inputs(), self.donor(cells=3))
        self.assertTrue(all(r["reason"] == "patch_obj_mask.sum() < 10" for r in trace.rows))

    def test_empty_target_context_skips(self):
        inputs = list(self.inputs(label=0))
        inputs[2].zero_()
        trace = self.run_synthetic(inputs, self.donor())
        self.assertTrue(all("context_mask.float().mean() < 0.1" == r["reason"] for r in trace.rows))

    def test_foreground_target_context_skips(self):
        trace = self.run_synthetic(self.inputs(label=1), self.donor())
        self.assertTrue(all("context_mask.float().mean() < 0.1" == r["reason"] for r in trace.rows))

    def test_bicyclist_without_bicycle_skips(self):
        trace = self.run_synthetic(self.inputs(label=11), self.donor(cls=7))
        self.assertTrue(all("vehicle_mask.sum() < 5" == r["reason"] for r in trace.rows))

    def test_existing_foreground_overlap_rejected_without_mutation(self):
        inputs = list(self.inputs(label=9))
        inputs[3][100:102, 32:36] = 1
        with patch("lib.dataset.SemanticKitti_PointNet.torch.randint", return_value=torch.tensor([0])), \
             patch("lib.dataset.SemanticKitti_PointNet.np.random.randint", return_value=0):
            trace = self.run_synthetic(inputs, self.donor())
        self.assertTrue(all(r["reason"] == "overlap_ratio > occ_th" for r in trace.rows))

    def test_visible_paste_updates_all_features_and_empty_cells(self):
        dataset = self.dataset()
        inputs = list(self.inputs())
        inputs[2][0, 100, 32:36] = 0
        donor = self.donor()
        dense_point = torch.full((16, 512, 512), 2.)
        with patch("lib.dataset.SemanticKitti_PointNet.gzip.open"), \
             patch("lib.dataset.SemanticKitti_PointNet.torch.load", return_value=donor), \
             patch.object(dataset, "load_pointnet_feature", return_value=(dense_point, donor["mask_t"][0])), \
             patch("lib.dataset.SemanticKitti_PointNet.torch.randint", return_value=torch.tensor([0])), \
             patch("lib.dataset.SemanticKitti_PointNet.np.random.randint", return_value=0), \
             patch("lib.dataset.SemanticKitti_PointNet.torch.rand", return_value=torch.tensor([0.])):
            trace = CandidateTrace()
            result, _ = trace.call(dataset, [v.clone() for v in inputs])
        m = result[4][0].bool()
        self.assertEqual(int(m.sum()), 16)
        self.assertTrue((result[0][:, m] == 1).all())
        self.assertTrue((result[1][:, m] == 2).all())
        self.assertTrue((result[2][0, m] == 1).all())
        self.assertTrue((result[3][m] == 18).all())
        self.assertTrue(all(torch.equal(a[..., ~m], b[..., ~m]) for a, b in zip(inputs, result)))

    def test_donor_read_error_is_safe_skip(self):
        inputs = self.inputs()
        trace = CandidateTrace()
        with patch("lib.dataset.SemanticKitti_PointNet.gzip.open", side_effect=OSError("synthetic read error")):
            result, _ = trace.call(self.dataset(), [v.clone() for v in inputs])
        self.assertEqual(len(trace.rows), 12)
        self.assertTrue(all(r["reason"] == "donor_read_error" for r in trace.rows))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, result)))

    def test_pointnet_inputs_change_after_radius_translation(self):
        points = torch.tensor([[10.01, .01, 1., .5], [10.02, .02, 2., .7]])
        moved = points.clone()
        moved[:, :2] *= 1.1
        before, _, _ = build_polar_point_features(points)
        after, _, _ = build_polar_point_features(moved)
        self.assertFalse(torch.equal(before[:, :2], after[:, :2]))

    def test_periodic_edge_projection_in_bounds(self):
        p = np.array([[-10, 1e-5, 0, .5], [-10, -1e-5, 0, .5]], np.float32)
        ids, valid = cell_ids_numpy(p)
        self.assertTrue(valid.all())
        self.assertEqual(set(ids % 512), {0, 511})
        self.assertTrue(((ids >= 0) & (ids < 512 * 512)).all())

    def test_legacy_crop_does_not_wrap_seam(self):
        # For an object on both angular edges, a 64-wide legacy crop loses one side.
        width, cx = 512, 0
        x1 = max(0, cx - 32)
        x2 = min(width, x1 + 64)
        x1 = max(0, x2 - 64)
        label = torch.zeros(512, dtype=torch.bool)
        label[[0, 511]] = True
        self.assertEqual(int(label[x1:x2].sum()), 1)
        self.assertEqual(int(label.sum()), 2)

    def test_remission_and_priority_label_not_majority(self):
        p = np.array([[10.01, .01, 0., .2], [10.01, .01, 2., .9],
                      [10.01, .01, 1., .3]], dtype=np.float32)
        packed = np.array([40, 30, 40], np.uint32)
        hand, mask, label = project_handcrafted(p, packed, {0: 0, 40: 9, 30: 6})
        cell = mask[0].nonzero()[0]
        y, x = cell.tolist()
        self.assertAlmostEqual(float(hand[2, y, x]), .9, places=6)
        self.assertEqual(int(label[y, x]), 6)
        self.assertAlmostEqual(float(hand[3, y, x]), .03, places=6)

    def test_normalization_depends_on_postpaste_scan_maximum(self):
        p = torch.zeros(7, 2, 2)
        p[2, 0, 0] = .5
        q = p.clone()
        q[2, 1, 1] = 1.
        self.assertEqual(float(normalize_handcrafted(p)[2, 0, 0]), 1.)
        self.assertEqual(float(normalize_handcrafted(q)[2, 0, 0]), .5)

    def test_raw_diagnostic_replacement_preserves_packed_labels(self):
        target = np.array([[10.03, .01, 0., .1]], np.float32)
        source = np.array([[10.04, .02, 1., .8]], np.float32)
        ids, _ = cell_ids_numpy(source)
        y, x = divmod(int(ids[0]), 512)
        paste = {"y1": y, "y2": y+1, "x1": x, "x2": x+1,
                 "ty": y+10, "tx": (x+100) % 512, "h": 1, "w": 1,
                 "m": torch.ones(1, 1, 1, dtype=torch.bool)}
        points, labels, info = transplant_cell_points(target, np.array([40], np.uint32),
            source, np.array([(42 << 16) | 30], np.uint32), paste)
        self.assertEqual(int(labels[-1]), (42 << 16) | 30)
        np.testing.assert_array_equal(points[-1, 2:], source[-1, 2:])
        self.assertEqual(info["transformed_cell_mismatch_points"], 0)
        self.assertEqual(info["invalid_transformed_points"], 0)

    def test_label_map_difference_counts_cells_not_columns(self):
        a = torch.zeros(3, 3, dtype=torch.long)
        b = a.clone()
        b[:, 0] = 1
        self.assertEqual(differences(a, b)["different_cells"], 3)

    def test_priority_tie_ambiguity_is_measured(self):
        p = np.array([[10.01, .01, 0., .2], [10.01, .01, 2., .9]], np.float32)
        ambiguous = ambiguous_priority_cells(p, np.array([40, 48], np.uint32), {0: 0, 40: 9, 48: 11})
        self.assertEqual(int(ambiguous.sum()), 1)

    def test_actual_focal_supervises_zero_observation_nonignore_label(self):
        # Isolate the unchanged loss class without importing legacy entrypoints.
        source = Path(__file__).resolve().parents[1] / "lib/trainer/trainer_PointNet.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "FocalLoss")
        namespace = {"nn": torch.nn, "F": torch.nn.functional, "torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        logits = torch.zeros(1, 20, 1, 2, requires_grad=True)
        label = torch.tensor([[[6, 0]]])
        criterion = namespace["FocalLoss"](ignore_index=0)
        criterion(logits, label).backward()
        self.assertGreater(float(logits.grad[0, :, 0, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, :, 0, 1].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
