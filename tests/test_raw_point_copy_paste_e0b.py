import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from lib.utils.laserscan_Pointnet import PolarPointEncoder
from lib.utils.raw_point_copy_paste_e0b import (
    PlacementRejected, PlacementRules, TargetScene, apply_dropout,
    compatibility_off, deterministic_labels, inspect_patch, load_patch,
    map_semantic, raw_copy_paste, read_train_scan, regenerate_hybrid,
    rigid_place, visibility_filter,
)


class RawPointCopyPasteE0bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = yaml.safe_load(Path("config/labels/semantic-kitti.yaml").read_text())
        cls.mapping = cls.data["learning_map"]

    def patch(self):
        rng = np.random.default_rng(1)
        xyz = rng.uniform([-.2, -.7, -.5], [.2, .7, .5], (120, 3)).astype(np.float32)
        xyz -= xyz.mean(0)
        return np.column_stack([xyz, np.full(120, .5), np.full(120, 11)]).astype(np.float32)

    def scene(self, collision=False):
        x, y = np.meshgrid(np.arange(7, 13.1, .2), np.arange(-3, 3.1, .2))
        points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -1.), np.full(x.size, .5)]).astype(np.float32)
        labels = np.full(len(points), 40, dtype=np.uint32)
        if collision:
            points = np.vstack([points, [10, 0, -.5, .5]]).astype(np.float32)
            labels = np.r_[labels, np.uint32(10)]
        return TargetScene(points, labels, self.mapping)

    def test_raw_and_learning_ids_use_yaml_including_moving_bus(self):
        actual = map_semantic(np.array([11, 18, 20, 253, 254, 255, 257, 259], np.uint32), self.mapping)
        np.testing.assert_array_equal(actual, [2, 4, 5, 7, 6, 8, 5, 5])
        with self.assertRaises(ValueError):
            map_semantic(np.array([999], np.uint32), self.mapping)

    def test_quality_small_mixed_nonfinite_filename_and_center(self):
        patch = self.patch()
        self.assertFalse(inspect_patch(patch, "class_11_000000.npy", self.mapping)["reasons"])
        self.assertIn("insufficient_class_specific_points", inspect_patch(patch[:10], "class_11_000000.npy", self.mapping)["reasons"])
        mixed = patch.copy(); mixed[0, 4] = 30
        self.assertIn("mixed_raw_semantic_ids", inspect_patch(mixed, "class_11_000000.npy", self.mapping)["reasons"])
        broken = patch.copy(); broken[0, 0] = np.nan
        self.assertIn("nonfinite_values", inspect_patch(broken, "class_11_000000.npy", self.mapping)["reasons"])
        self.assertIn("filename_content_mismatch", inspect_patch(patch, "class_15_000000.npy", self.mapping)["reasons"])
        shifted = patch.copy(); shifted[:, 0] += 1
        self.assertIn("relative_center_not_zero", inspect_patch(shifted, "class_11_000000.npy", self.mapping)["reasons"])

    def test_rigid_transform_preserves_distances_and_z_shape(self):
        xyz = self.patch()[:, :3]
        moved, rotation = rigid_place(xyz, 1.4, np.array([12, 5, -1]))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(moved[1:] - moved[0], axis=1),
                                   np.linalg.norm(xyz[1:] - xyz[0], axis=1), atol=1.5e-6)

    def test_grounded_raw_success_preserves_all_point_label_correspondence(self):
        scene, patch = self.scene(), self.patch()
        result = raw_copy_paste(scene, patch, self.mapping, [10, 0], 0)
        self.assertGreaterEqual(result["metadata"]["visible_fraction"], .8)
        self.assertAlmostEqual(result["metadata"]["minimum_ground_clearance"], .03, places=5)
        np.testing.assert_array_equal(result["points"][~result["paste_points"]], scene.points[result["target_keep"]])
        np.testing.assert_array_equal(result["labels"][result["paste_points"]], patch[result["donor_visible"], 4])
        np.testing.assert_array_equal(result["points"][result["paste_points"], 3], patch[result["donor_visible"], 3])
        self.assertEqual(len(result["points"]), len(result["labels"]))

    def test_collision_rejects_without_mutating_target_or_donor(self):
        scene, patch = self.scene(True), self.patch()
        before, donor_before = scene.points.copy(), patch.copy()
        with self.assertRaisesRegex(PlacementRejected, "collision"):
            raw_copy_paste(scene, patch, self.mapping, [10, 0], 0)
        np.testing.assert_array_equal(scene.points, before)
        np.testing.assert_array_equal(patch, donor_before)

    def test_empty_near_far_and_invalid_positions_reject(self):
        patch = self.patch()
        with self.assertRaisesRegex(PlacementRejected, "ground"):
            raw_copy_paste(TargetScene(np.empty((0, 4), np.float32), np.empty(0, np.uint32), self.mapping),
                           patch, self.mapping, [10, 0], 0)
        with self.assertRaisesRegex(PlacementRejected, "close"):
            raw_copy_paste(self.scene(), patch, self.mapping, [1, 0], 0)
        with self.assertRaises(PlacementRejected):
            raw_copy_paste(self.scene(), patch, self.mapping, [52, 0], 0)
        with self.assertRaisesRegex(PlacementRejected, "invalid_transform"):
            raw_copy_paste(self.scene(), patch, self.mapping, [np.nan, 0], 0)
        with self.assertRaisesRegex(PlacementRejected, "invalid_patch"):
            raw_copy_paste(self.scene(), np.ones((5, 3)), self.mapping, [10, 0], 0)

    def test_explicit_depth_occlusion_both_directions_and_self_occlusion(self):
        target = np.array([[5, 0, 0, .5], [20, .2, 0, .5]], np.float32)
        donor = np.array([[10, 0, 0, .5], [10, .1, 0, .5]], np.float32)
        keep, visible = visibility_filter(target, donor, PlacementRules())
        self.assertFalse(visible[0]); self.assertTrue(keep[0])
        self.assertTrue(visible[1]); self.assertFalse(keep[1])
        _, visible = visibility_filter(np.empty((0, 4)), np.array([[10, 0, 0], [11, 0, 0]]), PlacementRules())
        np.testing.assert_array_equal(visible, [True, False])

    def test_periodic_seam_does_not_clip_object(self):
        points, _ = rigid_place(self.patch()[:, :3], np.pi, [-10, 0, -1])
        angle = np.arctan2(points[:, 1], points[:, 0])
        self.assertTrue((angle > 3).any() and (angle < -3).any())
        keep, visible = visibility_filter(np.empty((0, 4)), points, PlacementRules(depth_tolerance=1.))
        self.assertEqual(keep.size, 0)
        self.assertTrue(visible.all())

    def test_seam_and_radial_boundaries_share_regenerated_cell_ids(self):
        encoder=PolarPointEncoder().eval().requires_grad_(False)
        points=np.array([[-10,1e-5,0,.5],[-10,-1e-5,0,.5],
                         [51.19,0,-1,.3],[51.21,0,-1,.3],[0,0,0,.3]],np.float32)
        result=regenerate_hybrid(points,np.full(5,40,np.uint32),encoder,self.mapping)
        np.testing.assert_array_equal(result["valid_points"],[True,True,True,False,False])
        self.assertEqual(set((result["cell_ids"]%512).tolist()),{0,511,256})
        self.assertIn(511,(result["cell_ids"]//512).tolist())
        self.assertEqual(int(result["mask"].sum()),3)

    def test_radius_crossing_is_rejected_not_truncated(self):
        original=self.scene()
        points=original.points.copy();points[:,0]+=41
        scene=TargetScene(points,original.labels,self.mapping)
        with self.assertRaisesRegex(PlacementRejected,"outside_polar_radius"):
            raw_copy_paste(scene,self.patch(),self.mapping,[51.1,0],0)

    def test_ground_sampler_and_raw_result_reproduce_with_same_seed(self):
        scene,patch=self.scene(),self.patch()
        a=scene.sample_ground(2,np.random.default_rng(8))
        b=scene.sample_ground(2,np.random.default_rng(8))
        np.testing.assert_array_equal(a,b)
        first=raw_copy_paste(scene,patch,self.mapping,[10,0],0)
        second=raw_copy_paste(scene,patch,self.mapping,[10,0],0)
        for key in("points","labels","paste_points","target_keep","donor_visible"):
            np.testing.assert_array_equal(first[key],second[key])

    def test_tie_break_is_deterministic_and_preserves_priorities(self):
        cells, inv = np.array([100]), np.array([0, 0, 0])
        label = deterministic_labels(cells, inv, np.array([40, 48, 30]), self.mapping)
        self.assertEqual(int(label.reshape(-1)[100]), 6)
        label = deterministic_labels(cells, inv, np.array([40, 48, 40]), self.mapping)
        self.assertEqual(int(label.reshape(-1)[100]), 11)
        other = deterministic_labels(cells, inv, np.array([40, 40, 48]), self.mapping)
        torch.testing.assert_close(label, other, atol=0, rtol=0)

    def test_all24_channels_share_point_mapping_and_missing_cells_are_not_stale(self):
        encoder = PolarPointEncoder().eval().requires_grad_(False)
        points = np.array([[10.01, .01, 0, .3], [10.02, .02, 1., .5], [-10, -.001, 0, .4]], np.float32)
        result = regenerate_hybrid(points, np.array([40, 30, 48], np.uint32), encoder, self.mapping,
                                   np.array([False, True, False]))
        self.assertEqual(tuple(result["hybrid24"].shape), (24, 512, 512))
        self.assertEqual(int(result["mask"].sum()), len(result["cell_ids"]))
        torch.testing.assert_close(result["hybrid24"][23:24], result["mask"], atol=0, rtol=0)
        self.assertFalse(result["point16"][:, result["mask"][0] == 0].any())
        self.assertTrue((result["hybrid24"][:2, result["mask"][0] == 0] == -.2).all())
        self.assertFalse((result["paste_mask"] > result["mask"]).any())
        self.assertFalse(result["point16"].requires_grad)
        encoder.train()
        with self.assertRaisesRegex(ValueError, "frozen_eval"):
            regenerate_hybrid(points, np.array([40, 30, 48], np.uint32), encoder, self.mapping)

    def test_dropout_clears_main_aux_ignore_labels_only_in_new_policy(self):
        tree = ast.parse(Path("lib/trainer/trainer_PointNet.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FocalLoss")
        namespace = dict(nn=torch.nn, F=torch.nn.functional, torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "original_focal_loss", "exec"), namespace)
        hybrid, labels, paste = torch.ones(24, 2, 2), torch.full((2, 2), 6), torch.ones(1, 2, 2)
        keep = torch.tensor([[True, False], [True, True]])
        h, target, p = apply_dropout(hybrid, labels, paste, keep)
        self.assertTrue((h[:, 0, 1] == 0).all()); self.assertEqual(int(target[0, 1]), 0)
        self.assertEqual(float(p[0, 0, 1]), 0)
        for _ in range(4):
            logits = torch.zeros(1, 20, 2, 2, requires_grad=True)
            namespace["FocalLoss"](gamma=1,ignore_index=0)(logits, target[None]).backward()
            self.assertEqual(float(logits.grad[0, :, 0, 1].abs().sum()), 0)
        _, denoise, _ = apply_dropout(hybrid, labels, paste, keep, policy="legacy_denoising")
        self.assertEqual(int(denoise[0, 1]), 6)
        self.assertTrue((labels == 6).all())

    def test_lovasz_and_boundary_have_zero_dropped_cell_gradient(self):
        from lib.losses.Lovasz_Softmax import Lovasz_softmax
        logits=torch.zeros(1,20,2,2,requires_grad=True)
        labels=torch.tensor([[[6,0],[6,6]]])
        Lovasz_softmax(ignore=0)(logits.softmax(1),labels).backward()
        self.assertEqual(float(logits.grad[0,:,0,1].abs().sum()),0.)
        boundary=torch.zeros(1,1,2,2,requires_grad=True)
        mask=torch.tensor([[[[1.,0.],[1.,1.]]]])
        loss_map=torch.nn.functional.binary_cross_entropy_with_logits(boundary,torch.zeros_like(boundary),reduction="none")
        ((loss_map*mask).sum()/mask.sum()).backward()
        self.assertEqual(float(boundary.grad[0,0,0,1]),0.)

    def test_independent_reference_checks_pooling_and_all_features(self):
        from verify_raw_copy_paste_e0b import independent_reference
        encoder=PolarPointEncoder().eval().requires_grad_(False)
        points=np.array([[10.01,.01,0.,.3],[10.02,.02,1.,.5],[-10.01,.001,.5,.4]],np.float32)
        labels=np.array([40,30,48],np.uint32)
        actual=regenerate_hybrid(points,labels,encoder,self.mapping)
        reference=independent_reference(points,labels,self.mapping,encoder,"cpu")
        torch.testing.assert_close(actual["point16"],reference["point16"],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(actual["label"],reference["label"],atol=0,rtol=0)
        torch.testing.assert_close(actual["mask"],reference["mask"],atol=0,rtol=0)

    def test_unknown_provenance_quality_and_manifest_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "class_11_000000.npy"
            np.save(path, self.patch())
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_patch(path, {"usable_for_audit": True}, self.mapping)
            result = load_patch(path, {"usable_for_audit": True}, self.mapping, audit_unverified_provenance=True)
            self.assertEqual(result.shape[1], 5)
            with self.assertRaisesRegex(ValueError, "changed"):
                load_patch(path, {"usable_for_audit": True, "sha256": "incorrect"}, self.mapping, audit_unverified_provenance=True)

    def test_validation_target_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "not_train"):
            read_train_scan("SemanticKitti/sequences/08/velodyne/000000.bin", self.mapping, self.data["split"]["train"])

    def test_compatibility_off_has_no_extra_rng_or_data_changes(self):
        class Dummy:
            def __getitem__(self, index):
                return torch.rand(3), index
        torch.manual_seed(9); expected = Dummy()[4]
        torch.manual_seed(9); actual = compatibility_off(Dummy(), 4)
        torch.testing.assert_close(expected[0], actual[0], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
