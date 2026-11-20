import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from lib.utils.raw_point_copy_paste_basic_e1c import basic_copy_paste, object_counts, place_basic
from lib.utils.raw_point_copy_paste_e0b import PlacementRejected
from lib.dataset.SemanticKitti_PointNet_BasicRawCopyPaste_E1c import SemanticKitti


class FakeDatabase:
    def __init__(self):
        self.index = 0

    def sample(self, rng):
        self.index += 1
        row = dict(donor_id=str(self.index), learning_id=6, sha256="test", source_sequence="00",
                   source_frame="000000", source_instance_id=123)
        return row, np.array([[-.1, -.1, -.5, .2, 30], [.1, .1, .5, .7, 30]], np.float32)


class BasicTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {10: 1, 30: 6}
        self.points = np.array([[20., 0, -1.7, .3], [21., 1, -1.7, .6]], np.float32)
        self.labels = np.full(2, (1 << 16) | 10, np.uint32)
        self.config = dict(mode="basic", probability=1., max_objects_per_scan=3, max_candidate_trials=2)

    def run_cp(self, function=None, limit=3):
        config = dict(self.config, max_objects_per_scan=limit)
        if function is None:
            return basic_copy_paste(self.points, self.labels, FakeDatabase(), self.mapping,
                                    np.random.default_rng(0), config)
        with patch("lib.utils.raw_point_copy_paste_basic_e1c.place_basic", side_effect=function):
            return self.run_cp(limit=limit)

    def test_real_placement_append_only_three_unique_ids(self):
        result = self.run_cp()
        self.assertEqual(result["successful_objects"], 3)
        np.testing.assert_array_equal(result["points"][:2], self.points)
        np.testing.assert_array_equal(result["packed"][:2], self.labels)
        np.testing.assert_array_equal(result["points"][2:, 3], np.tile([.2, .7], 3).astype(np.float32))
        self.assertTrue(np.all(result["packed"][2:] & 65535 == 30))
        self.assertEqual(set(result["packed"][2:] >> 16), {2, 3, 4})
        self.assertEqual(len(result["points"]), len(result["packed"]))
        self.assertEqual(int(result["paste_points"].sum()), 6)

    def test_all_zero_to_three_success_counts_and_bounded_retries(self):
        rows = []
        for desired in range(4):
            calls = []
            def placement(points, donor, mapping, center, yaw):
                calls.append(1)
                if len(calls) <= desired:
                    return place_basic(points, donor, mapping, np.array([20., 0]), 0.)
                raise PlacementRejected("deliberate")
            result = self.run_cp(placement)
            self.assertEqual(result["successful_objects"], desired)
            self.assertLessEqual(len(calls), 6)
            rows.append(result)
        self.assertEqual(object_counts(rows), {"0": 1, "1": 1, "2": 1, "3": 1})

    def test_no_advanced_functions(self):
        with patch("lib.utils.raw_point_copy_paste_e0b.TargetScene", side_effect=AssertionError), \
             patch("lib.utils.raw_point_copy_paste_e0b.raw_copy_paste", side_effect=AssertionError), \
             patch("lib.utils.raw_point_copy_paste_e0b.visibility_filter", side_effect=AssertionError), \
             patch("numpy.linalg.lstsq", side_effect=AssertionError):
            self.assertEqual(self.run_cp()["successful_objects"], 3)

    def test_bad_coordinates_labels_and_range_rejected(self):
        _, donor = FakeDatabase().sample(None)
        for center in (np.array([60., 0]), np.array([np.nan, 0])):
            with self.assertRaises(PlacementRejected):
                place_basic(self.points, donor, self.mapping, center, 0.)
        for raw in (30.5, 65535):
            broken = donor.copy(); broken[:, 4] = raw
            with self.assertRaises(PlacementRejected):
                place_basic(self.points, broken, self.mapping, [20., 0], 0.)
        with self.assertRaises(ValueError):
            basic_copy_paste(self.points, self.labels[:1], FakeDatabase(), self.mapping,
                             np.random.default_rng(0), self.config)
        for limit in (0, 4):
            with self.assertRaises(ValueError):
                self.run_cp(limit=limit)

    def test_fov_rejection_keeps_entire_object(self):
        _, donor = FakeDatabase().sample(None)
        donor[:, 2] = [-10., 10.]
        with self.assertRaisesRegex(PlacementRejected, "sensor_fov"):
            place_basic(self.points, donor, self.mapping, [20., 0], 0.)

    def test_off_and_advanced_dispatch_are_legacy(self):
        dataset = SemanticKitti.__new__(SemanticKitti)
        for enabled, mode in ((False, "basic"), (True, "advanced")):
            dataset.enabled, dataset.mode = enabled, mode
            with patch("lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1.SemanticKitti.__getitem__", return_value="legacy") as call:
                self.assertEqual(dataset[7], "legacy")
                call.assert_called_once_with(7)

    def test_probability_zero_and_one_and_reproducibility(self):
        dataset = SemanticKitti.__new__(SemanticKitti)
        dataset.enabled, dataset.mode, dataset.epoch = True, "basic", 0
        dataset.files = [Path("SemanticKitti/sequences/00/velodyne/000000.bin")]
        dataset.data, dataset.cp = {"learning_map": self.mapping}, dict(self.config, probability=0.)
        dataset.database = FakeDatabase()
        with patch("lib.dataset.SemanticKitti_PointNet_BasicRawCopyPaste_E1c.read_train_scan", return_value=(self.points, self.labels)):
            off = dataset[0]
            self.assertEqual(off["stats"]["successful_objects"], 0)
            self.assertEqual(dataset.database.index, 0)
            dataset.cp["probability"] = 1.
            on = dataset[0]
            dataset.database = FakeDatabase()
            replay = dataset[0]
            self.assertEqual(on["stats"]["successful_objects"], 3)
            np.testing.assert_array_equal(on["points"], replay["points"])
            self.assertEqual(on["augmentation_seed"], off["augmentation_seed"])

    def test_configuration_diff_is_cp_only(self):
        on = yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1b_cp_on.yaml").read_text())
        basic = yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1c_basic3.yaml").read_text())
        self.assertEqual(basic["copy_paste"].pop("mode"), "basic")
        self.assertEqual(basic["copy_paste"]["max_objects_per_scan"], 3)
        basic["copy_paste"]["max_objects_per_scan"] = 1
        self.assertEqual(basic, on)


if __name__ == "__main__":
    unittest.main()
