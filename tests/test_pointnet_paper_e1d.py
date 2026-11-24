from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

import lib.utils.raw_point_copy_paste_paper_e1d as paper
from lib.utils.raw_point_copy_paste_e0b import PlacementRejected


CP = yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1d_paper3.yaml").read_text())["copy_paste"]
POLICY = CP["paper"]


def candidate(cid=2, context=50, overlap=0, background=9, rng=None):
    target = np.zeros(100, dtype=int)
    target[:context] = background
    target[-overlap:] = 1 if overlap else target[-overlap:]
    return paper.select_cells(np.arange(100), target, cid, POLICY, rng or Mock(random=lambda: 0.1))


class Gates(unittest.TestCase):
    def test_config(self):
        self.assertEqual(POLICY["protected"], [1,2,3,4,5,6,7,8,13,14,16,18,19])
        self.assertEqual((CP["max_candidate_trials"], CP["max_objects_per_scan"], CP["probability"]), (10,3,.5))

    def test_background_classes_and_context_boundary(self):
        for cid in (2,3,4,5,6):
            for raw in (9,10,11,12,17):
                if raw in ((9,11) if cid==6 else (9,10)):
                    self.assertEqual(candidate(cid,11,background=raw)[2]["context"], .11)
                else:
                    with self.assertRaisesRegex(PlacementRejected,"context"):
                        candidate(cid,11,background=raw)
            with self.assertRaisesRegex(PlacementRejected,"context"):
                candidate(cid,10)

    def test_overlap_and_visibility_boundaries(self):
        for cid, maximum in ((2,20),(3,20),(4,30),(5,30),(6,20)):
            keep, cells, score = candidate(cid, overlap=maximum)
            self.assertEqual(score["overlap"], maximum/100)
            self.assertEqual(len(cells), 100-maximum)
            self.assertEqual(keep.sum(),len(cells))
            with self.assertRaisesRegex(PlacementRejected,"occlusion"):
                candidate(cid,overlap=maximum+1)
        rng = Mock()
        self.assertEqual(paper.visibility_reason(.299, POLICY, rng), "minimum_visibility")
        for v in (.3,.95):
            self.assertIsNone(paper.visibility_reason(v,POLICY,rng))
        candidate(overlap=5, rng=rng)
        rng.random.assert_not_called()
        rng.random.return_value=.301
        with self.assertRaisesRegex(PlacementRejected,"high_visibility_sampling"):
            candidate(overlap=4,rng=rng)
        rng.random.return_value=.299
        candidate(overlap=4,rng=rng)

    def test_unique_cells_and_all_blocked_points_removed(self):
        point_cells=np.r_[np.arange(100),np.repeat(99,1000)]
        target=np.full(100,9);target[99]=19
        keep,visible,s=paper.select_cells(point_cells,target,4,POLICY,Mock(random=lambda:.1))
        self.assertEqual((s["context"],s["overlap"],s["visibility"]),(.99,.01,.99))
        self.assertEqual(keep.sum(),99)
        np.testing.assert_array_equal(point_cells[keep],visible)

    def test_three_objects_reserve_retained_cells(self):
        target=np.zeros((100,4), np.float32)
        packed=np.full(100,40,np.uint32)
        sample=dict(points=target,packed=packed,epoch=0,index=0,stats=dict(candidate_trials=0,labels_before=[0]*20))
        db=Mock()
        patch_points=np.tile([0,0,0,.7,11],(20,1)).astype(np.float32)
        db.sample.side_effect=[(dict(learning_id=2,donor_id=str(i)),patch_points) for i in range(3)]
        # Each proposal overlaps the previous object by 2/20 cells.
        blocks=[np.arange(100),np.arange(20),np.arange(18,38),np.arange(36,56)]
        def project(*args):
            cells=blocks.pop(0)
            return cells,np.arange(len(cells)),np.ones(len(cells),bool)
        vis=Mock(random=lambda:.1)
        rng=np.random.default_rng(5)
        with patch.object(paper,"grid",side_effect=project), patch.object(paper,"place_basic",return_value=(patch_points[:,:4],{})), patch.object(paper.np.random,"default_rng",side_effect=[rng,vis]):
            paper.augment(sample,db,{40:9,11:2},dict(CP,probability=1.),"cpu")
        self.assertEqual([p["added_points"] for p in sample["placement"]],[20,18,18])
        self.assertEqual([p["synthetic_instance_id"] for p in sample["placement"]],[1,2,3])
        self.assertEqual(len(sample["points"]),len(sample["packed"]))
        np.testing.assert_array_equal(sample["packed"][:100],packed)
        self.assertTrue(np.all((sample["packed"][100:] & 65535)==11))
        self.assertEqual(sample["paste_points"].sum(),56)
        counts=Counter();paper.tally(counts,sample["stats"])
        self.assertEqual(paper.epoch_stats(counts)["scans_with_3"],1)

    def test_off_is_noop(self):
        sample=dict(stats={}, index=0,epoch=0)
        paper.augment(sample,None,None,dict(CP,enabled=False),"cpu")
        self.assertEqual(sample, dict(stats={}, index=0,epoch=0))

    def test_retry_counts(self):
        sample=dict(points=np.ones((1,4),np.float32),packed=np.array([40],np.uint32),epoch=0,index=0,
                    stats=dict(candidate_trials=0,labels_before=[0]*20))
        db=Mock()
        donor=np.array([[0,0,0,.5,11]],np.float32)
        db.sample.side_effect=[(dict(learning_id=2,donor_id=str(i)),donor) for i in range(3)]
        with patch.object(paper,"place_basic",side_effect=PlacementRejected("outside_sensor_fov")):
            paper.augment(sample,db,{40:9,11:2},dict(CP,probability=1.),"cpu")
        self.assertEqual(sample["stats"]["exhausted_objects"],3)
        self.assertEqual(sample["stats"]["candidate_trials"],30)
        self.assertEqual(sample["stats"]["successful_scans"],0)


if __name__ == "__main__":
    unittest.main()
