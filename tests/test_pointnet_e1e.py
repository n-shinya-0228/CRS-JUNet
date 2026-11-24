import unittest
from unittest.mock import Mock

import numpy as np

from build_donors_e1e import candidates, components
from lib.utils import raw_point_copy_paste_e1e as e
from lib.utils import raw_point_copy_paste_paper_e1d as d
from train_pointnet_paper_e1e import config, base_config, Trainer, E1d


class Extraction(unittest.TestCase):
    def test_original_riders_never_paired(self):
        rng=np.random.default_rng(0)
        p=rng.uniform([0,0,0,0],[1.4,.5,1.5,1],(340,4))
        for cid,raw in [(7,253),(8,255)]:
            label=np.full(340,(12<<16)|raw,np.uint32)
            ids,method,reasons,*_=next(candidates(p,label,np.full(340,cid),cid))
            np.testing.assert_array_equal(ids,np.arange(340))
            self.assertEqual((method,reasons),("whole_instance",[]))
            mapped=np.full(340,cid);mapped[-1]=6
            self.assertIn("mixed_instance",next(candidates(p,label,mapped,cid))[2])
            self.assertEqual(next(candidates(p,label & 65535,np.full(340,cid),cid))[1],"isolated_rider_cluster")

    def test_static_separation_and_labels(self):
        z=np.linspace(.6,2.2,24)
        for cid,raw in [(18,80),(19,81)]:
            obj=np.column_stack((np.zeros(len(z)),np.zeros(len(z)),z,np.ones(len(z))))
            if cid==19:
                obj=np.concatenate([obj+[0,dy,0,0] for dy in np.linspace(-.3,.3,8)])
            objects=np.concatenate([obj+[12,0,0,0],obj+[16,0,0,0]])
            ground=objects.copy();ground[:,2]=0
            p=np.concatenate([objects,ground]);mapped=np.repeat([cid,9],len(objects))
            labels=np.repeat([raw,40],len(objects)).astype(np.uint32)
            rows=list(candidates(p,labels,mapped,cid))
            self.assertEqual(len(rows),2)
            for n,(idx,_,reasons,*_) in enumerate(rows):
                self.assertEqual(reasons,[])
                np.testing.assert_array_equal(idx,np.arange(n*len(obj),(n+1)*len(obj)))
        with self.assertRaises(ValueError): e.check_sequence("08")


class Selection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch,cls.data=config()
        cls.policy=cls.arch["copy_paste"]["paper"]

    def test_contract(self):
        base,_=base_config()
        self.assertEqual({k:v for k,v in self.arch.items() if k!="copy_paste"},
                         {k:v for k,v in base.items() if k!="copy_paste"})
        self.assertIs(Trainer._mix_losses,E1d._mix_losses)
        self.assertIs(Trainer.validate,E1d.validate)
        self.assertIs(e.select_cells,d.select_cells)
        p=self.policy
        self.assertEqual(set(p["allowed_background"]),set(e.CLASSES))
        for c in e.CLASSES:
            self.assertEqual(p["allowed_background"][c], [9,10] if c<=5 else [9,11] if c<=8 else [9,11,17])
            self.assertEqual(p["max_overlap"][c],.4 if c>=18 else .3 if c in (4,5) else .2)

    def test_protected_cells_and_exact_boundaries(self):
        ids=np.arange(100);rng=Mock();rng.random.return_value=.31
        for c in e.CLASSES:
            n=int(100*self.policy["max_overlap"][c])
            labels=np.full(100,9);labels[:n]=1
            keep,vis,_=e.select_cells(np.r_[ids,ids[:4]],labels,c,self.policy,rng)
            self.assertFalse(keep[:n].any());self.assertFalse(keep[-4:].any())
            labels[n]=18
            with self.assertRaisesRegex(e.PlacementRejected,"occlusion"):
                e.select_cells(ids,labels,c,self.policy,rng)
        labels=np.zeros(100,int);labels[:10]=9
        with self.assertRaisesRegex(e.PlacementRejected,"context"):
            e.select_cells(ids,labels,7,self.policy,rng)
        self.assertIsNone(d.visibility_reason(.30,self.policy,rng))
        self.assertIsNone(d.visibility_reason(.95,self.policy,rng))
        self.assertEqual(d.visibility_reason(.29,self.policy,rng),"minimum_visibility")
        self.assertEqual(d.visibility_reason(.951,self.policy,rng),"high_visibility_sampling")
        rng.random.return_value=.29
        self.assertIsNone(d.visibility_reason(1.,self.policy,rng))
        seen=set();e.unique_mask(ids,seen)
        with self.assertRaisesRegex(e.PlacementRejected,"duplicate_mask"): e.unique_mask(ids.copy(),seen)

    def test_reprojection_and_fov(self):
        p=np.array([[0,-.04,0,.3,80],[0,.04,.2,.5,80]],np.float32)
        for r in (6.,43.):
            q,_=e.pose(p,[-r,0],0,-1.5,.03)
            cells,_,valid=e.grid(q,"cpu")
            self.assertTrue(valid.all());self.assertEqual(set(cells%512),{0,511})
            self.assertTrue(all(np.abs(cells//512-r*10)<=1))
        with self.assertRaisesRegex(e.PlacementRejected,"sensor_fov"): e.pose(p,[2,0],0,-2.,0.)
        with self.assertRaisesRegex(e.PlacementRejected,"polar_radius"): e.pose(p,[52,0],0,-2.,0.)


if __name__=="__main__":
    unittest.main()
