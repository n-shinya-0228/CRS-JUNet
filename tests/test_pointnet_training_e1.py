import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"lib/trainer"))
from trainer_PointNet_RawCopyPaste_E1 import Trainer, augmentation, class_weights, deterministic_encoder
from lib.utils.pointnet_c2_recipe_e1 import EarlyStopping, FocalLoss, SobelFilter, build_scheduler
from lib.utils.raw_point_copy_paste_e0c import check_sequence, quality, stable_seed, DonorDatabase
from lib.utils.raw_point_copy_paste_e0b import apply_dropout, deterministic_labels
from lib.losses import Lovasz_softmax


class E1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data=yaml.safe_load(Path("config/labels/semantic-kitti.yaml").read_text())
        cls.arch=yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1.yaml").read_text())

    def test_train_split_strict(self):
        for seq in ("00","07","09","10"):
            check_sequence(seq)
        for seq in ("08","11","21","0","../00"):
            with self.assertRaises(ValueError):
                check_sequence(seq)

    def test_formal_configs_differ_only_in_cp_enabled(self):
        off=yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1a_cp0.yaml").read_text())
        on=yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1b_cp_on.yaml").read_text())
        self.assertIs(off["copy_paste"]["enabled"],False)
        self.assertIs(on["copy_paste"]["enabled"],True)
        on["copy_paste"]["enabled"]=False
        self.assertEqual(off,on)
        self.assertEqual(off,self.arch)

    def test_streams_are_stable_distinct_and_independent(self):
        self.assertEqual(stable_seed(0,23,"regular"),stable_seed(0,23,"regular"))
        self.assertEqual(len({stable_seed(e,i,s) for e in range(2) for i in range(3) for s in ("regular","copy_paste","scan_order")}),18)

    def test_augmentation_replay_alignment_and_denoising(self):
        x=torch.ones(24,512,512)
        y=torch.arange(512).repeat(512,1)%19+1
        p=torch.ones(1,512,512)
        result=dict(hybrid24=x,label=y,paste_mask=p)
        for seed in range(5):
            a,b,c=augmentation(result,seed)
            a2,b2,c2=augmentation(result,seed)
            self.assertTrue(torch.equal(a,a2) and torch.equal(b,b2) and torch.equal(c,c2))
            self.assertTrue((b>0).all())
            self.assertTrue(torch.equal(a[23:24],c))
            self.assertTrue(torch.equal(a[0],a[23]))

    def test_dropout_retains_semantic_but_boundary_mask_zeros(self):
        trainer=Trainer.__new__(Trainer)
        trainer.sobel=SobelFilter()
        x=torch.ones(24,8,8); y=torch.ones(8,8,dtype=torch.long); p=torch.ones(1,8,8)
        keep=torch.ones(8,8,dtype=torch.bool);keep[2:4,2:4]=False
        a,b,c=apply_dropout(x,y,p,keep,policy="legacy_denoising")
        self.assertTrue(torch.equal(b,y))
        self.assertEqual(a[:,~keep].sum(),0)
        boundary=trainer.boundary(b[None],a[None,23:24])
        self.assertEqual(boundary[0,0,~keep].sum(),0)

    def test_determinism_scope_restored(self):
        previous=torch.are_deterministic_algorithms_enabled()
        precision=torch.get_float32_matmul_precision()
        with self.assertRaisesRegex(ValueError,"deliberate"):
            with deterministic_encoder():
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertEqual(torch.get_float32_matmul_precision(),"highest")
                raise ValueError("deliberate")
        self.assertEqual(torch.are_deterministic_algorithms_enabled(),previous)
        self.assertEqual(torch.get_float32_matmul_precision(),precision)

    def test_tie_policy_permutation_invariant(self):
        cells=np.array([2]); inv=np.array([0,0,0])
        a=np.array([40,48,44],np.uint32)
        b=deterministic_labels(cells,inv,a,self.data["learning_map"])
        self.assertTrue(torch.equal(b,deterministic_labels(cells,inv,a[::-1].copy(),self.data["learning_map"])))

    def test_packed_instance_quality_rejects_whole_mixed_instance(self):
        rng=np.random.default_rng(3)
        xyz=rng.uniform([-.3,-.2,-.8],[.3,.2,.8],(300,3)).astype(np.float32)
        pts=np.column_stack((xyz,np.full(300,.4))).astype(np.float32)
        packed=np.full(300,(12<<16)|30,np.uint32)
        donor,center,row=quality(pts,packed,self.data["learning_map"])
        self.assertFalse(row["reasons"])
        self.assertEqual(len(donor),len(pts))
        packed[0]=np.uint32((12<<16)|11)
        self.assertIn("mixed_raw_semantic_ids",quality(pts,packed,self.data["learning_map"])[2]["reasons"])
        self.assertIn("mixed_learning_classes",quality(pts,packed,self.data["learning_map"])[2]["reasons"])
        packed[:]=30
        self.assertIn("invalid_or_mixed_instance",quality(pts,packed,self.data["learning_map"])[2]["reasons"])

    def test_class_weights_match_historical_formula(self):
        result=class_weights(self.data,.01)
        content=torch.zeros(20)
        for raw,freq in self.data["content"].items():
            content[self.data["learning_map"][raw]]+=freq
        expected=(1/(content/(content.sum()+1e-6)+.01)).clamp(max=20.)
        expected[0]=0
        torch.testing.assert_close(result,expected,rtol=0,atol=0)

    def test_auxiliary_decay_only_aux2_aux4(self):
        trainer=Trainer.__new__(Trainer)
        for epoch,ratio in ((0,1.),(49,1.),(50,1.),(55,.5),(60,0.),(69,0.)):
            trainer.set_epoch_weights(epoch)
            self.assertEqual(trainer.w_aux2,.3*ratio)
            self.assertEqual(trainer.w_aux4,.15*ratio)
            self.assertEqual(trainer.w_aux8,.1)
            self.assertEqual(trainer.w_paste,0.)

    def test_losses_semantic_hidden_supervision_boundary_mask(self):
        trainer=Trainer.__new__(Trainer)
        trainer.loss_w=class_weights(self.data,.01)
        trainer.criterion_main=FocalLoss(trainer.loss_w,gamma=1.,ignore_index=0)
        trainer.lovasz=Lovasz_softmax(ignore=0)
        trainer.boundary_criterion=torch.nn.BCEWithLogitsLoss(reduction="none")
        trainer.sobel=SobelFilter()
        trainer.set_epoch_weights(0)
        labels=torch.ones(1,8,8,dtype=torch.long)
        mask=torch.zeros(1,1,8,8)
        logits=torch.randn(1,20,8,8,requires_grad=True)
        outs={k:logits for k in ("logits","aux2","aux4","aux8")}
        outs["boundary"]=torch.randn(1,1,8,8,requires_grad=True)
        loss=trainer._mix_losses(outs,labels,trainer.boundary(labels,mask),mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        self.assertEqual(float(outs["boundary"].grad.abs().sum()),0)

    def test_scheduler_preview_and_resume(self):
        p=torch.nn.Parameter(torch.zeros(()))
        opt=torch.optim.AdamW([p],lr=.002,weight_decay=.005)
        scheduler=build_scheduler(opt,self.arch["train"])
        rates=[]
        for epoch in range(7):
            rates.append(opt.param_groups[0]["lr"])
            opt.step();scheduler.step()
        np.testing.assert_allclose(rates[:6],[.0002,.00056,.00092,.00128,.00164,.002])
        self.assertLess(rates[6],rates[5])

    def test_early_stop_independent_of_strict_best(self):
        stop=EarlyStopping()
        self.assertFalse(stop.update(.5,0))
        self.assertFalse(stop.update(.5001,1))
        for epoch in range(2,30):
            self.assertFalse(stop.update(.5001,epoch))
        self.assertTrue(stop.update(.5001,30))
        copy=EarlyStopping();copy.load_state_dict(stop.state_dict())
        self.assertEqual(copy.state_dict(),stop.state_dict())

    def test_no_legacy_training_path_or_cuda_worker(self):
        tree=ast.parse(Path("lib/dataset/SemanticKitti_PointNet_RawCopyPaste_E1.py").read_text())
        source=ast.unparse(tree)
        self.assertNotIn(".cuda(",source)
        self.assertNotIn("load_encoder(",source)
        self.assertNotIn("CachedValidation",source)
        self.assertNotIn("compatibility_off",source)

    def test_incomplete_gate_never_starts_formal_training(self):
        from train_pointnet_raw_copy_paste_e1 import check_gate
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"gate.json"
            for report in ({"verdict":"FAIL"},{"verdict":"CONDITIONAL PASS"},
                           {"verdict":"PASS","mandatory_conditions":{}}):
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    check_gate(path)


if __name__=="__main__":
    unittest.main()
