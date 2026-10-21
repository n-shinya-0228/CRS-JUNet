import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

import train_pointnet_teacher_student_d4 as entry
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import ALPHA_TABLE, build_alpha_maps
from lib.models.SJNet3_PointNet_Gated_Local3_TeacherStudent import SJNet3_PointNet_Gated_Local3_TeacherStudent

d4 = entry.load_trainer()
torch.set_num_threads(2)


class TeacherStudentTests(unittest.TestCase):
    def model(self):
        model = SJNet3_PointNet_Gated_Local3_TeacherStudent(nclasses=20, base_ch=8, aspp_out=16, drop=0)
        model.initialize(copy.deepcopy(model.teacher.state_dict()))
        return model

    def test_strict_state_all_five_outputs_same_input_and_masks(self):
        model = self.model().eval()
        teacher_hash = d4.d1.state_digest(model.teacher.state_dict())
        self.assertEqual(teacher_hash, d4.d1.state_digest(model.student.state_dict()))
        self.assertEqual(set(model.teacher.state_dict()), set(model.student.state_dict()))
        self.assertFalse(any("alpha" in key for key in model.student.state_dict()))
        x = torch.randn(1, 24, 32, 320)
        seen = []
        handles = [net.register_forward_pre_hook(lambda net, args: seen.append(args[0].data_ptr()))
                   for net in (model.teacher, model.student)]
        for mask in (0, 1):
            x[:, 23] = mask
            with torch.no_grad():
                old_stage1 = model.student(x)
                _, alpha = build_alpha_maps(old_stage1["logits"], x[:, 23:24])
                expected = model.student(x, alpha_map=alpha)
                seen.clear()
                actual = model(x)
            self.assertEqual(set(actual), {"logits", "aux2", "aux4", "aux8", "boundary"})
            self.assertEqual(seen, [x.data_ptr(), x.data_ptr()])
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            _, soft = model.alpha_maps(x)
            self.assertFalse(soft.requires_grad)
            self.assertFalse(torch.is_inference(soft))
            self.assertTrue(torch.isfinite(soft).all())
            self.assertGreaterEqual(soft.min().item(), .6-1e-6)
            self.assertLessEqual(soft.max().item(), 1.6+1e-6)
            if not mask:
                torch.testing.assert_close(soft, torch.full_like(soft, 1.2), atol=0, rtol=0)
        for handle in handles:
            handle.remove()
        self.assertEqual(teacher_hash, d4.d1.state_digest(model.teacher.state_dict()))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 23, 32, 320))

    def test_three_updates_student_only_gradients_ema_and_teacher_eval(self):
        model = self.model().train()
        self.assertFalse(model.teacher.training)
        self.assertTrue(model.student.training)
        optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-4, weight_decay=.005)
        teacher_ids = {id(p) for p in model.teacher.parameters()}
        self.assertFalse(teacher_ids & {id(p) for group in optimizer.param_groups for p in group["params"]})
        helper = d4.Trainer.__new__(d4.Trainer)
        helper.model, helper.use_ema, helper.ema_decay = model.student, True, .99
        helper._build_ema_model()
        initial = d4.d1.state_digest(model.teacher.state_dict())
        x = torch.randn(1, 24, 32, 320)
        x[:, 23] = 1
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(output.square().mean() for output in model(x).values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.student.parameters()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.teacher.parameters()))
            before = {key:value.clone() for key, value in helper.ema_model.state_dict().items()}
            optimizer.step()
            helper._update_ema()
            for key, value in helper.ema_model.state_dict().items():
                torch.testing.assert_close(value, .99*before[key]+.01*model.student.state_dict()[key])
        self.assertEqual(d4.d1.state_digest(model.teacher.state_dict()), initial)
        self.assertNotEqual(d4.d1.state_digest(model.student.state_dict()), initial)
        self.assertNotEqual(d4.d1.state_digest(helper.ema_model.state_dict()), initial)
        self.assertEqual(model.fixed_table, ALPHA_TABLE)
        self.assertTrue(all(not p.requires_grad for p in helper.ema_model.parameters()))

    def test_original_c2_loss_coefficients_ignore_and_cp_zero(self):
        trainer = d4.Trainer.__new__(d4.Trainer)
        weights = torch.ones(20)
        weights[0] = 0
        trainer.criterion_main = d4._base.FocalLoss(weight=weights, gamma=1., ignore_index=0)
        trainer.lovasz = d4.Lovasz_softmax(ignore=0)
        trainer.sobel = d4._base.SobelFilter()
        trainer.boundary_criterion = nn.BCEWithLogitsLoss(reduction="none")
        trainer.w_aux2, trainer.w_aux4, trainer.w_aux8 = .3, .15, .1
        trainer.w_lovasz, trainer.w_boundary, trainer.w_paste = .5, .2, 0
        outputs = {key:torch.randn(2, 20, 8, 8, requires_grad=True) for key in ("logits", "aux2", "aux4", "aux8")}
        outputs["boundary"] = torch.randn(2, 1, 8, 8, requires_grad=True)
        labels = torch.randint(0, 20, (2, 8, 8))
        mask = torch.randint(0, 2, (2, 1, 8, 8)).float()
        boundary = trainer._compute_boundary_gt(labels)*mask
        expected = sum(weight*trainer.criterion_main(outputs[key], labels) for key, weight in
                       (("logits", 1), ("aux2", .3), ("aux4", .15), ("aux8", .1)))
        expected += .5*trainer.lovasz(outputs["logits"].softmax(1), labels)
        expected += .2*(trainer.boundary_criterion(outputs["boundary"], boundary)*mask).sum()/(mask.sum()+1e-6)
        actual = trainer.full_loss(outputs, labels, mask)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.backward()
        self.assertTrue(all(value.grad is not None for value in outputs.values()))
        trainer.device = torch.device("cpu")
        batch = [None]*16
        batch[14] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "all-zero"):
            trainer.batch_input(batch)

    def test_effective_batch_groups_determinism_drop_last_and_epoch_seed(self):
        reference = list(d4.EffectiveBatchSampler(11, 3))
        self.assertEqual(len(reference), 3)
        self.assertEqual(len({i for group in reference for i in group}), 9)
        for micro in (1, 2, 3):
            sampler = d4.EffectiveBatchSampler(11, micro)
            groups = list(sampler)
            self.assertEqual(len(groups), len(sampler))
            self.assertEqual([i for group in groups for i in group], [i for group in reference for i in group])
            self.assertEqual([len(g) for g in groups], ({1:[1]*9, 2:[2, 1]*3, 3:[3]*3})[micro])
            sampler.epoch = 1
            self.assertNotEqual(list(sampler), groups)
        with self.assertRaises(ValueError):
            d4.EffectiveBatchSampler(9, 4)

    def test_scheduler_early_stop_rng_and_checkpoint_roundtrip(self):
        arch = yaml.safe_load(entry.DEFAULT_ARCH.read_text())
        d4.validate_config(arch)
        optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=1e-4)
        scheduler = d4.build_scheduler(optimizer, arch["train"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        stopping = d4.EarlyStopping(**arch["train"]["early_stopping"])
        stopping.update(.5097, 0)
        strict_best = .5097
        for epoch in range(1, 9):
            score = .5097+epoch*1e-5
            strict_best = max(strict_best, score)
            self.assertEqual(stopping.update(score, epoch), epoch >= 8)
        self.assertGreater(strict_best, stopping.best_score)
        with patch.object(torch.cuda, "is_available", return_value=False):
            state = d4.rng_state()
            expected = (random.random(), np.random.rand(), torch.rand(3))
            d4.restore_rng(state)
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.rand(), expected[1])
            torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)
        saved = dict(experiment="Step D.4", arch_cfg=arch, data_cfg={}, metadata_id="id", teacher_identity={"sha":"t"},
                     alpha_table=list(ALPHA_TABLE), copy_paste_prob=0, reproduction_gate_passed=True)
        d4.validate_resume(saved, arch, {}, "id", {"sha":"t"})
        for key, value in (("teacher_identity", {}), ("alpha_table", []), ("reproduction_gate_passed", False), ("metadata_id", "bad")):
            with self.assertRaises(ValueError):
                d4.validate_resume({**saved, key:value}, arch, {}, "id", {"sha":"t"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"latest.path"
            d4.atomic_checkpoint(path, dict(saved, scheduler=scheduler.state_dict(), early_stopping=stopping.state_dict()))
            loaded = torch.load(path, weights_only=False)
            restored = d4.EarlyStopping(**arch["train"]["early_stopping"])
            restored.load_state_dict(loaded["early_stopping"])
            self.assertEqual(restored.state_dict(), stopping.state_dict())
            self.assertFalse(path.with_suffix(".path.tmp").exists())

    def test_protect_existing_experiments_and_disable_formal_training_before_gate(self):
        with self.assertRaises(ValueError):
            entry.check_output(entry.arguments(["--dataset", "SemanticKitti", "--log", "logs/SJNet3_PointNet_stepD2_class_fixed_alpha"]))
        with tempfile.TemporaryDirectory(prefix="SJNet3_PointNet_stepD4_", dir=entry.ROOT/"logs") as tmp:
            args = entry.arguments(["--dataset", "SemanticKitti", "--log", tmp])
            with self.assertRaises(FileExistsError):
                entry.check_output(args)
        trainer = d4.Trainer.__new__(d4.Trainer)
        trainer.gate_passed = False
        with self.assertRaisesRegex(RuntimeError, "reproduction gate"):
            trainer.train()

    def test_automated_final_diagnostics_package_and_nineteen_class_report(self):
        trainer = d4.Trainer.__new__(d4.Trainer)
        trainer.ARCH = yaml.safe_load(entry.DEFAULT_ARCH.read_text())
        trainer.DATA, trainer.metadata_id, trainer.teacher_identity = {}, "id", {"sha":"teacher"}
        trainer.metadata = {"protected_files":{}}
        trainer.model, trainer.ema_model = nn.Linear(2, 2), nn.Linear(2, 2)
        trainer.system = type("System", (), {"teacher":nn.Linear(2, 2)})()
        trainer.assert_teacher_fixed = lambda:None
        trainer.c2_best_miou, trainer.total_train_seconds = .508252935, 1234.
        trainer.early_stopping = d4.EarlyStopping(**trainer.ARCH["train"]["early_stopping"])
        trainer.history = [dict(epoch=0), dict(epoch=1, peak_allocated_GiB=9., peak_reserved_GiB=10.)]
        trainer.reference_run = json.loads((entry.ROOT/"logs/SJNet3_PointNet_stepD2_class_fixed_alpha/results.json").read_text())
        trainer.reference = trainer.reference_run["results"]["D2_Soft"]
        def evaluate(student=None, mode="soft", tag="validation"):
            return dict(miou=.51 if tag == "best_ema_soft" else .50, seconds=10., forward_cuda_seconds=8.,
                        per_class_iou={name:.51 for name in d4.CLASS_NAMES[1:]})
        trainer.validate = evaluate
        with tempfile.TemporaryDirectory() as tmp:
            trainer.output = Path(tmp)
            saved = dict(experiment="Step D.4", arch_cfg=trainer.ARCH, data_cfg={}, metadata_id="id",
                         teacher_identity=trainer.teacher_identity, alpha_table=list(ALPHA_TABLE),
                         copy_paste_prob=0, reproduction_gate_passed=True, model_ema=trainer.ema_model.state_dict(),
                         model_train=trainer.model.state_dict(), best_val_epoch=1, best_val_iou=.51)
            d4.atomic_checkpoint(trainer.output/"best_val.path", saved)
            d4.d1.atomic_json(trainer.output/"epoch0_validation.json", {"miou":d4.EXPECTED_D2})
            result = trainer.finalize()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["best_epoch"], 1)
            self.assertAlmostEqual(result["delta_vs_D2_pp"], 100*(.51-d4.EXPECTED_D2))
            self.assertEqual(len((trainer.output/"best_class_comparison.csv").read_text().splitlines()), 20)
            package = torch.load(trainer.output/"inference_package.path", weights_only=False)
            self.assertEqual(package["alpha_table"], list(ALPHA_TABLE))
            self.assertEqual(set(package["teacher"]), set(trainer.system.teacher.state_dict()))
            self.assertIn("D4_epoch0", result["comparisons"])
            self.assertIn("motorcyclist_above_zero", result["diagnostics"])


if __name__ == "__main__":
    unittest.main()
