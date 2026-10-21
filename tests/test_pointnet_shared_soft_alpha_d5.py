import contextlib
import copy
import inspect
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

import train_pointnet_shared_soft_alpha_d5 as entry
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    ALPHA_TABLE, CLASS_NAMES, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps,
)
from lib.models.SJNet3_PointNet_Gated_Local3_SharedSoftAlpha import SJNet3_PointNet_Gated_Local3_SharedSoftAlpha

d5 = entry.load_trainer()
torch.set_num_threads(2)


class SharedSoftAlphaTests(unittest.TestCase):
    def model(self):
        return SJNet3_PointNet_Gated_Local3_SharedSoftAlpha(nclasses=20, base_ch=8, aspp_out=16, drop=0)

    def test_full_model_matches_c2_random_initialization_not_trained(self):
        options = dict(in_channels=24, nclasses=20, drop=.3)
        torch.manual_seed(0)
        original = SJNet3_PointNet_Gated_Local3(**options)
        expected = d5.d1.state_digest(original.state_dict())
        keys = set(original.state_dict())
        del original
        torch.manual_seed(0)
        model = SJNet3_PointNet_Gated_Local3_SharedSoftAlpha(**options)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 28876586)
        self.assertEqual(d5.d1.state_digest(model.state_dict()), expected)
        self.assertEqual(set(model.state_dict()), keys)
        ref = json.loads((entry.ROOT/"logs/SJNet3_PointNet_stepD2_class_fixed_alpha/results.json").read_text())
        self.assertNotEqual(expected, ref["metadata"]["model_state_sha256"])
        self.assertFalse(any("teacher" in key or "alpha" in key for key in keys))

    def test_global_and_soft_all_five_outputs_masks_and_table(self):
        model = self.model().eval()
        ref = SJNet3_PointNet_Gated_Local3_ClassAlpha(nclasses=20, base_ch=8, aspp_out=16, drop=0).eval()
        ref.load_state_dict(model.state_dict(), strict=True)
        x = torch.randn(1, 24, 32, 320)
        for observed in (0, 1):
            x[:, 23] = observed
            with torch.no_grad():
                stage1 = ref(x)
                actual = model.forward_with_alpha(x)
                for name in stage1:
                    torch.testing.assert_close(actual[name], stage1[name], atol=0, rtol=0)
                _, soft = build_alpha_maps(stage1["logits"], x[:, 23:24])
                expected = ref(x, alpha_map=soft)
                actual = model(x)
                self.assertEqual(set(actual), {"logits", "aux2", "aux4", "aux8", "boundary"})
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            logits, alpha = model.stage1(x)
            self.assertFalse(logits.requires_grad or alpha.requires_grad)
            self.assertFalse(torch.is_inference(alpha))
            self.assertGreaterEqual(alpha.min().item(), .6-1e-6)
            self.assertLessEqual(alpha.max().item(), 1.6+1e-6)
            if not observed:
                torch.testing.assert_close(alpha, torch.full_like(alpha, 1.2), atol=0, rtol=0)
        self.assertEqual(list(inspect.signature(model.stage1).parameters), ["x"])
        self.assertEqual(ALPHA_TABLE, (1.2, 1., 1.6, 1.2, 1.6, 1.6, .6, .8, 1.2, .8, .6, .8, 1.6, 1.1, 1.6, 1.6, 1., 1.6, 1.3, 1.3))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 23, 32, 320))

    def test_shared_object_gradient_modes_three_updates_and_ema(self):
        model = self.model().train()
        helper = d5.Trainer.__new__(d5.Trainer)
        helper.model, helper.use_ema, helper.ema_decay = model, True, .99
        helper._build_ema_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.005)
        identities = {id(p) for p in model.parameters()}
        calls = []
        handle = model.hand_stem.register_forward_pre_hook(
            lambda module, args:calls.append((id(module), module.training, torch.is_grad_enabled(), args[0].data_ptr())))
        x = torch.randn(1, 24, 32, 320)
        x[:, 23] = 1
        initial = d5.d1.state_digest(model.state_dict())
        old_stage1 = model.stage1(x)[0].clone()
        for _ in range(3):
            calls.clear()
            optimizer.zero_grad(set_to_none=True)
            loss = sum(value.square().mean() for value in model(x).values())
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], (id(model.hand_stem), False, False, x.data_ptr()))
            self.assertEqual(calls[1], (id(model.hand_stem), True, True, x.data_ptr()))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            optimizer.step()
            helper._update_ema()
        handle.remove()
        self.assertEqual({id(p) for p in model.parameters()}, identities)
        self.assertNotEqual(d5.d1.state_digest(model.state_dict()), initial)
        self.assertNotEqual(d5.d1.state_digest(helper.ema_model.state_dict()), initial)
        self.assertFalse(torch.equal(old_stage1, model.stage1(x)[0]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in helper.ema_model.parameters()))

    def test_eval_restore_on_exceptions_mixed_modes_and_batchnorm(self):
        model = self.model().train()
        model.point_gate.eval()
        modes = [module.training for module in model.modules()]
        def fail(*args):
            raise RuntimeError("injected Stage 1 failure")
        handle = model.hand_stem.register_forward_pre_hook(fail)
        with self.assertRaisesRegex(RuntimeError, "Stage 1 failure"):
            model(torch.randn(1, 24, 32, 320))
        handle.remove()
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertIsNone(model.point_context._alpha_map)
        model.mode_probe = nn.BatchNorm2d(7)
        model.train()
        handle = model.hand_stem.register_forward_pre_hook(lambda module, args:model.mode_probe(args[0]))
        x = torch.randn(1, 24, 32, 320)
        x[:, 23] = 1
        model.stage1(x)
        self.assertEqual(model.mode_probe.num_batches_tracked.item(), 0)
        model(x)
        self.assertEqual(model.mode_probe.num_batches_tracked.item(), 1)
        handle.remove()

    def test_c2_config_losses_schedule_aux_decay_and_resume_guards(self):
        arch = yaml.safe_load(entry.DEFAULT_ARCH.read_text())
        d5.validate_config(arch)
        for key, value in (("lr", 1e-4), ("report_batch", 100), ("max_epochs", 20), ("amp", False)):
            changed = copy.deepcopy(arch)
            changed["train"][key] = value
            with self.assertRaises(ValueError):
                d5.validate_config(changed)
        trainer = d5.Trainer.__new__(d5.Trainer)
        for epoch, expected in ((0, (.3, .15)), (50, (.3, .15)), (55, (.15, .075)), (60, (0, 0)), (69, (0, 0))):
            trainer.set_epoch_loss_weights(epoch)
            self.assertEqual((trainer.w_aux2, trainer.w_aux4), expected)
        optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=.002)
        scheduler = d5.build_scheduler(optimizer, arch["train"])
        rates = []
        for _ in range(6):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        torch.testing.assert_close(torch.tensor(rates), torch.tensor([.0002, .00056, .00092, .00128, .00164, .002]))
        saved = dict(experiment="Step D.5", arch_cfg=arch, data_cfg={}, metadata_id="identity",
                     alpha_table=list(ALPHA_TABLE), copy_paste_prob=0, pretrained_initialization=False, initialization_verified=True)
        d5.validate_resume(saved, arch, {}, "identity")
        for key, value in (("experiment", "Step C.2"), ("experiment", "Step D.4"), ("pretrained_initialization", True),
                           ("alpha_table", []), ("metadata_id", "other")):
            with self.assertRaises(ValueError):
                d5.validate_resume({**saved, key:value}, arch, {}, "identity")
        trainer.device = torch.device("cpu")
        batch = [None]*16
        batch[14] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "all-zero"):
            trainer.batch_input(batch)

    def test_progress_frequency_effective_denominator_format_and_read_only_meters(self):
        self.assertEqual([index for index in range(6376) if d5.should_report(index)], [0, 3000, 6000])
        logger, stream = self.logger()
        meters = [d5.AverageMeter() for _ in range(3)]
        for index, meter in enumerate(meters):
            meter.update(index+.1, 3)
            meter.update(index+.2, 3)
        before = copy.deepcopy([meter.__dict__ for meter in meters])
        d5.log_training_progress(logger, 2, 3000, 6376, .002, *meters)
        self.assertEqual(before, [meter.__dict__ for meter in meters])
        self.assertEqual(stream.getvalue(), "Lr: 2.000e-03 | Epoch: [2][3000/6376] | Loss 0.2000 (0.1500) | acc 1.200 (1.150) | IoU 2.200 (2.150)\n")
        sampler = d5.d4.EffectiveBatchSampler(19130, 2)
        self.assertEqual(len(sampler), 12752)
        self.assertEqual(sampler.size//3, 6376)

    def logger(self):
        stream = io.StringIO()
        logger = logging.Logger("d5_test")
        logger.addHandler(logging.StreamHandler(stream))
        return logger, stream

    def fake_trainer(self, data, micro, report):
        trainer = d5.Trainer.__new__(d5.Trainer)
        torch.manual_seed(2)
        trainer.model = nn.Conv2d(24, 20, 1)
        trainer.use_ema, trainer.ema_decay = True, .99
        trainer._build_ema_model()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.002)
        class Scaler:
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        trainer.scaler = Scaler()
        trainer.device = torch.device("cpu")
        trainer.ARCH = {"train":{"report_batch":3000}}
        trainer.parser = type("Parser", (), {"train_dataset":list(range(6))})()
        trainer.train_generator = torch.Generator()
        trainer.sampler = d5.d4.EffectiveBatchSampler(6, micro)
        trainer.train_loader = []
        for start in (0, 3):
            for offset in range(0, 3, micro):
                sl = slice(start+offset, start+min(3, offset+micro))
                trainer.train_loader.append(tuple(value[sl] for value in data))
        trainer.batch_input = lambda batch:batch
        # Real Trainer metrics expect the Stage 2 logits dictionary.
        original_forward = trainer.model.forward
        trainer.model.forward = lambda x:{"logits":original_forward(x)}
        trainer.full_loss = lambda outputs, labels, mask:nn.functional.cross_entropy(outputs["logits"], labels)
        trainer.logger, stream = self.logger()
        with patch.object(torch, "autocast", side_effect=lambda *a, **k:contextlib.nullcontext()):
            result = trainer.train_epoch(0, report=report)
        return trainer, result, stream.getvalue()

    def test_reporting_never_changes_updates_or_c2_stage2_metric_definitions(self):
        torch.manual_seed(3)
        data = (torch.randn(6, 24, 4, 4), torch.randint(1, 20, (6, 4, 4)), torch.ones(6, 1, 4, 4))
        for micro in (1, 2, 3):
            first, metrics1, logs = self.fake_trainer(data, micro, True)
            second, metrics2, _ = self.fake_trainer(data, micro, False)
            self.assertEqual(d5.d1.state_digest(first.model.state_dict()), d5.d1.state_digest(second.model.state_dict()))
            self.assertEqual(d5.nested_digest(first.optimizer.state_dict()), d5.nested_digest(second.optimizer.state_dict()))
            self.assertEqual(d5.d1.state_digest(first.ema_model.state_dict()), d5.d1.state_digest(second.ema_model.state_dict()))
            self.assertEqual(metrics1["optimizer_updates"], 2)
            self.assertEqual(metrics1["samples"], 6)
            for key in ("loss", "accuracy", "miou"):
                self.assertEqual(metrics1[key], metrics2[key])
            self.assertEqual(logs.count("Lr:"), 1)
            self.assertIn("Epoch: [0][0/2]", logs)
            self.assertNotIn("microbatch", logs)

    def result(self, stage1=.8, stage2=.2):
        return dict(stage1=dict(miou=stage1, accuracy=.9, per_class_iou={n:.1 for n in CLASS_NAMES[1:]}),
                    stage2=dict(miou=stage2, accuracy=.85, loss=.5, per_class_iou={n:.2 for n in CLASS_NAMES[1:]}),
                    stage2_minus_stage1=stage2-stage1,
                    alpha_statistics={"observed":dict(count=10, mean=1.2, std=.1, min=.6, max=1.6)})

    def test_validation_logs_both_stages_all_19_classes_alpha_and_best(self):
        logger, stream = self.logger()
        stop = d5.EarlyStopping()
        stop.update(.2, 0)
        d5.log_validation_summary(logger, 0, .0002, self.result(), .2, 0, stop)
        text = stream.getvalue()
        self.assertEqual(text.count("Stage 1 | IoU class"), 19)
        self.assertEqual(text.count("Stage 2 | IoU class"), 19)
        for word in ("Stage 1 mIoU", "Stage 1 Accuracy", "Stage 2 mIoU", "Stage 2 Accuracy", "Stage 2 Loss",
                     "Stage 2 - Stage 1", "Best Validation mIoU", "Best Epoch", "Early stopping", "Learning Rate", "Soft alpha (observed)"):
            self.assertIn(word, text)
        self.assertIn("-0.600000000", text)

    def test_training_best_selection_uses_only_stage2_and_saves_epoch_csv(self):
        trainer = d5.Trainer.__new__(d5.Trainer)
        trainer.initialization_verified = True
        trainer.next_epoch, trainer.best_val_epoch, trainer.best_val_iou = 0, -1, -1
        trainer.ARCH = {"train":{"max_epochs":2}}
        trainer.early_stopping = d5.EarlyStopping()
        trainer.history, trainer.total_train_seconds = [], 0.
        trainer.model, trainer.ema_model = nn.Linear(2, 2), nn.Linear(2, 2)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.002)
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch:1)
        def train(epoch):
            trainer.optimizer.step()
            return dict(loss=1., accuracy=.5, miou=.2)
        trainer.train_epoch = train
        trainer.validate = lambda epoch:self.result(.8+epoch*.1, .2-epoch*.01)
        trainer.checkpoint = lambda epoch, is_best=False:dict(epoch=epoch, best=is_best)
        trainer.logger, stream = self.logger()
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.cuda, "reset_peak_memory_stats"), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=2**30), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=2**31):
            trainer.output = Path(tmp)
            trainer.train()
            best = torch.load(trainer.output/"best_val.path", weights_only=False)
            self.assertEqual(best["epoch"], 0)
            self.assertEqual(trainer.best_val_epoch, 0)
            self.assertEqual(len((trainer.output/"validation_history.csv").read_text().splitlines()), 3)
            self.assertTrue((trainer.output/"alpha_statistics.csv").is_file())
            self.assertIn("Best mean IoU on validation so far, saving!", stream.getvalue())

    def test_file_logging_is_timestamped_append_and_protects_old_experiments(self):
        with self.assertRaises(ValueError):
            entry.check_output(entry.arguments(["--dataset", "SemanticKitti", "--log", "logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4"]))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for message in ("first run", "resume"):
                logger = entry.configure_logging(output)
                logger.info(message)
            text = (output/"train_log.txt").read_text()
            self.assertIn("first run", text)
            self.assertIn("resume", text)
            self.assertRegex(text.splitlines()[0], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO first run$")
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
            logging.getLogger("py.warnings").handlers.clear()
            logging.captureWarnings(False)

    def test_automatic_best_comparison_and_single_model_inference_package(self):
        trainer = d5.Trainer.__new__(d5.Trainer)
        trainer.ARCH = yaml.safe_load(entry.DEFAULT_ARCH.read_text())
        trainer.DATA, trainer.metadata_id, trainer.initial_hash = {}, "id", "random-hash"
        trainer.metadata = {"protected_files":{}, "accumulation_caveat":"recorded microbatch caveat"}
        trainer.ema_model = nn.Linear(2, 2)
        trainer.early_stopping = d5.EarlyStopping()
        trainer.total_train_seconds = 100.
        trainer.reference_run = json.loads((entry.ROOT/"logs/SJNet3_PointNet_stepD2_class_fixed_alpha/results.json").read_text())
        trainer.reference = trainer.reference_run["results"]["D2_Soft"]
        trainer.logger, _ = self.logger()
        best = self.result(.50, .51)
        best["seconds"] = 10.
        trainer.validate = lambda epoch, tag=None:best
        with tempfile.TemporaryDirectory() as tmp:
            trainer.output = Path(tmp)
            state = dict(experiment="Step D.5", arch_cfg=trainer.ARCH, data_cfg={}, metadata_id="id",
                alpha_table=list(ALPHA_TABLE), copy_paste_prob=0, pretrained_initialization=False,
                initialization_verified=True, model_ema=trainer.ema_model.state_dict(), best_val_epoch=2, best_val_iou=.51)
            d5.d4.atomic_checkpoint(trainer.output/"best_val.path", state)
            result = trainer.finalize()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(len((trainer.output/"best_class_comparison.csv").read_text().splitlines()), 20)
            self.assertIn("D5_Stage1", result["comparisons"])
            package = torch.load(trainer.output/"inference_package.path", weights_only=False)
            self.assertEqual(package["alpha_table"], list(ALPHA_TABLE))
            self.assertNotIn("teacher", package)
            self.assertIn("Stage 2", (trainer.output/"comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
