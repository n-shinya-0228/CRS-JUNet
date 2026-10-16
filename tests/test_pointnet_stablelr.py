"""Step C.1b scheduler, diagnostics and unchanged-learning-contract checks."""

import ast
import copy
import inspect
import math
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn
import yaml

import train_pointnet_hybrid_gated_stablelr
import trainer_PointNet_Gated_StableLR as stable
import trainer_PointNet_Gated as previous

ROOT = Path(__file__).resolve().parents[1]


def config():
    with (ROOT / "config/arch/sjnet_pointnet_gated70_stablelr.yaml").open() as f:
        return yaml.safe_load(f)


class StableLRTests(unittest.TestCase):
    def test_scheduler_values_and_no_constant_phase(self):
        train = config()["train"]
        rates = stable.scheduler_preview(train, through_epoch=70)
        expected = [2e-4, 5.6e-4, 9.2e-4, 1.28e-3, 1.64e-3, 2e-3]
        for epoch, rate in enumerate(expected):
            self.assertAlmostEqual(rates[epoch], rate, places=12)
        for epoch in range(6, 71):
            expected = 1e-5 + (2e-3 - 1e-5) * (1 + math.cos(math.pi * (epoch - 5) / 65)) / 2
            self.assertAlmostEqual(rates[epoch], expected, places=12)
            self.assertLess(rates[epoch], rates[epoch - 1])
        self.assertAlmostEqual(rates[70], 1e-5)
        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=2e-3)
        scheduler = stable.build_scheduler(optimizer, train)
        self.assertEqual(len(scheduler._schedulers), 2)
        self.assertEqual(scheduler._milestones, [5])

    def test_only_allowed_config_and_model_changes(self):
        current = config()
        current.pop("experiment")
        with (ROOT / "config/arch/sjnet_pointnet_gated70.yaml").open() as f:
            old = yaml.safe_load(f)
        for key in ("lr", "min_lr", "wup_epochs", "cos_epochs"):
            current["train"][key] = old["train"][key]
        current["train"]["early_stopping"]["patience"] = 10
        self.assertEqual(current, old)
        self.assertIs(stable.SJNet3_PointNet_Gated, previous.SJNet3_PointNet_Gated)
        self.assertIs(stable.Parser, previous.Parser)
        for name in ("_mix_losses", "_focal_loss_map", "_compute_boundary_gt", "_build_ema_model", "_update_ema"):
            self.assertEqual(inspect.getsource(getattr(stable.Trainer, name)),
                             inspect.getsource(getattr(previous.Trainer, name)))
        self.assertEqual(inspect.getsource(stable.FocalLoss), inspect.getsource(previous.FocalLoss))

        class WithoutLogging(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and ast.unparse(node.value.func).startswith("self.logger."):
                    return None
                return self.generic_visit(node)

        def training_ast(cls):
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls.train_epoch)))
            return ast.dump(WithoutLogging().visit(tree), include_attributes=False)

        self.assertEqual(training_ast(stable.Trainer), training_ast(previous.Trainer))

    def test_gate_hook_is_read_only_and_removed(self):
        model = nn.Module()
        model.point_gate = nn.Sequential(nn.Conv2d(32, 16, 1), nn.Sigmoid())
        x = torch.randn(2, 32, 8, 8)
        reference = model.point_gate(x)
        rng = torch.get_rng_state().clone()
        with stable.fusion_gate_statistics(model) as stats:
            actual = model.point_gate(x)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(set(stats), {"mean", "std", "min", "max"})
        self.assertAlmostEqual(stats["std"], reference.std(unbiased=False).item(), places=6)
        self.assertEqual(len(model.point_gate._forward_hooks), 0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(model.point_gate[0].weight.grad).all())
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with stable.fusion_gate_statistics(model):
                raise RuntimeError("interrupted")
        self.assertEqual(len(model.point_gate._forward_hooks), 0)

    def test_early_stopping_waits_until_epoch_30_and_12_bad_epochs(self):
        early = stable.EarlyStopping(**config()["train"]["early_stopping"])
        for epoch in range(30):
            self.assertFalse(early.update(0.4, epoch))
        self.assertTrue(early.update(0.4, 30))
        early = stable.EarlyStopping()
        self.assertFalse(early.update(0.4, 30))
        for epoch in range(31, 42):
            self.assertFalse(early.update(0.4, epoch))
        restored = stable.EarlyStopping()
        restored.load_state_dict(early.state_dict())
        self.assertTrue(restored.update(0.4, 42))
        self.assertFalse(restored.update(0.4 + 0.001, 43))

    def make_trainer(self, directory, resume=None):
        parser = MagicMock()
        parser.get_n_classes.return_value = 2
        parser.to_xentropy.side_effect = lambda label: label
        parser.train_dataset.copy_paste_prob = 0.0
        parser.get_train_set.return_value.batch_size = 3
        parser.get_train_size.return_value = 6376
        data = {"content": {0: 0.1, 1: 0.9}, "learning_ignore": {0: True, 1: False}}
        with patch.object(stable, "Parser", return_value=parser), \
             patch.object(stable, "SJNet3_PointNet_Gated", side_effect=lambda **kwargs: nn.Linear(2, 2)), \
             patch.object(stable, "set_tensorboard", return_value=MagicMock()), \
             patch.object(torch.cuda, "is_available", return_value=False):
            return stable.Trainer(config(), data, "unused", directory, MagicMock(), resume=resume)

    def test_actual_loop_lr_logging_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)
            rates = []

            def epoch(**kwargs):
                rates.append(trainer.optimizer.param_groups[0]["lr"])
                trainer.optimizer.zero_grad()
                trainer.model(torch.ones(1, 2)).sum().backward()
                trainer.optimizer.step()
                return 0.8, 0.7, 0.3, 0.0

            trainer.train_epoch = MagicMock(side_effect=epoch)
            trainer.validate = MagicMock(side_effect=[(0.8, 0.2 + i * 0.005, 0.5) for i in range(70)])
            trainer.train()
            self.assertEqual(rates, stable.scheduler_preview(config()["train"], 69))
            logs = [call.args[0] for call in trainer.logger.info.call_args_list]
            self.assertEqual(sum(line.startswith("Epoch ") for line in logs), 70)
            self.assertTrue(any("best epoch=69" in line for line in logs))
            restored = self.make_trainer(tmp, str(Path(tmp) / "latest.path"))
            self.assertEqual(restored.start_epoch, 70)
            self.assertEqual(restored.scheduler.state_dict(), trainer.scheduler.state_dict())
            self.assertEqual(restored.early_stopping.state_dict(), trainer.early_stopping.state_dict())
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], trainer.optimizer.param_groups[0]["lr"])
            for stage in (2, 5, 10):
                optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=0.002)
                scheduler = stable.build_scheduler(optimizer, config()["train"])
                for _ in range(stage):
                    optimizer.step()
                    scheduler.step()
                state = copy.deepcopy(scheduler.state_dict())
                optim_state = copy.deepcopy(optimizer.state_dict())
                restored_opt = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=0.002)
                restored_sch = stable.build_scheduler(restored_opt, config()["train"])
                restored_opt.load_state_dict(optim_state)
                restored_sch.load_state_dict(state)
                optimizer.step()
                scheduler.step()
                restored_opt.step()
                restored_sch.step()
                self.assertEqual(restored_opt.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])

    def test_rejects_previous_experiment_and_constant_phase(self):
        cfg = config()
        stable.validate_config(cfg)
        cfg["train"]["cos_epochs"] = 64
        with self.assertRaises(ValueError):
            stable.validate_config(cfg)
        with self.assertRaises(ValueError):
            stable.validate_resume_checkpoint({"model_name": "SJNet3_PointNet_Gated"}, config(), {})


if __name__ == "__main__":
    unittest.main()
