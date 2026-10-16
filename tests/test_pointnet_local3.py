"""Step C.2 architecture, diagnostics and shared-trainer regression tests."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn
import yaml

import train_pointnet_hybrid_gated_local3 as entry
import train_pointnet_hybrid_gated_stablelr as baseline_entry
import trainer_PointNet_Gated_StableLR as training
from lib.models.SJNet3_PointNet import RingConv2d
from lib.models.SJNet3_PointNet_Gated import SJNet3_PointNet_Gated
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.utils.pointnet_local3_diagnostics import local3_feature_statistics

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def config():
    with (ROOT / "config/arch/sjnet_pointnet_gated_local3_stablelr.yaml").open() as f:
        return yaml.safe_load(f)


class Local3ModelTests(unittest.TestCase):
    def setUp(self):
        options = dict(nclasses=4, base_ch=8, aspp_out=16, drop=0.0)
        torch.manual_seed(0)
        self.baseline = SJNet3_PointNet_Gated(**options).eval()
        torch.manual_seed(0)
        self.model = SJNet3_PointNet_Gated_Local3(**options).eval()
        self.x = torch.randn(2, 24, 32, 320)
        self.x[:, 23:24] = (torch.rand(2, 1, 32, 320) > 0.5).float()
        self.x[:, 7:23] *= self.x[:, 23:24]

    def test_only_2336_new_parameters_and_identical_existing_initialization(self):
        current = self.model.state_dict()
        for name, value in self.baseline.state_dict().items():
            torch.testing.assert_close(current[name], value, rtol=0, atol=0)
        extra = set(current) - set(self.baseline.state_dict())
        self.assertEqual(extra, {"point_context.local.0.conv.weight",
                                 "point_context.local.1.weight", "point_context.local.1.bias"})
        self.assertEqual(sum(p.numel() for p in self.model.parameters()) -
                         sum(p.numel() for p in self.baseline.parameters()), 2336)
        self.assertIsInstance(self.model.point_context.local[0], RingConv2d)
        self.assertIsInstance(self.model.point_context.local[1], nn.InstanceNorm2d)
        self.assertIsInstance(self.model.point_context.local[2], nn.ReLU)
        self.assertEqual(self.model.point_context.local[0].conv.bias, None)
        self.assertEqual(self.model.point_context.local[0].conv.kernel_size, (3, 3))
        self.assertFalse(any("alpha" in name for name, _ in self.model.named_parameters()))

    def test_identity_context_matches_all_c1b_outputs_and_masks(self):
        for mask in (None, 0.0, 1.0):
            x = self.x.clone()
            if mask is not None:
                x[:, 23:24] = mask
            with torch.no_grad(), patch.object(self.model, "point_context", nn.Identity()):
                actual, expected = self.model(x), self.baseline(x)
            for name in ("logits", "aux2", "aux4", "aux8", "boundary"):
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_gate_and_fusion_use_residual_sum_and_local_gradients(self):
        seen = {}

        def output(name):
            def hook(module, args, value):
                seen[name] = value.detach().clone()
            return hook

        def input_value(name):
            def hook(module, args):
                seen[name] = args[0].detach().clone()
            return hook

        handles = [self.model.hand_stem.register_forward_hook(output("hand")),
                   self.model.point_stem.register_forward_hook(output("base")),
                   self.model.point_context.local.register_forward_hook(output("residual")),
                   self.model.point_context.register_forward_hook(output("local")),
                   self.model.point_gate.register_forward_pre_hook(input_value("gate_input")),
                   self.model.point_gate.register_forward_hook(output("gate")),
                   self.model.stem_feat.register_forward_pre_hook(input_value("fusion_input")),
                   self.model.stem_mask.register_forward_pre_hook(input_value("mask"))]
        x_before = self.x.clone()
        outputs = self.model(self.x)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(seen["local"], seen["base"] + seen["residual"])
        torch.testing.assert_close(seen["gate_input"], torch.cat([seen["hand"], seen["local"]], 1))
        torch.testing.assert_close(seen["fusion_input"],
                                   torch.cat([seen["hand"], seen["local"] * seen["gate"]], 1))
        torch.testing.assert_close(seen["mask"], self.x[:, 23:24])
        torch.testing.assert_close(self.x, x_before, rtol=0, atol=0)
        sum(value.square().mean() for value in outputs.values()).backward()
        for parameter in self.model.point_context.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_diagnostics_are_read_only_and_expose_unobserved_features(self):
        with torch.no_grad():
            expected = self.model(self.x)
            rng = torch.get_rng_state().clone()
            with local3_feature_statistics(self.model) as stats:
                actual = self.model(self.x)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        self.assertEqual(stats["point_input_unobserved"]["nonzero_fraction"], 0)
        self.assertGreater(stats["point_local_unobserved"]["nonzero_fraction"], 0)
        self.assertTrue(all(value["finite"] for value in stats.values()))
        with self.assertRaises(RuntimeError):
            with local3_feature_statistics(self.model):
                raise RuntimeError("interrupted")
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        self.x[:, 23:24] = 1
        with torch.no_grad(), local3_feature_statistics(self.model) as stats:
            self.model(self.x)
        self.assertIsNone(stats["point_local_unobserved"]["mean"])

    def test_angular_wrap_without_radial_wrap_and_bad_input(self):
        conv = self.model.point_context.local[0]
        nn.init.ones_(conv.conv.weight)
        x = torch.zeros(1, 16, 5, 5)
        x[0, 0, 0, 0] = 1
        value = conv(x)
        self.assertEqual(value[0, 0, 0, -1].item(), 1)
        self.assertEqual(value[0, 0, -1, 0].item(), 0)
        with self.assertRaises(AssertionError):
            self.model(torch.zeros(1, 23, 32, 320))


class Local3TrainerTests(unittest.TestCase):
    def test_shared_trainer_and_unchanged_training_config(self):
        self.assertIs(entry.Trainer, baseline_entry.Trainer)
        with (ROOT / "config/arch/sjnet_pointnet_gated70_stablelr.yaml").open() as f:
            base = yaml.safe_load(f)
        local = config()
        for key in ("train", "dataset", "post"):
            self.assertEqual(local[key], base[key])
        local_model = copy.deepcopy(local["model"])
        local_model["name"] = base["model"]["name"]
        self.assertEqual(local_model, base["model"])
        self.assertEqual(len(local["experiment"]["step_c1b_class_iou"]), 12)
        self.assertEqual(training.scheduler_preview(local["train"], 70),
                         training.scheduler_preview(base["train"], 70))
        training.validate_config(local)

    def make_trainer(self, path, resume=None):
        class TinyLocal(nn.Module):
            def __init__(self):
                super().__init__()
                self.point_context = nn.Linear(2, 2)

            def forward(self, x):
                return self.point_context(x)

        parser = MagicMock()
        parser.get_n_classes.return_value = 2
        parser.to_xentropy.side_effect = lambda label: label
        parser.train_dataset.copy_paste_prob = 0.0
        data = {"content": {0: 0.1, 1: 0.9}, "learning_ignore": {0: True, 1: False}}
        with patch.object(training, "Parser", return_value=parser), \
             patch.object(training, "SJNet3_PointNet_Gated_Local3", side_effect=lambda **kwargs: TinyLocal()), \
             patch.object(training, "set_tensorboard", return_value=MagicMock()), \
             patch.object(torch.cuda, "is_available", return_value=False):
            return training.Trainer(config(), data, "unused", path, MagicMock(), resume=resume)

    def test_early_stopping_deferred_logs_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)

            def epoch(**kwargs):
                trainer.optimizer.zero_grad()
                trainer.model(torch.ones(1, 2)).sum().backward()
                trainer.optimizer.step()
                return 0.8, 0.7, 0.3, 0.0

            trainer.train_epoch = MagicMock(side_effect=epoch)
            trainer.validate = MagicMock(return_value=(0.8, 0.4, 0.5))
            trainer.train()
            self.assertEqual(trainer.train_epoch.call_count, 31)
            logs = [call.args[0] for call in trainer.logger.info.call_args_list]
            self.assertTrue(any("Early stopping deferred:" in line and "epoch 29" in line for line in logs))
            path = Path(tmp) / "latest.path"
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["experiment"], "Step C.2")
            self.assertEqual(checkpoint["model_name"], "SJNet3_PointNet_Gated_Local3")
            self.assertEqual(checkpoint["epoch"], 30)
            resumed = self.make_trainer(tmp, str(path))
            self.assertEqual(resumed.start_epoch, 31)
            self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
            self.assertEqual(resumed.early_stopping.state_dict(), trainer.early_stopping.state_dict())
            bad = dict(checkpoint, experiment="Step C.1b")
            with self.assertRaises(ValueError):
                training.validate_resume_checkpoint(bad, config(), trainer.DATA)


if __name__ == "__main__":
    unittest.main()
