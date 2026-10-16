"""Step C.3: single-scalar ablation, diagnostics and resume regression tests."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
import yaml

import train_pointnet_hybrid_gated_local3_alpha as entry
import train_pointnet_hybrid_gated_local3 as c2_entry
import trainer_PointNet_Gated_StableLR as training
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3, PointNetLocal3
from lib.models.SJNet3_PointNet_Gated_Local3_Alpha import SJNet3_PointNet_Gated_Local3_Alpha, PointNetLocal3Alpha
from lib.utils.pointnet_alpha_diagnostics import build_alpha_optimizer, local_alpha_statistics

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def config():
    with (ROOT / "config/arch/sjnet_pointnet_gated_local3_alpha_stablelr.yaml").open() as f:
        return yaml.safe_load(f)


class AlphaModelTests(unittest.TestCase):
    def setUp(self):
        options = dict(nclasses=4, base_ch=8, aspp_out=16, drop=0.0)
        torch.manual_seed(0)
        self.c2 = SJNet3_PointNet_Gated_Local3(**options).eval()
        self.rng_c2 = torch.get_rng_state().clone()
        torch.manual_seed(0)
        self.c3 = SJNet3_PointNet_Gated_Local3_Alpha(**options).eval()
        self.rng_c3 = torch.get_rng_state().clone()
        self.x = torch.randn(2, 24, 32, 320)
        self.x[:, 23:24] = (torch.rand(2, 1, 32, 320) > 0.5).float()
        self.x[:, 7:23] *= self.x[:, 23:24]

    def test_one_scalar_identical_weights_rng_and_full_forward(self):
        torch.testing.assert_close(self.rng_c3, self.rng_c2, rtol=0, atol=0)
        self.assertIs(self.c3.forward.__func__, self.c2.forward.__func__)
        c2, c3 = self.c2.state_dict(), self.c3.state_dict()
        self.assertEqual(set(c3) - set(c2), {"point_context.local_alpha"})
        self.assertEqual(self.c3.local_alpha.shape, torch.Size([]))
        self.assertEqual(self.c3.local_alpha.dtype, torch.float32)
        self.assertEqual(self.c3.local_alpha.item(), 1.0)
        self.assertEqual(sum(p.numel() for p in self.c3.parameters()) -
                         sum(p.numel() for p in self.c2.parameters()), 1)
        for name, value in c2.items():
            torch.testing.assert_close(c3[name], value, atol=0, rtol=0)
        for mask in (None, 0.0, 1.0):
            x = self.x.clone()
            if mask is not None:
                x[:, 23:24] = mask
            with torch.no_grad():
                actual, expected = self.c3(x), self.c2(x)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def test_unconstrained_residual_gate_fusion_and_finite_alpha_gradient(self):
        seen = {}

        def output(name):
            def hook(module, inputs, value):
                seen[name] = value.detach().clone()
            return hook

        def input_value(name):
            def hook(module, inputs):
                seen[name] = inputs[0].detach().clone()
            return hook

        handles = [self.c3.hand_stem.register_forward_hook(output("hand")),
                   self.c3.point_stem.register_forward_hook(output("base")),
                   self.c3.point_context.local.register_forward_hook(output("residual")),
                   self.c3.point_context.register_forward_hook(output("local")),
                   self.c3.point_gate.register_forward_pre_hook(input_value("gate_input")),
                   self.c3.point_gate.register_forward_hook(output("gate")),
                   self.c3.stem_feat.register_forward_pre_hook(input_value("fusion")),
                   self.c3.stem_mask.register_forward_pre_hook(input_value("mask"))]
        try:
            for alpha in (-5.0, 0.0, 0.7, 1.0, 10.0):
                with torch.no_grad():
                    self.c3.local_alpha.fill_(alpha)
                    self.c3(self.x)
                torch.testing.assert_close(seen["local"], seen["base"] + alpha * seen["residual"])
                torch.testing.assert_close(seen["gate_input"], torch.cat([seen["hand"], seen["local"]], 1))
                torch.testing.assert_close(seen["fusion"],
                                           torch.cat([seen["hand"], seen["local"] * seen["gate"]], 1))
                torch.testing.assert_close(seen["mask"], self.x[:, 23:24], atol=0, rtol=0)
            with torch.no_grad():
                self.c3.local_alpha.fill_(1.0)
            sum(v.square().mean() for v in self.c3(self.x).values()).backward()
            grad = self.c3.local_alpha.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad))
            self.assertNotEqual(grad.item(), 0.0)
        finally:
            for handle in handles:
                handle.remove()

    def test_diagnostics_exact_read_only_and_hook_cleanup(self):
        seen = {}

        def save_residual(module, inputs, output):
            seen["residual"] = output.detach()
            seen["base"] = inputs[0].detach()

        handle = self.c3.point_context.local.register_forward_hook(save_residual)
        with torch.no_grad():
            self.c3.local_alpha.fill_(-0.73)
            expected = self.c3(self.x)
            rng = torch.get_rng_state().clone()
            with local_alpha_statistics(self.c3) as stats:
                actual = self.c3(self.x)
        handle.remove()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        scaled = self.c3.local_alpha.detach() * seen["residual"]
        self.assertAlmostEqual(stats["scaled_local_residual"]["mean"], scaled.mean().item(), places=6)
        self.assertAlmostEqual(stats["scaled_local_residual"]["std"], scaled.std(unbiased=False).item(), places=6)
        ratio = scaled.abs().mean() / (seen["base"].abs().mean() + 1e-8)
        self.assertAlmostEqual(stats["local_contribution_ratio"], ratio.item(), places=6)
        with self.assertRaises(RuntimeError):
            with local_alpha_statistics(self.c3):
                raise RuntimeError("test cleanup")
        for module in self.c3.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        with local_alpha_statistics(self.c2) as stats:
            self.c2(self.x)
        self.assertEqual(stats, {})

    def test_optimizer_exact_coverage_alpha_zero_decay_and_matching_lr(self):
        train = config()["train"]
        optimizer = build_alpha_optimizer(self.c3, train)
        groups = optimizer.param_groups
        self.assertEqual([g["weight_decay"] for g in groups], [0.005, 0.0])
        self.assertEqual(len(groups[1]["params"]), 1)
        self.assertIs(groups[1]["params"][0], self.c3.local_alpha)
        grouped_ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), {id(p) for p in self.c3.parameters()})
        original = [p for p in self.c3.parameters() if p is not self.c3.local_alpha]
        self.assertEqual([id(p) for p in groups[0]["params"]], [id(p) for p in original])
        # No gradient signal must not pull alpha toward zero via AdamW decay.
        self.c3.local_alpha.grad = torch.zeros_like(self.c3.local_alpha)
        optimizer.step()
        self.assertEqual(self.c3.local_alpha.item(), 1.0)
        scheduler = training.build_scheduler(optimizer, train)
        expected_rates = training.scheduler_preview(train, 70)
        for rate in expected_rates:
            self.assertEqual([g["lr"] for g in optimizer.param_groups], [rate, rate])
            optimizer.step()
            scheduler.step()


class AlphaTrainerTests(unittest.TestCase):
    def make_trainer(self, path, resume=None):
        class TinyAlpha(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))
                self.local_alpha = nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                return self.weight * x + self.local_alpha * x

        parser = MagicMock()
        parser.get_n_classes.return_value = 2
        parser.to_xentropy.side_effect = lambda label: label
        parser.train_dataset.copy_paste_prob = 0.0
        data = {"content": {0: 0.1, 1: 0.9}, "learning_ignore": {0: True, 1: False}}
        with patch.object(training, "Parser", return_value=parser), \
             patch.object(training, "SJNet3_PointNet_Gated_Local3_Alpha", side_effect=lambda **kw: TinyAlpha()), \
             patch.object(training, "set_tensorboard", return_value=MagicMock()), \
             patch.object(torch.cuda, "is_available", return_value=False):
            return training.Trainer(config(), data, "unused", path, MagicMock(), resume=resume)

    def test_config_shared_trainer_and_reference_epoch90(self):
        self.assertIs(entry.Trainer, c2_entry.Trainer)
        with (ROOT / "config/arch/sjnet_pointnet_gated_local3_stablelr.yaml").open() as f:
            c2 = yaml.safe_load(f)
        c3 = config()
        for key in ("train", "dataset", "post"):
            self.assertEqual(c3[key], c2[key])
        model = copy.deepcopy(c3["model"])
        model["name"] = c2["model"]["name"]
        self.assertEqual(model, c2["model"])
        self.assertEqual(c3["experiment"]["baseline_8ch_best_epoch"], 90)
        self.assertEqual(c3["experiment"]["baseline_8ch_miou"], 0.507)
        self.assertEqual(c3["experiment"]["baseline_8ch_acc"], 0.891)
        self.assertEqual(c3["experiment"]["baseline_8ch_log_dir"], "logs/SJNet3_48")
        self.assertEqual(c3["experiment"]["baseline_8ch_class_iou"], {
            "unlabeled": 0.000, "car": 0.947, "bicycle": 0.247,
            "motorcycle": 0.313, "truck": 0.645, "other-vehicle": 0.306,
            "person": 0.411, "bicyclist": 0.492, "motorcyclist": 0.000,
            "road": 0.937, "parking": 0.495, "sidewalk": 0.802,
            "other-ground": 0.009, "building": 0.839, "fence": 0.272,
            "vegetation": 0.807, "trunk": 0.432, "terrain": 0.723,
            "pole": 0.487, "traffic-sign": 0.474,
        })
        self.assertEqual(training.experiment_id(c3), "Step C.3")
        training.validate_config(c3)

    def test_raw_ema_alpha_best_checkpoint_resume_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)
            trainer.model(torch.ones(1)).sum().backward()
            trainer.optimizer.step()
            trainer._update_ema()
            trainer.scheduler.step()
            trainer.best_val_epoch = 0
            trainer.early_stopping.update(0.51, 0)
            trainer.last_class_iou = {name: 0.5 for name in config()["experiment"]["step_c2_class_iou"]}
            trainer._log_local_alpha()
            trainer._save_checkpoint(0, 0.6, 0.51, is_best=True)
            trainer._save_checkpoint(0, 0.6, 0.51)
            best = torch.load(Path(tmp) / "best_val.path", weights_only=False)
            self.assertEqual(best["experiment"], "Step C.3")
            self.assertEqual(best["best_local_alpha"], best["model"]["local_alpha"].item())
            self.assertNotEqual(best["local_alpha_eval"], best["local_alpha_train"])
            self.assertEqual(best["model_train"]["local_alpha"].item(), best["local_alpha_train"])
            for filename in ("latest.path", "best_val.path"):
                resumed = self.make_trainer(tmp, str(Path(tmp) / filename))
                self.assertEqual(resumed.start_epoch, 1)
                self.assertEqual(resumed.best_local_alpha, best["best_local_alpha"])
                self.assertEqual(resumed.model.local_alpha.item(), trainer.model.local_alpha.item())
                self.assertEqual(resumed.ema_model.local_alpha.item(), trainer.ema_model.local_alpha.item())
                self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
                self.assertEqual([g["weight_decay"] for g in resumed.optimizer.param_groups], [0.005, 0.0])
                torch.testing.assert_close(resumed.optimizer.state[resumed.model.local_alpha]["exp_avg"],
                                           trainer.optimizer.state[trainer.model.local_alpha]["exp_avg"])
            logs = [call.args[0] for call in trainer.logger.info.call_args_list]
            self.assertTrue(any(line.startswith("Local alpha:") for line in logs))
            self.assertTrue(any(line.startswith("Best local alpha:") for line in logs))
            comparisons = [line for line in logs if line.startswith("Best class comparison")]
            self.assertEqual(len(comparisons), 12)
            self.assertTrue(all("N/A" not in line for line in comparisons))
            with self.assertRaises(ValueError):
                training.validate_resume_checkpoint(dict(best, experiment="Step C.2"), config(), trainer.DATA)
            with torch.no_grad():
                trainer.model.local_alpha.fill_(10.0)
            trainer._log_local_alpha()
            self.assertTrue(trainer.logger.warning.called)
            self.assertEqual(trainer.model.local_alpha.item(), 10.0)

    def test_entry_refuses_c2_config_existing_logs_and_c2_resume_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_created"
            common = ["train", "--dataset", "unused", "--data_cfg", str(ROOT / "config/labels/semantic-kitti.yaml")]
            c2_config = str(ROOT / "config/arch/sjnet_pointnet_gated_local3_stablelr.yaml")
            c3_config = str(ROOT / "config/arch/sjnet_pointnet_gated_local3_alpha_stablelr.yaml")
            with patch("sys.argv", common + ["--arch_cfg", c2_config, "--log", str(path)]), \
                 self.assertRaises(SystemExit):
                entry.main()
            self.assertFalse(path.exists())
            checkpoint = Path(tmp) / "c2.path"
            torch.save({"experiment": "Step C.2"}, checkpoint)
            with patch("sys.argv", common + ["--arch_cfg", c3_config, "--log", str(path), "--resume", str(checkpoint)]), \
                 self.assertRaises(SystemExit):
                entry.main()
            self.assertFalse(path.exists())
            path.mkdir()
            with patch("sys.argv", common + ["--arch_cfg", c3_config, "--log", str(path)]), \
                 self.assertRaises(SystemExit):
                entry.main()
            self.assertFalse(list(path.iterdir()))


if __name__ == "__main__":
    unittest.main()
