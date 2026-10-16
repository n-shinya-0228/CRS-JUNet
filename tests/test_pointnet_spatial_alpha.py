"""C.4 architecture ablation, label-free forward, streaming stats and persistence."""

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
import yaml

import train_pointnet_hybrid_gated_local3_spatial_alpha as entry
import train_pointnet_hybrid_gated_local3_alpha as c3_entry
import trainer_PointNet_Gated_StableLR as training
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_Alpha import SJNet3_PointNet_Gated_Local3_Alpha
from lib.models.SJNet3_PointNet_Gated_Local3_SpatialAlpha import (
    SJNet3_PointNet_Gated_Local3_SpatialAlpha, PointNetLocal3SpatialAlpha, SpatialAlpha,
)
from lib.utils.pointnet_spatial_alpha_diagnostics import (
    build_spatial_alpha_optimizer, spatial_alpha_capture, SpatialAlphaValidation,
    RunningMoments, log_spatial_validation,
)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def config():
    with (ROOT / "config/arch/sjnet_pointnet_gated_local3_spatial_alpha_stablelr.yaml").open() as f:
        return yaml.safe_load(f)


class SpatialModelTests(unittest.TestCase):
    def setUp(self):
        options = dict(nclasses=4, base_ch=8, aspp_out=16, drop=0.0)
        torch.manual_seed(0)
        self.c2 = SJNet3_PointNet_Gated_Local3(**options).eval()
        c2_rng = torch.get_rng_state().clone()
        torch.manual_seed(0)
        self.c4 = SJNet3_PointNet_Gated_Local3_SpatialAlpha(**options).eval()
        torch.testing.assert_close(c2_rng, torch.get_rng_state(), atol=0, rtol=0)
        self.x = torch.randn(2, 24, 32, 320)
        self.x[:, 23:24] = (torch.rand(2, 1, 32, 320) > .5).float()
        self.x[:, 7:23] *= self.x[:, 23:24]

    def test_a_seventeen_parameters_shared_weights_and_full_c2_equivalence(self):
        c2, c4 = self.c2.state_dict(), self.c4.state_dict()
        self.assertEqual(set(c4) - set(c2), {"point_context.alpha.alpha_global",
                                          "point_context.alpha.alpha_predictor.weight"})
        self.assertEqual(sum(p.numel() for p in self.c4.parameters()) -
                         sum(p.numel() for p in self.c2.parameters()), 17)
        self.assertIs(self.c4.forward.__func__, self.c2.forward.__func__)
        self.assertEqual(self.c4.alpha_global.shape, torch.Size([]))
        self.assertEqual(self.c4.alpha_global.item(), 1.0)
        predictor = self.c4.alpha_predictor
        self.assertEqual(predictor.weight.shape, (1, 16, 1, 1))
        self.assertIsNone(predictor.bias)
        self.assertEqual(predictor.weight.count_nonzero().item(), 0)
        for name in c2:
            torch.testing.assert_close(c4[name], c2[name], atol=0, rtol=0)
        for mask in (None, 0.0, 1.0):
            x = self.x.clone()
            if mask is not None:
                x[:, 23:24] = mask
            with torch.no_grad(), spatial_alpha_capture(self.c4, detailed=True) as stats:
                actual, expected = self.c4(x), self.c2(x)
            self.assertTrue(torch.equal(stats["alpha_map"], torch.ones_like(stats["alpha_map"])))
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def test_b_c3_equivalence_and_c_spatial_variation_without_constraints(self):
        torch.manual_seed(0)
        c3 = SJNet3_PointNet_Gated_Local3_Alpha(nclasses=4, base_ch=8, aspp_out=16, drop=0).eval()
        with torch.no_grad():
            c3.local_alpha.fill_(.641107)
            self.c4.alpha_global.fill_(.641107)
            expected, actual = c3(self.x), self.c4(self.x)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        block = self.c4.point_context
        base = torch.arange(16 * 4 * 4, dtype=torch.float32).reshape(1, 16, 4, 4)
        with torch.no_grad():
            block.alpha.alpha_predictor.weight.fill_(-.1)
            delta = block.alpha.alpha_predictor(base)
            alpha = block.alpha(base)
            torch.testing.assert_close(alpha, .641107 + delta)
            self.assertGreater(alpha.std().item(), 0)
            self.assertLess(alpha.min().item(), -5)
            torch.testing.assert_close(block(base), base + alpha * block.local(base))

    def test_d_gradients_optimizer_membership_updates_and_identical_lr(self):
        optimizer = build_spatial_alpha_optimizer(self.c4, config()["train"])
        groups = optimizer.param_groups
        self.assertEqual([g["weight_decay"] for g in groups], [.005, 0])
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in self.c4.parameters()})
        self.assertEqual({id(p) for p in groups[1]["params"]},
                         {id(self.c4.alpha_global), id(self.c4.alpha_predictor.weight)})
        outputs = self.c4(self.x)
        sum(value.square().mean() for value in outputs.values()).backward()
        before = [p.detach().clone() for p in groups[1]["params"]]
        for p in groups[1]["params"]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        optimizer.step()
        for previous, p in zip(before, groups[1]["params"]):
            self.assertFalse(torch.equal(previous, p))
        optimizer.zero_grad(set_to_none=True)
        scheduler = training.build_scheduler(optimizer, config()["train"])
        for rate in training.scheduler_preview(config()["train"], 70):
            self.assertEqual([g["lr"] for g in groups], [rate, rate])
            optimizer.step()
            scheduler.step()

    def test_gate_fusion_masks_and_label_free_read_only_diagnostics(self):
        self.assertEqual(list(inspect.signature(self.c4.forward).parameters), ["x"])
        seen = {}

        def output(name):
            def hook(module, inputs, value):
                seen[name] = value.detach().clone()
            return hook

        def input_value(name):
            def hook(module, inputs):
                seen[name] = inputs[0].detach().clone()
            return hook

        handles = [self.c4.hand_stem.register_forward_hook(output("hand")),
                   self.c4.point_stem.register_forward_hook(output("base")),
                   self.c4.point_context.local.register_forward_hook(output("residual")),
                   self.c4.point_context.register_forward_hook(output("local")),
                   self.c4.point_gate.register_forward_pre_hook(input_value("gate_input")),
                   self.c4.point_gate.register_forward_hook(output("gate")),
                   self.c4.stem_feat.register_forward_pre_hook(input_value("fusion")),
                   self.c4.stem_mask.register_forward_pre_hook(input_value("mask"))]
        try:
            with torch.no_grad():
                self.c4.alpha_predictor.weight.fill_(.03)
                original = self.x.clone()
                expected = self.c4(self.x)
                rng = torch.get_rng_state().clone()
                with spatial_alpha_capture(self.c4, detailed=True) as stats:
                    actual = self.c4(self.x)
            alpha = stats["alpha_map"]
            torch.testing.assert_close(seen["local"], seen["base"] + alpha * seen["residual"])
            torch.testing.assert_close(seen["gate_input"], torch.cat([seen["hand"], seen["local"]], 1))
            torch.testing.assert_close(seen["fusion"],
                                       torch.cat([seen["hand"], seen["local"] * seen["gate"]], 1))
            torch.testing.assert_close(seen["mask"], self.x[:, 23:24], atol=0, rtol=0)
            torch.testing.assert_close(original, self.x, atol=0, rtol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            ratio = (alpha * seen["residual"]).abs().double().mean() / (seen["base"].abs().double().mean() + 1e-8)
            self.assertAlmostEqual(stats["local_contribution_ratio"], ratio.item(), places=7)
            self.assertFalse(alpha.requires_grad)
            self.assertEqual(alpha.device.type, "cpu")
            summaries = []
            for cid in (1, 2):
                collector = SpatialAlphaValidation(self.c4, {1: "car", 2: "truck"})
                collector.update(dict(stats), torch.full((2, 32, 320), cid), self.x[:, 23:24])
                summaries.append(collector.summary())
            self.assertGreater(summaries[0]["class_alpha"]["car"]["count"], 0)
            self.assertEqual(summaries[1]["class_alpha"]["car"]["count"], 0)
        finally:
            for handle in handles:
                handle.remove()
        with self.assertRaises(RuntimeError):
            with spatial_alpha_capture(self.c4, detailed=True):
                raise RuntimeError("interruption")
        for module in self.c4.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        with torch.no_grad(), spatial_alpha_capture(self.c2) as stats:
            self.c2(self.x)
        self.assertEqual(stats, {})


class StatisticsTests(unittest.TestCase):
    def model(self):
        model = nn.Module()
        model.point_context = PointNetLocal3SpatialAlpha(nn.Identity())
        return model

    def test_online_class_filtering_population_std_and_empty_classes(self):
        collector = SpatialAlphaValidation(self.model(), {1: "car", 2: "truck", 3: "person"}, min_class_cells=3)
        values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
        labels = torch.tensor([[[1, 2, 0, 255], [1, 2, 2, -1]]])
        mask = torch.tensor([[[[1, 1, 1, 1], [1, 0, 1, 1]]]])
        for offset in (0, 10):
            capture = dict(alpha_map=values + offset)
            collector.update(capture, labels, mask)
            self.assertNotIn("alpha_map", capture)
        summary = collector.summary()
        selected = torch.tensor([0., 4., 10., 14.]).double()
        stats = summary["class_alpha"]["car"]
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["mean"], selected.mean().item())
        self.assertAlmostEqual(stats["std"], selected.std(unbiased=False).item())
        self.assertEqual(summary["class_alpha"]["truck"]["count"], 4)
        self.assertEqual(summary["class_alpha"]["person"]["count"], 0)
        self.assertIsNone(summary["class_alpha"]["person"]["mean"])
        self.assertEqual(summary["observed"]["count"], 14)
        self.assertEqual(summary["unobserved"]["count"], 2)
        self.assertEqual(summary["alpha_map"]["count"], 16)
        quantiles = torch.quantile(values.flatten(), torch.tensor([.05, .5, .95]))
        for name, expected in zip(("p05", "p50", "p95"), quantiles):
            self.assertEqual(summary["batch0"]["alpha_map"][name], expected.item())
        self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(collector).values()))
        # Fewer than the declared reporting threshold yields N/A, not a noisy estimate.
        small = SpatialAlphaValidation(self.model(), {1: "car"}, min_class_cells=5)
        small.update(dict(alpha_map=values), labels, mask)
        self.assertEqual(small.summary()["class_alpha"]["car"]["count"], 2)
        self.assertIsNone(small.summary()["class_alpha"]["car"]["mean"])

    def test_fractions_nonfinite_empty_observations_and_warning(self):
        moments = RunningMoments()
        moments.update(torch.tensor([-2., 0., 3., 4.]))
        self.assertEqual(moments.summary()["negative_fraction"], .25)
        self.assertEqual(moments.summary()["gt3_fraction"], .25)
        moments.update(torch.tensor([float("nan"), float("inf")]))
        self.assertFalse(moments.summary()["finite"])
        self.assertEqual(moments.summary()["finite_count"], 4)
        self.assertIsNone(moments.summary()["mean"])
        collector = SpatialAlphaValidation(self.model(), {1: "car"})
        collector.update(dict(alpha_map=torch.tensor([[[[-6., 11.]]]])),
                         torch.ones(1, 1, 2).long(), torch.zeros(1, 1, 1, 2))
        summary = collector.summary()
        self.assertIsNone(summary["observed"]["mean"])
        logger = MagicMock()
        log_spatial_validation(logger, MagicMock(), summary, 0)
        self.assertTrue(logger.warning.called)
        self.assertTrue(any("mean=N/A" in call.args[0] for call in logger.info.call_args_list))


class SpatialTrainerTests(unittest.TestCase):
    def make_trainer(self, path, resume=None):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.point_context = PointNetLocal3SpatialAlpha(nn.Conv2d(16, 16, 1, bias=False))

            def forward(self, x):
                return self.point_context(x)

        parser = MagicMock()
        parser.get_n_classes.return_value = 2
        parser.to_xentropy.side_effect = lambda label: label
        parser.train_dataset.copy_paste_prob = 0.0
        data = {"content": {0: .1, 1: .9}, "learning_ignore": {0: True, 1: False}}
        with patch.object(training, "Parser", return_value=parser), \
             patch.object(training, "SJNet3_PointNet_Gated_Local3_SpatialAlpha", side_effect=lambda **kw: Tiny()), \
             patch.object(training, "set_tensorboard", return_value=MagicMock()), \
             patch.object(torch.cuda, "is_available", return_value=False):
            return training.Trainer(config(), data, "unused", path, MagicMock(), resume=resume)

    def test_config_comparison_references_and_shared_trainer(self):
        self.assertIs(entry.Trainer, c3_entry.Trainer)
        with (ROOT / "config/arch/sjnet_pointnet_gated_local3_alpha_stablelr.yaml").open() as f:
            c3 = yaml.safe_load(f)
        c4 = config()
        for key in ("train", "dataset", "post"):
            self.assertEqual(c4[key], c3[key])
        changed = copy.deepcopy(c4["model"])
        changed["name"] = c3["model"]["name"]
        self.assertEqual(changed, c3["model"])
        self.assertEqual(c4["experiment"]["baseline_8ch_class_iou"], c3["experiment"]["baseline_8ch_class_iou"])
        for key in ("step_c1b_class_iou", "step_c2_class_iou", "step_c3_class_iou"):
            self.assertEqual(len(c4["experiment"][key]), 20)
        self.assertEqual(c4["experiment"]["step_c3_class_iou"]["truck"], .533154)
        self.assertEqual(training.experiment_id(c4), "Step C.4")
        training.validate_config(c4)

    def test_best_latest_raw_ema_optimizer_resume_and_comparison_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)
            trainer.model(torch.rand(1, 16, 4, 4)).square().mean().backward()
            trainer.optimizer.step()
            trainer._update_ema()
            trainer.scheduler.step()
            trainer.best_val_epoch = 0
            trainer.early_stopping.update(.51, 0)
            names = {cid: name for cid, name in enumerate(config()["experiment"]["baseline_8ch_class_iou"]) if cid}
            stats = SpatialAlphaValidation(trainer.ema_model, names)
            with torch.no_grad(), spatial_alpha_capture(trainer.ema_model, detailed=True) as capture:
                trainer.ema_model(torch.rand(1, 16, 4, 4))
            stats.update(capture, torch.ones(1, 4, 4).long(), torch.ones(1, 1, 4, 4))
            trainer.last_validation_summary = dict(epoch=0, miou=.51, acc=.89, loss=.7,
                class_iou={name: .5 for name in names.values()}, spatial_alpha=stats.summary(), evaluated_weights="EMA")
            trainer._save_checkpoint(0, .6, .51, is_best=True)
            trainer._save_checkpoint(0, .6, .51)
            best = torch.load(Path(tmp) / "best_val.path", weights_only=False)
            self.assertEqual(best["experiment"], "Step C.4")
            key = "point_context.alpha.alpha_global"
            self.assertEqual(best["best_spatial_summary"]["spatial_alpha"]["alpha_global"], best["model"][key].item())
            self.assertNotEqual(best["model_train"][key].item(), best["model"][key].item())
            saved = json.loads((Path(tmp) / "best_summary.json").read_text())
            self.assertEqual(saved["spatial_alpha"], best["best_spatial_summary"]["spatial_alpha"])
            for filename in ("best_val.path", "latest.path"):
                resumed = self.make_trainer(tmp, str(Path(tmp) / filename))
                self.assertEqual(resumed.start_epoch, 1)
                self.assertEqual(resumed.best_spatial_summary, trainer.best_spatial_summary)
                for name, parameter in trainer.model.state_dict().items():
                    torch.testing.assert_close(resumed.model.state_dict()[name], parameter, rtol=0, atol=0)
                    torch.testing.assert_close(resumed.ema_model.state_dict()[name],
                                               trainer.ema_model.state_dict()[name], rtol=0, atol=0)
                self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
                self.assertEqual([g["weight_decay"] for g in resumed.optimizer.param_groups], [.005, 0.0])
                for p, q in zip(trainer.optimizer.param_groups[1]["params"], resumed.optimizer.param_groups[1]["params"]):
                    torch.testing.assert_close(trainer.optimizer.state[p]["exp_avg"], resumed.optimizer.state[q]["exp_avg"])
            comparisons = [call.args[0] for call in trainer.logger.info.call_args_list
                           if call.args[0].startswith("Best class comparison")]
            self.assertEqual(len(comparisons), 19)
            for line in comparisons:
                self.assertNotIn("Baseline=N/A", line)
                self.assertNotIn("C.3=N/A", line)
            for old in ("Step C.2", "Step C.3"):
                with self.assertRaises(ValueError):
                    training.validate_resume_checkpoint(dict(best, experiment=old), config(), trainer.DATA)
            # Saving a later non-best must not overwrite the best epoch's statistics.
            trainer.last_validation_summary["epoch"] = 1
            trainer._save_checkpoint(1, .6, .51)
            latest = torch.load(Path(tmp) / "latest.path", weights_only=False)
            self.assertEqual(latest["best_spatial_summary"]["epoch"], 0)
            self.assertEqual(latest["spatial_validation"]["epoch"], 1)

    def test_entry_refuses_old_experiments_and_existing_logs_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_created"
            common = ["train", "--dataset", "unused", "--data_cfg", str(ROOT / "config/labels/semantic-kitti.yaml")]
            new_cfg = str(ROOT / "config/arch/sjnet_pointnet_gated_local3_spatial_alpha_stablelr.yaml")
            old_cfg = str(ROOT / "config/arch/sjnet_pointnet_gated_local3_alpha_stablelr.yaml")
            with patch("sys.argv", common + ["--arch_cfg", old_cfg, "--log", str(path)]), self.assertRaises(SystemExit):
                entry.main()
            self.assertFalse(path.exists())
            checkpoint = Path(tmp) / "old.path"
            torch.save({"experiment": "Step C.3"}, checkpoint)
            with patch("sys.argv", common + ["--arch_cfg", new_cfg, "--log", str(path), "--resume", str(checkpoint)]), \
                    self.assertRaises(SystemExit):
                entry.main()
            self.assertFalse(path.exists())
            path.mkdir()
            with patch("sys.argv", common + ["--arch_cfg", new_cfg, "--log", str(path)]), self.assertRaises(SystemExit):
                entry.main()
            self.assertFalse(list(path.iterdir()))


if __name__ == "__main__":
    unittest.main()
