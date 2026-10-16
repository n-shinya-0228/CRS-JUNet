"""Focused Step C.1 tests; run with python -B -m unittest discover -s tests."""

import copy
import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn
import yaml

import train_pointnet_hybrid_gated as entry
import trainer_PointNet_Gated as training
from lib.dataset.Parser_PointNet import bev_collate_fn
from lib.dataset.SemanticKitti_PointNet import SemanticKitti
from lib.models.SJNet3_PointNet import RingConv2d, SJNet3_PointNet
from lib.models.SJNet3_PointNet_Gated import SJNet3_PointNet_Gated

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def arch_config():
    with (ROOT / "config/arch/sjnet_pointnet_gated70.yaml").open() as f:
        return yaml.safe_load(f)


class GatedModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SJNet3_PointNet_Gated(nclasses=4, base_ch=8, aspp_out=16, drop=0.0)
        # The original ASPP needs bottleneck width >= its maximum dilation (18).
        self.x = torch.randn(2, 24, 32, 320)
        self.x[:, 23:24] = (torch.rand(2, 1, 32, 320) > 0.5).float()

    def test_shapes_gate_and_gradients(self):
        seen = {}

        def capture_input(name):
            def hook(module, args):
                seen[name] = args[0].detach().clone()
            return hook

        hooks = [self.model.hand_stem.register_forward_pre_hook(capture_input("hand")),
                 self.model.point_stem.register_forward_pre_hook(capture_input("point")),
                 self.model.stem_mask.register_forward_pre_hook(capture_input("mask"))]
        gate_values = []
        hooks.append(self.model.point_gate.register_forward_hook(
            lambda module, args, out: gate_values.append(out.detach())))
        outputs = self.model(self.x)
        for hook in hooks:
            hook.remove()
        self.assertEqual(set(outputs), {"logits", "aux2", "aux4", "aux8", "boundary"})
        for name, value in outputs.items():
            self.assertEqual(value.shape, (2, 1 if name == "boundary" else 4, 32, 320))
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(seen["hand"], self.x[:, :7])
        torch.testing.assert_close(seen["point"], self.x[:, 7:23])
        torch.testing.assert_close(seen["mask"], self.x[:, 23:24])
        gate = gate_values[0]
        self.assertEqual(gate.shape, (2, 16, 32, 320))
        self.assertTrue(((gate > 0) & (gate < 1)).all())
        self.assertTrue(0.35 < gate.mean().item() < 0.65)
        self.assertEqual(self.model.point_gate[0].bias.count_nonzero().item(), 0)
        self.assertIsInstance(self.model.stem_feat[0], RingConv2d)
        sum(value.square().mean() for value in outputs.values()).backward()
        for weight in (self.model.hand_stem[0].weight, self.model.point_stem[0].weight,
                       self.model.point_gate[0].weight, self.model.stem_feat[0].conv.weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)
        self.assertIsNone(self.model.gate_scale.grad)  # Also unused in Step C.

    def test_mask_and_backbone_match_original_forward(self):
        model = self.model.eval()
        fusion = model.stem_feat

        class ReferenceStem(nn.Module):
            def forward(self, features):
                hand = model.hand_stem(features[:, :7])
                point = model.point_stem(features[:, 7:23])
                gate = model.point_gate(torch.cat([hand, point], dim=1))
                return fusion(torch.cat([hand, point * gate], dim=1))

        for mask in (0.0, 1.0, None):
            x = self.x.clone()
            if mask is not None:
                x[:, 23:24] = mask
            with torch.no_grad():
                actual = model(x)
                with patch.object(model, "stem_feat", ReferenceStem()):
                    reference = SJNet3_PointNet.forward(model, x)
            for key in actual:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)

    def test_rejects_wrong_channels(self):
        for channels in (23, 25):
            with self.assertRaises(AssertionError):
                self.model(torch.zeros(1, channels, 64, 64))
        with self.assertRaises(ValueError):
            SJNet3_PointNet_Gated(in_channels=23)

    def test_ring_conv_wraps_angle_not_radius(self):
        conv = RingConv2d(1, 1, 3)
        nn.init.ones_(conv.conv.weight)
        x = torch.zeros(1, 1, 5, 5)
        x[0, 0, 0, 0] = 1
        out = conv(x)
        self.assertEqual(out[0, 0, 0, -1].item(), 1)
        self.assertEqual(out[0, 0, -1, 0].item(), 0)


class DataOrderingTests(unittest.TestCase):
    def test_real_dataset_and_collate_channel_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "sequences" / "00"
            bev = seq / "polar_512_prlb"
            pn = seq / "pointnet_16ch"
            bev.mkdir(parents=True)
            pn.mkdir()
            raw = torch.tensor([1, 6, 2, 5, 10, 2, 4.0])[:, None, None].expand(7, 8, 8).clone()
            mask = torch.ones(1, 8, 8)
            with gzip.open(bev / "000000.pt", "wb") as f:
                torch.save(dict(proj_tensor=raw, mask_t=mask, labels_t=torch.ones(8, 8)), f)
            features = torch.arange(1, 17).float().expand(64, 16)
            with gzip.open(pn / "000000.pt", "wb") as f:
                torch.save(dict(cell_ids=torch.arange(64), features=features,
                                feature_dim=16, H=8, W=8, max_radius=51.2), f)
            dataset = SemanticKitti(tmp, [0], {}, {}, {}, {},
                                   sensor={"img_prop": {"height": 8, "width": 8}})
            batch = bev_collate_fn([dataset[0]])
            x = torch.cat([batch[0], batch[1]], dim=1)
            self.assertEqual(x.shape, (1, 24, 8, 8))
            expected_hand = torch.tensor([0, 1, 1, 1, 1, 0.2, 0.4])
            torch.testing.assert_close(x[0, :7, 0, 0], expected_hand)
            torch.testing.assert_close(x[0, 7:23], features[0, :, None, None].expand(16, 8, 8))
            torch.testing.assert_close(x[0, 23:24], mask)
            self.assertEqual(batch[14].count_nonzero().item(), 0)


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_delta_resume_and_minimum(self):
        early = training.EarlyStopping(patience=2, min_delta=0.001, min_epochs=4)
        self.assertFalse(early.update(0.4, 0))
        self.assertFalse(early.update(0.4005, 1))
        self.assertFalse(early.update(0.399, 2))
        restored = training.EarlyStopping(patience=2, min_delta=0.001, min_epochs=4)
        restored.load_state_dict(early.state_dict())
        self.assertTrue(restored.update(0.4, 3))
        self.assertFalse(restored.update(0.402, 4))
        self.assertEqual(restored.bad_epochs, 0)
        with self.assertRaises(ValueError):
            restored.update(float("nan"), 5)
        with self.assertRaises(ValueError):
            training.EarlyStopping(patience=3).load_state_dict(early.state_dict())

    def test_disabled_and_config_guards(self):
        early = training.EarlyStopping(enabled=False, patience=1, min_epochs=0)
        early.update(0.5, 0)
        self.assertFalse(early.update(0.4, 99))
        cfg = arch_config()
        training.validate_config(cfg)
        for key, value in (("max_epochs", 100), ("cos_epochs", 97),
                           ("copy_paste_prob", 0.5), ("report_epoch", 2)):
            broken = copy.deepcopy(cfg)
            broken["train"][key] = value
            with self.assertRaises(ValueError):
                training.validate_config(broken)
        with self.assertRaises(ValueError):
            training.validate_resume_checkpoint({"model": {}}, cfg, {})


class CheckpointTests(unittest.TestCase):
    def make_trainer(self, directory, arch=None, resume=None):
        parser = MagicMock()
        parser.get_n_classes.return_value = 2
        parser.to_xentropy.side_effect = lambda label: label
        parser.train_dataset.copy_paste_prob = 0.0
        parser.get_train_set.return_value.batch_size = 3
        parser.get_train_size.return_value = 1
        data = {"content": {0: 0.1, 1: 0.9}, "learning_ignore": {0: True, 1: False}}
        with patch.object(training, "Parser", return_value=parser), \
             patch.object(training, "SJNet3_PointNet_Gated", side_effect=lambda **kwargs: nn.Linear(2, 2)), \
             patch.object(training, "set_tensorboard", return_value=MagicMock()), \
             patch.object(torch.cuda, "is_available", return_value=False):
            return training.Trainer(arch or arch_config(), data, "unused", directory,
                                    MagicMock(), use_mps=False, resume=resume)

    def test_latest_best_and_resume_restore_raw_and_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)
            trainer.model(torch.ones(2, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.early_stopping.update(0.497, 4)
            trainer.early_stopping.update(0.490, 5)
            trainer.best_val_epoch = 4
            with torch.no_grad():
                trainer.ema_model.weight.add_(1.0)
            for is_best, name in ((False, "latest.path"), (True, "best_val.path")):
                trainer._save_checkpoint(5, 0.7, 0.497, is_best=is_best)
                path = Path(tmp) / name
                checkpoint = torch.load(path, weights_only=False)
                self.assertEqual(checkpoint["early_stopping"]["bad_epochs"], 1)
                expected = trainer.ema_model if is_best else trainer.model
                torch.testing.assert_close(checkpoint["model"]["weight"], expected.weight)
                resumed = self.make_trainer(tmp, resume=str(path))
                self.assertEqual(resumed.start_epoch, 6)
                self.assertEqual(resumed.best_val_epoch, 4)
                self.assertEqual(resumed.best_val_iou, 0.497)
                self.assertEqual(resumed.early_stopping.state_dict(), trainer.early_stopping.state_dict())
                torch.testing.assert_close(resumed.model.weight, trainer.model.weight)
                torch.testing.assert_close(resumed.ema_model.weight, trainer.ema_model.weight)
                self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
                expected_state = trainer.optimizer.state_dict()["state"]
                for key, state in resumed.optimizer.state_dict()["state"].items():
                    for item, value in state.items():
                        torch.testing.assert_close(value, expected_state[key][item])
                before = path.read_bytes()
                with patch.object(training.torch, "save", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        trainer._save_checkpoint(6, 0.8, 0.5, is_best=is_best)
                self.assertEqual(before, path.read_bytes())

    def test_training_stops_and_saves_sub_delta_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = arch_config()
            cfg["train"]["early_stopping"].update(patience=2, min_epochs=0)
            trainer = self.make_trainer(tmp, cfg)

            def epoch(**kwargs):
                trainer.optimizer.zero_grad()
                trainer.model(torch.ones(1, 2)).sum().backward()
                trainer.optimizer.step()
                return 0.8, 0.7, 0.3, 0.0

            trainer.train_epoch = MagicMock(side_effect=epoch)
            trainer.validate = MagicMock(side_effect=[(0.8, score, 0.5) for score in (0.4, 0.4005, 0.4008)])
            trainer.train()
            self.assertEqual(trainer.train_epoch.call_count, 3)
            latest = torch.load(Path(tmp) / "latest.path", weights_only=False)
            best = torch.load(Path(tmp) / "best_val.path", weights_only=False)
            self.assertEqual(latest["epoch"], 2)
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(best["best_val_iou"], 0.4008)
            self.assertEqual(latest["early_stopping"]["bad_epochs"], 2)
            resumed = self.make_trainer(tmp, cfg, str(Path(tmp) / "latest.path"))
            resumed.train_epoch = MagicMock()
            resumed.train()
            resumed.train_epoch.assert_not_called()

    def test_training_reaches_70_epochs_when_improving(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)

            def epoch(**kwargs):
                trainer.optimizer.zero_grad()
                trainer.model(torch.ones(1, 2)).sum().backward()
                trainer.optimizer.step()
                return 0.8, 0.7, 0.3, 0.0

            trainer.train_epoch = MagicMock(side_effect=epoch)
            trainer.validate = MagicMock(side_effect=[(0.8, 0.2 + i * 0.005, 0.5) for i in range(70)])
            trainer.train()
            self.assertEqual(trainer.train_epoch.call_count, 70)
            latest = torch.load(Path(tmp) / "latest.path", weights_only=False)
            self.assertEqual(latest["epoch"], 69)
            self.assertEqual(latest["early_stopping"]["bad_epochs"], 0)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 2e-5)
            self.assertEqual(trainer.w_aux2, 0)
            self.assertEqual(trainer.w_aux4, 0)

    def test_cli_does_not_touch_existing_step_c_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "log.txt"
            marker.write_text("existing Step C log\n")
            argv = ["train_pointnet_hybrid_gated.py", "--dataset", "SemanticKitti",
                    "--arch_cfg", str(ROOT / "config/arch/sjnet_pointnet_gated70.yaml"),
                    "--data_cfg", str(ROOT / "config/labels/semantic-kitti.yaml"), "--log", tmp]
            with patch("sys.argv", argv), patch("sys.stderr", new=io.StringIO()), \
                 patch.object(entry, "Trainer") as constructor:
                with self.assertRaises(SystemExit) as error:
                    entry.main()
                self.assertEqual(error.exception.code, 2)
                constructor.assert_not_called()
            self.assertEqual(marker.read_text(), "existing Step C log\n")
            self.assertFalse((Path(tmp) / "arch_cfg.yaml").exists())

            baseline_arch = arch_config()
            baseline_arch["model"]["name"] = "SJNet3_PointNet"
            (Path(tmp) / "arch_cfg.yaml").write_text(yaml.safe_dump(baseline_arch))
            with (ROOT / "config/labels/semantic-kitti.yaml").open() as f:
                data = yaml.safe_load(f)
            checkpoint = dict(model_name="SJNet3_PointNet_Gated", arch_cfg=arch_config(),
                              data_cfg=data, early_stopping={})
            with patch("sys.argv", argv + ["--resume", "gated_latest.path"]), \
                 patch("sys.stderr", new=io.StringIO()), \
                 patch.object(entry.torch, "load", return_value=checkpoint), \
                 patch.object(entry, "Trainer") as constructor:
                with self.assertRaises(SystemExit) as error:
                    entry.main()
                self.assertEqual(error.exception.code, 2)
                constructor.assert_not_called()
            self.assertEqual(marker.read_text(), "existing Step C log\n")
            self.assertEqual(yaml.safe_load((Path(tmp) / "arch_cfg.yaml").read_text()), baseline_arch)


if __name__ == "__main__":
    unittest.main()
