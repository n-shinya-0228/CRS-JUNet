import os
import json
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from .dataset import ParserBEVSphere
from .model import SphericalBEVFusion


class Trainer:
    def __init__(self, arch, data, datadir, logdir, logger, pretrained=None,
                 use_mps=False, resume=None):
        self.arch = arch
        self.data = data
        self.datadir = datadir
        self.logdir = logdir
        self.logger = logger
        self.pretrained = pretrained
        self.resume = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.logdir, exist_ok=True)

        self.parser = ParserBEVSphere(datadir, data, arch)
        self.nclasses = self.parser.get_n_classes()
        self.model = SphericalBEVFusion(nclasses=self.nclasses).to(self.device)

        frequencies = torch.zeros(self.nclasses, dtype=torch.float32)
        for raw_id, frequency in data["content"].items():
            train_id = int(data["learning_map"].get(int(raw_id), 0))
            if train_id < self.nclasses:
                frequencies[train_id] += float(frequency)
        epsilon = float(arch["train"].get("epsilon_w", 0.01))
        weights = 1.0 / (frequencies + epsilon)
        weights = torch.clamp(weights, max=20.0)
        weights[0] = 0.0
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0).to(self.device)
        self.sphere_loss_weight = float(arch["train"].get("sphere_loss_weight", 0.3))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(arch["train"]["lr"]),
            weight_decay=float(arch["train"].get("w_decay", 0.01)),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(arch["train"]["max_epochs"])),
            eta_min=float(arch["train"].get("min_lr", 1e-5)),
        )
        self.start_epoch = 0
        self.best_miou = 0.0
        self.writer = SummaryWriter(osp.join(self.logdir, "tfrecord"))

        checkpoint = resume or pretrained
        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        if osp.isdir(path):
            candidates = [
                osp.join(path, "best_val.path"),
                osp.join(path, "latest.path"),
            ]
            path = next((candidate for candidate in candidates if osp.isfile(candidate)), "")
            if not path:
                raise FileNotFoundError("No best_val.path or latest.path in checkpoint directory")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        self.model.load_state_dict(state)
        if self.resume and "optim" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optim"])
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.best_miou = float(checkpoint.get("best_miou", 0.0))

    def _move_batch(self, batch):
        keys = (
            "bev", "sphere", "bev_labels", "sphere_labels",
            "bev_to_sphere", "bev_to_sphere_valid",
        )
        return {key: batch[key].to(self.device, non_blocking=True) for key in keys}

    def _forward_loss(self, batch):
        outputs = self.model(
            batch["bev"],
            batch["sphere"],
            batch["bev_to_sphere"],
            batch["bev_to_sphere_valid"],
        )
        bev_loss = self._safe_loss(outputs["bev_logits"], batch["bev_labels"])
        sphere_loss = self._safe_loss(outputs["sphere_logits"], batch["sphere_labels"])
        return outputs, bev_loss + self.sphere_loss_weight * sphere_loss, bev_loss, sphere_loss

    def _safe_loss(self, logits, target):
        if not (target != 0).any():
            return logits.sum() * 0.0
        return self.criterion(logits, target)

    def _update_confusion(self, confusion, prediction, target):
        valid = (target > 0) & (target < self.nclasses)
        encoded = target[valid] * self.nclasses + prediction[valid]
        confusion += torch.bincount(
            encoded, minlength=self.nclasses * self.nclasses
        ).reshape(self.nclasses, self.nclasses)

    def _metrics(self, confusion):
        confusion = confusion.clone()
        confusion[0, :] = 0
        confusion[:, 0] = 0
        intersection = confusion.diag()[1:]
        union = confusion.sum(dim=0)[1:] + confusion.sum(dim=1)[1:] - intersection
        iou = intersection.float() / union.clamp_min(1).float()
        precision = intersection.sum().float() / confusion.sum(dim=0)[1:].sum().clamp_min(1)
        return {"miou": iou.mean().item(), "accuracy": precision.item(),
                "class_iou": iou.tolist()}

    def _run_epoch(self, loader, training):
        self.model.train(training)
        confusion = torch.zeros(
            (self.nclasses, self.nclasses), dtype=torch.int64, device=self.device
        )
        loss_total = 0.0
        bev_loss_total = 0.0
        sphere_loss_total = 0.0
        sample_count = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for raw_batch in loader:
                batch = self._move_batch(raw_batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                outputs, loss, bev_loss, sphere_loss = self._forward_loss(batch)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                self._update_confusion(
                    confusion, outputs["bev_logits"].argmax(dim=1), batch["bev_labels"]
                )
                batch_size = batch["bev"].shape[0]
                loss_total += loss.item() * batch_size
                bev_loss_total += bev_loss.item() * batch_size
                sphere_loss_total += sphere_loss.item() * batch_size
                sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("The fusion dataloader is empty; run prepare_dual_projection.py first.")
        return {
            "loss": loss_total / sample_count,
            "bev_loss": bev_loss_total / sample_count,
            "sphere_loss": sphere_loss_total / sample_count,
            **self._metrics(confusion),
        }

    def _save_checkpoint(self, path, epoch, metrics):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_miou": self.best_miou,
                "metrics": metrics,
            },
            path,
        )

    def train(self):
        max_epochs = int(self.arch["train"]["max_epochs"])
        report_epoch = max(1, int(self.arch["train"].get("report_epoch", 1)))
        for epoch in range(self.start_epoch, max_epochs):
            train_metrics = self._run_epoch(self.parser.trainloader, training=True)
            self.scheduler.step()
            self.logger.info(
                f"Epoch {epoch + 1} train loss {train_metrics['loss']:.4f} "
                f"(BEV {train_metrics['bev_loss']:.4f}, "
                f"sphere {train_metrics['sphere_loss']:.4f}), "
                f"mIoU {train_metrics['miou']:.4f}"
            )
            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("train/miou", train_metrics["miou"], epoch)

            if (epoch + 1) % report_epoch == 0:
                valid_metrics = self._run_epoch(self.parser.validloader, training=False)
                self.logger.info(
                    f"Epoch {epoch + 1} validation loss {valid_metrics['loss']:.4f}, "
                    f"mIoU {valid_metrics['miou']:.4f}"
                )
                self.logger.info(f"Validation accuracy {valid_metrics['accuracy']:.4f}")
                for cid, iou in enumerate(valid_metrics["class_iou"], 1):
                    raw_id = self.data["learning_map_inv"][cid]
                    self.logger.info(f"IoU class {cid} [{self.data['labels'][raw_id]}] = {iou:.4f}")
                with open(osp.join(self.logdir, "metrics.jsonl"), "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"epoch": epoch + 1, "train": train_metrics,
                                             "valid": valid_metrics}) + "\n")
                self.writer.add_scalar("valid/loss", valid_metrics["loss"], epoch)
                self.writer.add_scalar("valid/miou", valid_metrics["miou"], epoch)
                self._save_checkpoint(
                    osp.join(self.logdir, "latest.path"), epoch, valid_metrics
                )
                if valid_metrics["miou"] >= self.best_miou:
                    self.best_miou = valid_metrics["miou"]
                    self._save_checkpoint(
                        osp.join(self.logdir, "best_val.path"), epoch, valid_metrics
                    )

        self.writer.close()
        with open(osp.join(self.logdir, "results.json"), "w", encoding="utf-8") as stream:
            json.dump({"best_validation_miou": self.best_miou,
                       "metric_domain": "polar_bev_pixels",
                       "epochs_completed": max_epochs}, stream, indent=2)