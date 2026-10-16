#!/usr/bin/env python3
# Step C.1b: identical gated model/losses, stable LR and diagnostic logging.
# - expects model(in_vol) -> dict with keys: 'logits', 'aux2', 'aux4', 'boundary'
# - combines CE/Focal + Lovasz (optional) + Boundary BCE + Aux losses
# - keeps scheduler/optimizer structure + EMA model for better mIoU

# trainer_PointNet_Gated_StableLR
import torch
torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import yaml
import time
import os
import os.path as osp
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from contextlib import contextmanager
import copy  # EMA 用
import math

from lib.utils.avgmeter import *
from lib.utils.warmupLR import *   # (互換性のために import のみ)
from lib.utils.ioueval import *
from lib.dataset.Parser_PointNet import Parser
from lib.models.SJNet3_PointNet_Gated import SJNet3_PointNet_Gated
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_Alpha import SJNet3_PointNet_Gated_Local3_Alpha
from lib.models.SJNet3_PointNet_Gated_Local3_SpatialAlpha import SJNet3_PointNet_Gated_Local3_SpatialAlpha
from lib.utils.pointnet_local3_diagnostics import local3_feature_statistics
from lib.utils.pointnet_alpha_diagnostics import (
    build_alpha_optimizer, local_alpha_parameter, local_alpha_statistics, log_best_comparison,
)
from lib.utils.pointnet_spatial_alpha_diagnostics import (
    spatial_alpha_module, build_spatial_alpha_optimizer, spatial_alpha_capture,
    SpatialAlphaValidation, log_spatial_validation, save_spatial_best_summary,
)
from lib.losses import *           # Lovasz_softmax / Depth_Loss など



class EarlyStopping:
    """Monitor validation mIoU, independently of strict best-checkpoint selection."""

    def __init__(self, enabled=True, patience=12, min_delta=0.001, min_epochs=30):
        self.enabled = bool(enabled)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        if self.patience < 1 or self.min_epochs < 0:
            raise ValueError("Early stopping needs patience >= 1 and min_epochs >= 0")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early stopping min_delta must be finite and >= 0")
        self.best_score = None
        self.bad_epochs = 0

    def update(self, score, epoch):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"Non-finite validation mIoU at epoch {epoch}: {score}")
        if self.best_score is None or (score > self.best_score and score >= self.best_score + self.min_delta):
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.should_stop(epoch)

    def should_stop(self, epoch):
        return (self.enabled and epoch >= self.min_epochs
                and self.bad_epochs >= self.patience)

    def state_dict(self):
        return dict(enabled=self.enabled, patience=self.patience,
                    min_delta=self.min_delta, min_epochs=self.min_epochs,
                    best_score=self.best_score, bad_epochs=self.bad_epochs)

    def load_state_dict(self, state):
        for key in ("enabled", "patience", "min_delta", "min_epochs"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Early stopping setting changed on resume: {key}")
        self.best_score = state["best_score"]
        self.bad_epochs = int(state["bad_epochs"])


def experiment_id(arch):
    names = {"SJNet3_PointNet_Gated": "Step C.1b",
             "SJNet3_PointNet_Gated_Local3": "Step C.2",
             "SJNet3_PointNet_Gated_Local3_Alpha": "Step C.3",
             "SJNet3_PointNet_Gated_Local3_SpatialAlpha": "Step C.4"}
    name = arch["model"]["name"]
    if name not in names:
        raise ValueError(f"Unsupported StableLR model: {name}")
    return names[name]


def validate_config(arch):
    experiment_id(arch)
    if int(arch["model"].get("in_channels", 24)) != 24:
        raise ValueError("Step C.1 requires 24 input channels")
    if int(arch["dataset"].get("pointnet", {}).get("feature_dim", 16)) != 16:
        raise ValueError("Step C.1 requires 16 PointNet channels")
    train = arch["train"]
    if float(train.get("copy_paste_prob", 0.0)) != 0.0:
        raise ValueError("Step C.1 reuses the Copy-Paste OFF parser")
    max_epochs = int(train["max_epochs"])
    if not 1 <= max_epochs <= 70:
        raise ValueError("Step C.1 max_epochs must be between 1 and 70")
    warmup, cosine = int(train["wup_epochs"]), int(train["cos_epochs"])
    if warmup < 1 or cosine < 1 or warmup + cosine != max_epochs:
        raise ValueError("Step C.1b requires warmup + cosine = max_epochs; no constant phase")
    if not 0 < float(train["min_lr"]) < float(train["lr"]):
        raise ValueError("Require 0 < min_lr < max LR")
    if int(train["report_epoch"]) != 1:
        raise ValueError("Early stopping requires validation every epoch")
    stopping = EarlyStopping(**train.get("early_stopping", {}))
    if stopping.min_epochs > max_epochs:
        raise ValueError("Early stopping min_epochs exceeds max_epochs")


def validate_resume_checkpoint(state, arch, data):
    if state.get("experiment") != experiment_id(arch):
        raise ValueError("Resume requires a checkpoint from the same experiment")
    if state.get("model_name") != arch["model"]["name"]:
        raise ValueError("Resume model differs from the requested model")
    if state.get("arch_cfg") != arch or state.get("data_cfg") != data:
        raise ValueError("Resume config differs from the saved Step C.1 checkpoint")
    if "early_stopping" not in state:
        raise ValueError("Resume checkpoint is missing early stopping state")



def build_scheduler(optimizer, train):
    """LR is for the current zero-based epoch; step once AFTER train_epoch."""
    warmup = int(train["wup_epochs"])
    cosine = int(train["cos_epochs"])
    if warmup < 1 or cosine < 1 or warmup + cosine != int(train["max_epochs"]):
        raise ValueError("Expected positive warmup/cosine phases with no constant phase")
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.1 + 0.9 * min(epoch / warmup, 1.0))
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cosine, eta_min=float(train["min_lr"]))
    return SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup])


def scheduler_preview(train, through_epoch=10):
    """Exercise the same scheduler and end-of-epoch step order as training."""
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = optim.AdamW([parameter], lr=float(train["lr"]),
                            weight_decay=float(train["w_decay"]))
    scheduler = build_scheduler(optimizer, train)
    rates = []
    for epoch in range(through_epoch + 1):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


@contextmanager
def fusion_gate_statistics(model):
    """Read-only hook on the existing sigmoid gate, including DataParallel shards."""
    core = model.module if isinstance(model, nn.DataParallel) else model
    parts = []
    result = {}

    def capture(module, inputs, output):
        value = output.detach().float()
        variance, mean = torch.var_mean(value, unbiased=False)
        parts.append((value.numel(), mean.item(), variance.item(),
                      value.min().item(), value.max().item()))

    handle = core.point_gate.register_forward_hook(capture)
    try:
        yield result
    finally:
        handle.remove()
        if parts:
            count = sum(part[0] for part in parts)
            mean = sum(n * m for n, m, _, _, _ in parts) / count
            variance = sum(n * (v + (m - mean) ** 2)
                           for n, m, v, _, _ in parts) / count
            result.update(mean=mean, std=math.sqrt(max(variance, 0.0)),
                          min=min(part[3] for part in parts),
                          max=max(part[4] for part in parts))


# -----------------------------
# Focal Loss (class weight + ignore_index 対応)
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0,
                 ignore_index=0, reduction='mean'):
        super().__init__()
        if weight is not None:
            self.register_buffer('class_weight', weight.clone().detach())
        else:
            self.class_weight = None
        self.gamma  = gamma
        self.ignore = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B,C,H,W), targets: (B,H,W)
        logp  = F.log_softmax(inputs, dim=1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt    = logpt.exp()
        ce_loss = -logpt

        if self.class_weight is not None:
            w = self.class_weight[targets]
            ce_loss = ce_loss * w

        loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.ignore is not None:
            valid = targets != self.ignore
            loss  = loss[valid]
            if self.class_weight is not None:
                w = w[valid]

        if self.reduction == 'mean':
            if self.class_weight is not None:
                return loss.sum() / (w.sum() + 1e-6)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# -----------------------------
# Sobel フィルタで境界強度
# -----------------------------
class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', sobel_x)
        self.register_buffer('ky', sobel_y)

    def forward(self, x):  # x: (B,1,H,W) float
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy)


def set_tensorboard(path):
    return SummaryWriter(path)

def apply_polarmix(in_vol, proj_mask, proj_labels, paste_mask):
    B, C, H, W = in_vol.shape
    rand_index = torch.randperm(B).to(in_vol.device)
    cut_w = np.random.randint(0, W)

    mixed_vol = in_vol.clone()
    mixed_vol[:, :, :, cut_w:] = in_vol[rand_index, :, :, cut_w:]

    mixed_mask = proj_mask.clone()
    if mixed_mask.dim() == 3:
        mixed_mask[:, :, cut_w:] = proj_mask[rand_index, :, cut_w:]
    else:
        mixed_mask[:, :, :, cut_w:] = proj_mask[rand_index, :, :, cut_w:]

    mixed_labels = proj_labels.clone()
    mixed_labels[:, :, cut_w:] = proj_labels[rand_index, :, cut_w:]

    mixed_paste = paste_mask.clone()
    if mixed_paste.dim() == 3:
        mixed_paste[:, :, cut_w:] = paste_mask[rand_index, :, cut_w:]
    else:
        mixed_paste[:, :, :, cut_w:] = paste_mask[rand_index, :, :, cut_w:]

    return mixed_vol, mixed_mask, mixed_labels, mixed_paste

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, logger, pretrained=None, use_mps=True, resume=None):
        validate_config(ARCH)
        self.early_stopping = EarlyStopping(**ARCH['train'].get('early_stopping', {}))
        self.best_val_epoch = -1
        self.best_local_alpha = None
        self.best_spatial_summary = None
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.logger = logger
        self.pretrained = pretrained
        self.use_mps = use_mps
        self.resume = resume
        self.scaler = torch.amp.GradScaler('cuda')
        # シード
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        self.writer = set_tensorboard(osp.join(logdir, 'tfrecord'))

        # save
        import sys
        from datetime import datetime

        os.makedirs(self.log, exist_ok=True)

        yaml_path = None
        for i, arg in enumerate(sys.argv):
            if arg == '--arch_cfg' and i + 1 < len(sys.argv):
                yaml_path = sys.argv[i + 1]
                break

        model_name = self.ARCH['model']['name']
        model_py_path = osp.join('lib', 'models', f"{model_name}.py")
        dataset_name = 'SemanticKitti_PointNet.py'
        parser_name = 'Parser_PointNet.py'
        trainer_name = 'trainer_PointNet_Gated_StableLR.py'
        dataset_py_path = osp.join('lib', 'dataset', dataset_name)
        parser_py_path = osp.join('lib', 'dataset', parser_name)

        used_files_txt = osp.join(self.log, 'used_files.txt')
        with open(used_files_txt, 'a' if self.resume else 'w') as f:
            f.write("=== Training Execution Log ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")
            f.write("-" * 30 + "\n")
            f.write("[Used Files]\n")
            f.write(f"Config YAML : {yaml_path if yaml_path else 'Not specified'}\n")
            f.write(f"Model File  : {model_py_path}\n")
            f.write("Base Model  : lib/models/SJNet3_PointNet.py (read-only)\n")
            f.write(f"Dataset File: {dataset_py_path}\n")
            f.write(f"Parser File : {parser_py_path}\n")
            f.write(f"Trainer File : {trainer_name}\n")
            f.write("-" * 30 + "\n")
            f.write("[Key Settings]\n")
            f.write(f"Dataset Root: {self.datadir}\n")
            f.write(f"Log Dir     : {self.log}\n")

        self.logger.info(f"Saved execution info to {used_files_txt}")

        # Data
        self.parser = Parser(root=self.datadir,
                             data_cfg=DATA,
                             arch_cfg=ARCH,
                             gt=True,
                             shuffle_train=True)
        self.logger.info(
            f"Copy-Paste probability: {self.parser.train_dataset.copy_paste_prob:.1f}"
        )
        self.logger.info(f"Train batch size: {self.parser.get_train_set().batch_size}")
        self.logger.info(f"Train scans: {len(self.parser.train_dataset)}")
        self.logger.info(f"Train batches per epoch: {self.parser.get_train_size()}")

        # Class weights
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq

        prob = content / (content.sum() + 1e-6)

        self.loss_w = 1 / (prob + epsilon_w)

        self.loss_w = torch.clamp(self.loss_w, max=20.0)

        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0.0
        self.logger.info("Loss weights from content: %s" %
                         (self.loss_w.data.cpu().numpy().tolist(),))

        # Model
        model_cls = (SJNet3_PointNet_Gated if model_name == "SJNet3_PointNet_Gated"
                     else SJNet3_PointNet_Gated_Local3)
        if model_name == "SJNet3_PointNet_Gated_Local3_Alpha":
            model_cls = SJNet3_PointNet_Gated_Local3_Alpha
        elif model_name == "SJNet3_PointNet_Gated_Local3_SpatialAlpha":
            model_cls = SJNet3_PointNet_Gated_Local3_SpatialAlpha
        self.expected_in_channels = int(self.ARCH["model"].get("in_channels", 24))
        self.model = model_cls(
            in_channels=self.expected_in_channels,
            nclasses=self.parser.get_n_classes(),
            drop=self.ARCH["model"].get("dropout", 0.5)
        )
        # self.model = torch.compile(self.model, backend="inductor", mode="default")
        weights_total = sum(p.numel() for p in self.model.parameters())
        weights_grad = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
        self.logger.info("Total number of parameters: " + str(weights_total))
        self.logger.info("Total number of parameters requires_grad: " +
                         str(weights_grad))
        if model_name == "SJNet3_PointNet_Gated_Local3":
            added = sum(p.numel() for p in self.model.point_context.parameters())
            self.logger.info(f"Local3 parameters added vs C.1b: {added}")

        # Device
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Training in device: %s" % str(self.device))
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

        # ---------------- EMA model ----------------
        self.use_ema = bool(self.ARCH["train"].get("use_ema", True))
        self.ema_decay = float(self.ARCH["train"].get("ema_decay", 0.99))
        self.ema_model = None
        if self.use_ema:
            self._build_ema_model()
        # -------------------------------------------

        # Losses -------------- ignore_index=0 に統一
        train_loss_type = self.ARCH["train"].get("loss", "focal_gamma")
        if train_loss_type == "xentropy":
            self.criterion_main = nn.CrossEntropyLoss(
                weight=self.loss_w, ignore_index=0).to(self.device)
            self.lovasz = Lovasz_softmax(ignore=0).to(self.device)
        elif train_loss_type == "focal_gamma":
            self.criterion_main = FocalLoss(
                weight=self.loss_w, gamma=1.0, ignore_index=0).to(self.device)
            self.lovasz = Lovasz_softmax(ignore=0).to(self.device)
        else:
            raise Exception('Loss not defined in config file')

        # boundary BCE は後段でマスク平均を取るため reduction='none'
        self.boundary_criterion = nn.BCEWithLogitsLoss(
            reduction='none').to(self.device)
        self.sobel = SobelFilter().to(self.device)

        # Optimizer
        if spatial_alpha_module(self.model) is not None:
            self.optimizer = build_spatial_alpha_optimizer(self.model, self.ARCH["train"])
            self.logger.info("C.4 parameters added vs C.2: 17 (global=1, predictor=16)")
            for index, group in enumerate(self.optimizer.param_groups):
                self.logger.info(
                    f"AdamW group {index}: parameters={sum(p.numel() for p in group['params'])} | "
                    f"weight_decay={group['weight_decay']} | max_lr={group['lr']}"
                )
        elif local_alpha_parameter(self.model) is not None:
            self.optimizer = build_alpha_optimizer(self.model, self.ARCH["train"])
            self.logger.info("C.3 parameters added vs C.2: 1 (scalar local_alpha)")
            self.logger.info(f"Initial local alpha: {local_alpha_parameter(self.model).item():.6f}")
            for index, group in enumerate(self.optimizer.param_groups):
                self.logger.info(
                    f"AdamW group {index}: parameters={sum(p.numel() for p in group['params'])} | "
                    f"weight_decay={group['weight_decay']} | max_lr={group['lr']}"
                )
        else:
            self.optimizer = optim.AdamW(self.model.parameters(),
                                         lr=self.ARCH["train"]["lr"],
                                         weight_decay=self.ARCH["train"]["w_decay"])

        self.scheduler = build_scheduler(self.optimizer, self.ARCH["train"])

        # (optional) resume
        self.start_epoch = 0
        self.best_train_iou = 0.0
        self.best_val_iou = -float('inf')
        if self.resume is not None:
            try:
                try:
                    torch.serialization.add_safe_globals([torch.optim.lr_scheduler.CyclicLR])
                except AttributeError:
                    pass
                w_dict = torch.load(self.resume, map_location=self.device, weights_only=False)
                validate_resume_checkpoint(w_dict, self.ARCH, self.DATA)

                self.model.load_state_dict(w_dict.get('model_train', w_dict['model']))
                self.optimizer.load_state_dict(w_dict['optim'])
                self.scheduler.load_state_dict(w_dict['scheduler'])
                if 'scaler' in w_dict:
                    self.scaler.load_state_dict(w_dict['scaler'])
                self.start_epoch = w_dict.get('epoch', 0) + 1
                self.best_train_iou = w_dict.get('best_train_iou', 0.0)
                self.best_val_iou = w_dict['best_val_iou']
                self.best_val_epoch = w_dict['best_val_epoch']
                self.best_local_alpha = w_dict.get('best_local_alpha')
                self.best_spatial_summary = w_dict.get('best_spatial_summary')
                self.early_stopping.load_state_dict(w_dict['early_stopping'])

                if self.use_ema:
                    self._build_ema_model()
                    if 'model_ema' in w_dict:
                        self.ema_model.load_state_dict(w_dict['model_ema'])
                self.logger.info(f"✅ Successfully RESUMED training from {self.resume}. Starting at epoch {self.start_epoch}")
            except Exception as e:
                raise RuntimeError(f"Couldn't resume from {self.resume}: {e}") from e

        elif self.pretrained is not None:
            try:
                w_dict = torch.load(self.pretrained, map_location=self.device, weights_only=False)
                state_dict = w_dict['model'] if 'model' in w_dict else w_dict

                self.model.load_state_dict(state_dict, strict=True)

                if self.use_ema:
                    self._build_ema_model()
                self.logger.info(f"✅ Successfully loaded PRETRAINED weights from {self.pretrained}. Starting from scratch.")
            except Exception as e:
                raise RuntimeError(f"Couldn't load gated pretrained parameters: {e}") from e

        # Loss weights (tunable)
        self.w_aux2 = 0.30
        self.w_aux4 = 0.15
        self.w_aux8 = 0.10
        self.w_lovasz = 0.50
        self.w_boundary = 0.20
        self.w_paste = 0.00
        self.logger.info(f"Paste-region weighted loss: {self.w_paste:.2f}")
        self.logger.info(
            f"Starting epoch: {self.start_epoch} | "
            f"Resume: {self.resume} | Pretrained: {self.pretrained}"
        )

        self.logger.info(
            f"{self.ARCH.get('experiment', {}).get('name', experiment_id(self.ARCH))} | "
            f"32->48 RingConv fusion | max_epochs={self.ARCH['train']['max_epochs']} | "
            f"early_stopping={self.early_stopping.state_dict()}"
        )

        self.logger.info("Scheduler sanity preview (not completed training epochs):")
        for epoch, rate in enumerate(scheduler_preview(self.ARCH["train"])):
            self.logger.info(f"LR preview | Epoch {epoch:02d} | Learning rate: {rate:.6e}")
        reference = self.ARCH.get("experiment", {})
        self.logger.info(
            f"Comparison references: baseline 8ch={reference.get('baseline_8ch_miou')} | "
            f"naive hybrid={reference.get('naive_hybrid_miou')} | "
            f"Step C.1={reference.get('step_c1_miou')} "
            f"(epoch {reference.get('step_c1_best_epoch')})"
        )
        if "step_c1b_miou" in reference:
            self.logger.info(
                f"Step C.1b reference={reference['step_c1b_miou']:.6f} "
                f"(epoch {reference['step_c1b_best_epoch']}); targets: "
                f">{reference['step_c1b_miou']:.6f}, >{reference['baseline_8ch_miou']:.6f}, "
                f"clear improvement >= {reference['clear_improvement_miou']:.6f}"
            )
            self.logger.info(
                "Mask audit: unchanged soft fusion gate; no hard zeroing at missing cells. "
                "Local3 may spread features into unobserved cells; observation mask is unchanged."
            )
        if local_alpha_parameter(self.model) is not None:
            self.logger.info(
                f"Step C.2 reference={reference['step_c2_miou']:.6f} "
                f"(epoch {reference['step_c2_best_epoch']}); Step C.3 mIoU=?; "
                f"first target >{reference['step_c2_miou']:.6f}, "
                f"clear improvement >= {reference['clear_improvement_miou']:.6f}"
            )
            self.logger.info("Seed unchanged from C.2: torch=0, numpy=0; fixed PointNet features.")
            self.logger.info(
                f"Class reference sources: Baseline epoch {reference.get('baseline_8ch_best_epoch')} "
                f"({reference.get('baseline_8ch_reference_log')}, 3 decimals); "
                "C.1b epoch 9 (3 decimals); C.2 epoch 18 (6 decimals)"
            )
            if not reference.get('baseline_8ch_class_iou'):
                self.logger.info("Baseline class references unavailable: reported as N/A, not inferred.")
        if spatial_alpha_module(self.model) is not None:
            self.logger.info(
                f"C.4 references: Baseline={reference['baseline_8ch_miou']:.6f} | "
                f"C.1b={reference['step_c1b_miou']:.6f} | C.2={reference['step_c2_miou']:.6f} | "
                f"C.3={reference['step_c3_miou']:.6f}; C.4 mIoU=?"
            )
            self.logger.info("C.4 initialization: global=1, predictor=0; seed torch=0/numpy=0 unchanged")
            self.logger.info("Spatial alpha class statistics: full validation, valid labels AND observed cells only")

    def _log_local_alpha(self):
        alpha_train = local_alpha_parameter(self.model)
        if alpha_train is None:
            return
        eval_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
        alpha = local_alpha_parameter(eval_model).detach().item()
        raw = alpha_train.detach().item()
        source = "EMA" if eval_model is self.ema_model else "training"
        self.logger.info(f"Local alpha: {alpha:.6f} ({source}, validation) | training alpha: {raw:.6f}")
        epoch = getattr(self, 'current_epoch', self.start_epoch)
        self.writer.add_scalar('validating/local_alpha', alpha, epoch)
        self.writer.add_scalar('training/local_alpha', raw, epoch)
        if any(not math.isfinite(value) or value <= -5.0 or value >= 10.0 for value in (alpha, raw)):
            self.logger.warning(f"Extreme local alpha: validation={alpha}, training={raw}; no clamp applied")

    def _build_ema_model(self):
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema or self.ema_model is None:
            return
        with torch.no_grad():
            msd = self.model.state_dict()
            esd = self.ema_model.state_dict()
            for k, v in msd.items():
                if k not in esd:
                    continue
                ema_v = esd[k]
                # float 以外（long / int / bool / etc）はそのままコピー
                if (not torch.is_tensor(v)) or (not torch.is_tensor(ema_v)):
                    esd[k] = v.detach()
                    continue
                if (not torch.is_floating_point(v)) or (not torch.is_floating_point(ema_v)):
                    esd[k] = v.detach()
                    continue
                # 通常の EMA 更新
                ema_v.mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)
            self.ema_model.load_state_dict(esd)

    def _compute_boundary_gt(self, proj_labels):
        # proj_labels: (B,H,W) long -> float -> (B,1,H,W)
        proj_labels_float = proj_labels.float().unsqueeze(1)
        boundary_mag = self.sobel(proj_labels_float)
        boundary_bin = (boundary_mag > 0.1).float()
        return boundary_bin

    def _focal_loss_map(self, logits, labels):
        # logits: (B,C,H,W), labels: (B,H,W)
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        loss = -((1.0 - pt) ** 1.0) * logpt  # gamma=1.0

        if self.loss_w is not None:
            class_w = self.loss_w.to(logits.device)
            loss = loss * class_w[labels]

        valid = labels != 0
        loss = loss * valid.float()

        return loss

    def train(self):
        best_train_iou = self.best_train_iou
        best_val_iou = self.best_val_iou

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                self.logger.info(f"Ignoring class {i} in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

        if self.early_stopping.should_stop(self.start_epoch - 1):
            self.logger.info("Checkpoint already reached early stopping; no further training.")
            if self.best_local_alpha is not None:
                self.logger.info(f"Best local alpha: {self.best_local_alpha:.6f} (epoch {self.best_val_epoch})")
            self.writer.close()
            return

        # train & validate
        for epoch in range(self.start_epoch,
                           self.ARCH["train"]["max_epochs"]):
            should_stop = False
            self.current_epoch = epoch
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Epoch {epoch:02d} | Learning rate: {lr:.6e}")
            self.writer.add_scalar("training/lr", lr, epoch)


            decay_start = 50  # 減衰を開始するエポック
            decay_end = 60    # 完全に 0.0 にするエポック

            if decay_start <= epoch <= decay_end:
                ratio = (decay_end - epoch) / (decay_end - decay_start)

                self.w_aux2 = 0.30 * ratio
                self.w_aux4 = 0.15 * ratio

                self.logger.info("-" * 50)
                self.logger.info(f"[Epoch {epoch}] Softly decaying Aux Losses... ratio: {ratio:.2f}")
                self.logger.info(f" -> w_aux2: {self.w_aux2:.4f}, w_aux4: {self.w_aux4:.4f}")
                self.logger.info("-" * 50)

            elif epoch > decay_end:
                self.w_aux2 = 0.0
                self.w_aux4 = 0.0

            acc, iou, loss, update_mean = self.train_epoch(
                train_loader=self.parser.get_train_set(),
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                evaluator=self.evaluator,
                color_fn=None,
                report=self.ARCH["train"]["report_batch"]
            )

            self.scheduler.step()

            self.writer.add_scalar('training/acc', acc, epoch)
            self.writer.add_scalar('training/mIoU', iou, epoch)
            self.writer.add_scalar('training/loss', loss, epoch)
            self.writer.add_scalar('training/update_mean',
                                   update_mean, epoch)

            if iou > best_train_iou:
                self.logger.info("Best mean IoU in training so far.")
                best_train_iou = iou

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                self.logger.info("*" * 80)
                acc_v, iou_v, loss_v = self.validate(
                    val_loader=self.parser.get_valid_set(),
                    model=self.model,
                    evaluator=self.evaluator,
                    class_func=self.parser.get_xentropy_class_string
                )
                self.writer.add_scalar('validating/acc', acc_v, epoch)
                self.writer.add_scalar('validating/mIoU', iou_v, epoch)
                self.writer.add_scalar('validating/loss', loss_v, epoch)

                should_stop = self.early_stopping.update(iou_v, epoch)
                self.writer.add_scalar('validating/early_stopping_bad_epochs',
                                       self.early_stopping.bad_epochs, epoch)

                if iou_v > best_val_iou:
                    self.logger.info(
                        "Best mean IoU on validation so far, saving!")
                    self.logger.info("*" * 80)
                    best_val_iou = iou_v

                    self.best_val_epoch = epoch
                    self._save_checkpoint(epoch, best_train_iou, best_val_iou, is_best=True)
                self.logger.info(
                    f"Early stopping: current mIoU={iou_v:.6f} | best mIoU={best_val_iou:.6f} | "
                    f"best epoch={self.best_val_epoch} | "
                    f"bad epochs={self.early_stopping.bad_epochs}/{self.early_stopping.patience} | "
                    f"significant-improvement reference={self.early_stopping.best_score:.6f}"
                )
                if (self.early_stopping.enabled and
                        self.early_stopping.bad_epochs >= self.early_stopping.patience and
                        epoch < self.early_stopping.min_epochs):
                    self.logger.info(
                        f"Early stopping deferred: bad epochs={self.early_stopping.bad_epochs}/"
                        f"{self.early_stopping.patience}, but epoch {epoch} < "
                        f"min_epochs {self.early_stopping.min_epochs}; continuing."
                    )
                self.logger.info("*" * 80)

            self._save_checkpoint(epoch, best_train_iou, best_val_iou)
            self.writer.flush()
            if should_stop:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}\n"
                    f"Best epoch: {self.best_val_epoch}\n"
                    f"Best validation mIoU: {best_val_iou:.6f}"
                )
                break

        self.writer.close()
        if self.best_local_alpha is not None:
            self.logger.info(f"Best local alpha: {self.best_local_alpha:.6f} (epoch {self.best_val_epoch})")
        self.logger.info('Finished Training')
        return

    def _save_checkpoint(self, epoch, best_train_iou, best_val_iou, is_best=False):
        # Resume needs the training weights paired with the optimizer, not EMA weights.
        state = {
            'epoch': epoch,
            'model_name': self.ARCH['model']['name'],
            'experiment': experiment_id(self.ARCH),
            'arch_cfg': self.ARCH,
            'data_cfg': self.DATA,
            'early_stopping': self.early_stopping.state_dict(),
            'best_val_epoch': self.best_val_epoch,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_train_iou': best_train_iou,
            'best_val_iou': best_val_iou,
        }
        if self.use_ema and self.ema_model is not None:
            state['model_ema'] = self.ema_model.state_dict()
            if is_best:
                # Inference reads EMA from 'model'; resume still has raw weights.
                state['model_train'] = state['model']
                state['model'] = state['model_ema']
        alpha = local_alpha_parameter(self.model)
        if alpha is not None:
            eval_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
            state['local_alpha_train'] = alpha.detach().item()
            state['local_alpha_eval'] = local_alpha_parameter(eval_model).detach().item()
            if is_best:
                self.best_local_alpha = state['local_alpha_eval']
            state['best_local_alpha'] = self.best_local_alpha
        if spatial_alpha_module(self.model) is not None:
            if is_best:
                if not hasattr(self, 'last_validation_summary'):
                    raise RuntimeError("C.4 best checkpoint requires a completed validation summary")
                self.best_spatial_summary = copy.deepcopy(self.last_validation_summary)
            state['spatial_validation'] = getattr(self, 'last_validation_summary', None)
            state['best_spatial_summary'] = self.best_spatial_summary
        path = osp.join(self.log, 'best_val.path' if is_best else 'latest.path')
        torch.save(state, path + '.tmp')
        os.replace(path + '.tmp', path)
        self.logger.info(f"Saved checkpoint: {path} (epoch {epoch})")
        if alpha is not None and is_best:
            self.logger.info(f"Best local alpha: {self.best_local_alpha:.6f} (saved inference model)")
            log_best_comparison(self.logger, self.ARCH.get('experiment', {}),
                                getattr(self, 'last_class_iou', {}), best_val_iou)
        if spatial_alpha_module(self.model) is not None and is_best:
            save_spatial_best_summary(self.log, self.best_spatial_summary,
                                      self.ARCH.get('experiment', {}), self.logger)

    def _mix_losses(self, outs, labels, boundary_gt, proj_mask,paste_mask=None):
        # outs is dict from JunNet / ChatNet4
        logits = outs['logits'].float()
        loss = self.criterion_main(logits, labels)

        # aux
        if 'aux2' in outs:
            loss = loss + self.w_aux2 * self.criterion_main(outs['aux2'].float(), labels)
        if 'aux4' in outs:
            loss = loss + self.w_aux4 * self.criterion_main(outs['aux4'].float(), labels)
        if 'aux8' in outs:
            loss = loss + self.w_aux8 * self.criterion_main(outs['aux8'].float(), labels)

        # Lovasz on probs (ignore=0)
        probs = torch.softmax(logits, dim=1)
        loss = loss + self.w_lovasz * self.lovasz(probs, labels)

        # boundary: proj_mask で有効画素のみ平均
        if self.w_boundary > 0 and 'boundary' in outs and boundary_gt is not None:
            # boundary も float32 に
            bmap = self.boundary_criterion(outs['boundary'].float(), boundary_gt)

            if proj_mask.dim() == 3:
                pmask = proj_mask.unsqueeze(1).float()
            else:
                pmask = proj_mask.float()

            loss_b = (bmap * pmask).sum() / (pmask.sum() + 1e-6)
            loss = loss + self.w_boundary * loss_b

        # paste-region weighted loss
        if paste_mask is not None and self.w_paste > 0:
            logits = outs['logits'].float()
            paste_mask = paste_mask.float()

            if paste_mask.dim() == 4:
                paste_mask_2d = paste_mask.squeeze(1)
            else:
                paste_mask_2d = paste_mask

            if proj_mask.dim() == 4:
                valid_mask = proj_mask.squeeze(1).float()
            else:
                valid_mask = proj_mask.float()

            loss_map = self._focal_loss_map(logits, labels)

            paste_weight = paste_mask_2d * valid_mask
            paste_sum = paste_weight.sum()

            if paste_sum > 0:
                loss_paste = (loss_map * paste_weight).sum() / (paste_sum + 1e-6)
                loss = loss + self.w_paste * loss_paste

        return loss

    def train_epoch(self, train_loader, model, optimizer, epoch,
                    evaluator, color_fn, report=10):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        update_ratio_meter = AverageMeter()

        model.train()
        end = time.time()
        evaluator.reset()

        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                _, _, proj_range, _, _, _, _, _, paste_mask, edge) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if epoch == self.start_epoch and i < 3:
                if torch.count_nonzero(paste_mask).item() != 0:
                    raise RuntimeError("Copy-Paste OFF requires an empty paste_mask")
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(
                    non_blocking=True).long()
                paste_mask = paste_mask.cuda(non_blocking=True).float()

            # if np.random.rand() < 0.5:
            #     in_vol, proj_mask, proj_labels, paste_mask = apply_polarmix(in_vol, proj_mask, proj_labels, paste_mask)

            if proj_mask.dim() == 3:
                proj_mask_exp = proj_mask.unsqueeze(1).float()
            else:
                proj_mask_exp = proj_mask.float()

            if self.w_boundary > 0:
                boundary_gt = self._compute_boundary_gt(proj_labels)
                boundary_gt = boundary_gt * proj_mask_exp
                if self.gpu:
                    boundary_gt = boundary_gt.cuda(non_blocking=True)
            else:
                boundary_gt = None

            in_vol24 = torch.cat([in_vol, proj_mask_exp], dim=1)
            if in_vol24.shape[1] != self.expected_in_channels:
                raise RuntimeError(
                    f"Hybrid model input has {in_vol24.shape[1]} channels, "
                    f"expected {self.expected_in_channels}"
                )

            optimizer.zero_grad()
            # autocastブロックで囲む
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outs = model(in_vol24)
                loss = self._mix_losses(outs, proj_labels, boundary_gt, proj_mask,paste_mask=paste_mask)

            # スケーラーを使ってバックプロパゲーション
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            self.scaler.step(optimizer)
            self.scaler.update()

            # EMA 更新（float tensor のみ）
            self._update_ema()

            # metrics
            loss_val = loss.mean().item()
            with torch.no_grad():
                preds = outs['logits'].argmax(dim=1)
                max_z = in_vol[:, 0, :, :] * 5.0 + 1.0
                preds[(preds == 7) & (max_z < -0.8)] = 2  # 低い部分は bicycle
                preds[(preds == 8) & (max_z < -0.8)] = 3  # 低い部分は motorcycle
                evaluator.addBatch(preds, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, _ = evaluator.getIoU()

            losses.update(loss_val, in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            # update ratio stats
            update_mean = 0.0
            update_std = 0.0

            if i % report == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    'Lr: {lr:.3e} | '
                    'Epoch: [{ep}][{it}/{tot}] | '
                    'Loss {lcur:.4f} ({lavg:.4f}) | '
                    'acc {acur:.3f} ({aavg:.3f}) | '
                    'IoU {icur:.3f} ({iavg:.3f})'
                    .format(
                        lr=lr,
                        dcur=data_time.val, davg=data_time.avg,
                        bcur=batch_time.val, bavg=batch_time.avg,
                        ep=epoch, it=i, tot=len(train_loader),
                        lcur=losses.val, lavg=losses.avg,
                        acur=acc.val, aavg=acc.avg,
                        icur=iou.val, iavg=iou.avg
                    )
                )

            # if i % report == 0:
            #     update_ratios = []
            #     for g in optimizer.param_groups:
            #         lr = g["lr"]
            #         for value in g["params"]:
            #             if value.grad is not None:
            #                 w = value.data.norm().item()
            #                 upd = (max(lr, 1e-10) * value.grad).norm().item()
            #                 update_ratios.append(upd / max(w, 1e-10))

            # update_ratios = np.array(update_ratios) if len(update_ratios) else np.array([0.0])
            # update_mean = float(np.nanmean(update_ratios)) if len(update_ratios) else 0.0
            # update_std = float(np.nanstd(update_ratios)) if len(update_ratios) > 1 else 0.0
            update_ratio_meter.update(update_mean)

            # if i % report == 0:
            #     lr = self.optimizer.param_groups[0]["lr"]
            #     self.logger.info(
            #         'Lr: {lr:.3e} | Update: {umean:.3e} mean,{ustd:.3e} std | '
            #         'Epoch: [{ep}][{it}/{tot}] | Loss {lcur:.4f} ({lavg:.4f}) | '
            #         'acc {acur:.3f} ({aavg:.3f}) | IoU {icur:.3f} ({iavg:.3f})'
            #         .format(
            #             lr=lr, umean=update_mean, ustd=update_std,
            #             ep=epoch, it=i, tot=len(train_loader),
            #             lcur=losses.val, lavg=losses.avg,
            #             acur=acc.val, aavg=acc.avg,
            #             icur=iou.val, iavg=iou.avg))
            if i == 0:
                self.logger.info(f"in_vol shape: {in_vol.shape}")
                self.logger.info(f"model input shape: {in_vol24.shape}")
                self.logger.info(f"hand shape: {in_vol24[:, 0:7].shape}")
                self.logger.info(f"point shape: {in_vol24[:, 7:23].shape}")
                self.logger.info(f"mask shape: {in_vol24[:, 23:24].shape}")
                self.logger.info(f"proj_mask shape: {proj_mask.shape}")
                self.logger.info(f"proj_labels shape: {proj_labels.shape}")
                self.logger.info(f"paste_mask shape: {paste_mask.shape}")

            if i % report == 0 or (epoch == self.start_epoch and i < 3):
                self.logger.info(f"paste_mask sum: {paste_mask.sum().item():.1f}")

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg

    def validate(self, val_loader, model, evaluator, class_func):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        # 検証は EMA モデルを優先して使う
        eval_model = self.ema_model if (self.use_ema and
                                        self.ema_model is not None) else model
        eval_model.eval()
        evaluator.reset()
        spatial_stats = None
        if spatial_alpha_module(eval_model) is not None:
            class_names = {cid: class_func(cid) for cid in range(self.parser.get_n_classes())
                           if not self.DATA['learning_ignore'].get(cid, False)}
            spatial_stats = SpatialAlphaValidation(
                eval_model, class_names,
                min_class_cells=int(self.ARCH.get('diagnostics', {}).get('alpha_min_class_cells', 5)))

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                    _, _, proj_range, _, _, _, _, _, paste_mask, edge) in enumerate(val_loader):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                if proj_mask.dim() == 3:
                    proj_mask_exp = proj_mask.unsqueeze(1).float()
                else:
                    proj_mask_exp = proj_mask.float()

                if self.w_boundary > 0:
                    boundary_gt = self._compute_boundary_gt(proj_labels)
                    boundary_gt = boundary_gt * proj_mask_exp
                    if self.gpu:
                        boundary_gt = boundary_gt.cuda(non_blocking=True)
                else:
                    boundary_gt = None

                in_vol24 = torch.cat([in_vol, proj_mask_exp], dim=1)
                if in_vol24.shape[1] != self.expected_in_channels:
                    raise RuntimeError(
                        f"Hybrid model input has {in_vol24.shape[1]} channels, "
                        f"expected {self.expected_in_channels}"
                    )

                if i == 0:
                    with local3_feature_statistics(eval_model) as local_stats, \
                            local_alpha_statistics(eval_model) as alpha_stats, \
                            spatial_alpha_capture(eval_model, detailed=True) as spatial_batch, \
                            fusion_gate_statistics(eval_model) as gate_stats:
                        outs = eval_model(in_vol24)
                    if alpha_stats:
                        self.last_alpha_statistics = alpha_stats
                        self.logger.info(f"Local3 alpha diagnostic (validation batch 0): {alpha_stats}")
                        for name, stats in alpha_stats.items():
                            for key, value in (stats.items() if isinstance(stats, dict) else [("value", stats)]):
                                self.writer.add_scalar(f"validating/local_alpha_{name}_{key}", value,
                                                       getattr(self, 'current_epoch', self.start_epoch))
                    if local_stats:
                        self.last_local_statistics = local_stats
                        for name, stats in local_stats.items():
                            self.logger.info(f"Local3 (validation batch 0) [{name}]: {stats}")
                            for key, value in stats.items():
                                if value is not None:
                                    self.writer.add_scalar(f"validating/local3_{name}_{key}", value,
                                                           getattr(self, 'current_epoch', self.start_epoch))
                    self.last_gate_statistics = gate_stats
                    self.logger.info(
                        "Fusion Gate (validation batch 0): "
                        + " | ".join(f"{name} = {value:.6f}" for name, value in gate_stats.items())
                    )
                    for name, value in gate_stats.items():
                        self.writer.add_scalar(f"validating/fusion_gate_{name}", value,
                                               getattr(self, "current_epoch", self.start_epoch))
                else:
                    with spatial_alpha_capture(eval_model) as spatial_batch:
                        outs = eval_model(in_vol24)

                if spatial_stats is not None:
                    spatial_stats.update(spatial_batch, proj_labels, proj_mask_exp)

                loss = self._mix_losses(outs, proj_labels, boundary_gt, proj_mask,paste_mask=None)
                losses.update(loss.item(), in_vol.size(0))

                preds = outs['logits'].argmax(dim=1)
                max_z = in_vol[:, 0, :, :] * 5.0 + 1.0

                preds[(preds == 7) & (max_z < -0.8)] = 2
                preds[(preds == 8) & (max_z < -0.8)] = 3

                evaluator.addBatch(preds, proj_labels)

        acc_v = evaluator.getacc()
        iou_v, class_iou = evaluator.getIoU()
        self.logger.info(
            f"Validation: acc {acc_v.item():.3f} | mIoU {iou_v.item():.3f} | loss {losses.avg:.4f}")
        for cid, jacc in enumerate(class_iou):
            self.logger.info('IoU class {i} [{name}] = {j:.3f}'.format(
                i=cid, name=class_func(cid), j=jacc))
        self.last_class_iou = {class_func(cid): jacc.item() for cid, jacc in enumerate(class_iou)}
        if spatial_stats is not None:
            self.last_spatial_statistics = spatial_stats.summary()
            epoch = getattr(self, 'current_epoch', self.start_epoch)
            self.last_validation_summary = dict(
                epoch=epoch, miou=iou_v.item(), acc=acc_v.item(), loss=losses.avg,
                class_iou=dict(self.last_class_iou), spatial_alpha=self.last_spatial_statistics,
                evaluated_weights='EMA' if eval_model is self.ema_model else 'training',
            )
            log_spatial_validation(self.logger, self.writer, self.last_spatial_statistics, epoch)
        reference = self.ARCH.get("experiment", {})
        for name, previous in reference.get("step_c1_class_iou", {}).items():
            if name in self.last_class_iou:
                current = self.last_class_iou[name]
                self.logger.info(
                    f"Class comparison [{name}]: current={current:.6f} | "
                    f"Step C.1 epoch 23 (rounded)={previous:.3f} | delta={current - previous:+.6f}"
                )
        for name, previous in reference.get("step_c1b_class_iou", {}).items():
            if name in self.last_class_iou:
                current = self.last_class_iou[name]
                self.logger.info(
                    f"C.1b class comparison [{name}]: current={current:.6f} | "
                    f"Step C.1b epoch 9 (rounded)={previous:.3f} | delta={current - previous:+.6f}"
                )
        for name in ("step_c1_miou", "step_c1b_miou", "baseline_8ch_miou", "step_c2_miou", "step_c3_miou"):
            if name in reference:
                self.logger.info(
                    f"mIoU comparison [{name}]: current={iou_v.item():.6f} | "
                    f"reference={reference[name]:.6f} | delta={iou_v.item() - reference[name]:+.6f}"
                )
        self._log_local_alpha()
        return acc_v.item(), iou_v.item(), losses.avg
