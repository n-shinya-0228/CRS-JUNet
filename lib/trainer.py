#!/usr/bin/env python3
# Trainer (cleaned) tailored for JunNetBest / ChatNet4
# - expects model(in_vol) -> dict with keys: 'logits', 'aux2', 'aux4', 'boundary'
# - combines CE/Focal + Lovasz (optional) + Boundary BCE + Aux losses
# - keeps scheduler/optimizer structure + EMA model for better mIoU

#trainer_Polar7
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
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, StepLR, LambdaLR
import copy  # EMA 用

from .utils.avgmeter import *
from .utils.warmupLR import *   # (互換性のために import のみ)
from .utils.ioueval import *
from .dataset.Parser import Parser
from .models import *           # get_model(...)
from .losses import *           # Lovasz_softmax / Depth_Loss など


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


def apply_polarmix(in_vol, proj_mask, proj_labels):
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

    return mixed_vol, mixed_mask, mixed_labels


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, logger, pretrained=None, use_mps=True, resume=None):
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
        dataset_name = 'SemanticKitti_Polar3.py' # ファイル名変更
        dataset_py_path = osp.join('lib', 'dataset', dataset_name)

        used_files_txt = osp.join(self.log, 'used_files.txt')
        with open(used_files_txt, 'w') as f:
            f.write("=== Training Execution Log ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")
            f.write("-" * 30 + "\n")
            f.write("[Used Files]\n")
            f.write(f"Config YAML : {yaml_path if yaml_path else 'Not specified'}\n")
            f.write(f"Model File  : {model_py_path}\n")
            f.write(f"Dataset File: {dataset_py_path}\n")
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
        self.model = get_model(ARCH['model']['name'])(
            nclasses=self.parser.get_n_classes(),
            drop=self.ARCH["model"].get("dropout", 0.5)  
        )
        self.model = torch.compile(self.model, backend="inductor", mode="default")
        weights_total = sum(p.numel() for p in self.model.parameters())
        weights_grad = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
        self.logger.info("Total number of parameters: " + str(weights_total))
        self.logger.info("Total number of parameters requires_grad: " +
                         str(weights_grad))

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
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.ARCH["train"]["lr"],
                                     weight_decay=self.ARCH["train"]["w_decay"])

        # Scheduler: warmup -> const -> cosine
        up_epochs = int(self.ARCH["train"]["wup_epochs"])
        max_epochs = int(self.ARCH["train"]["max_epochs"])
        down_epochs = int(self.ARCH["train"]["cos_epochs"])
        stay_epochs = max_epochs - up_epochs - down_epochs

        warmup_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: 0.1 + 0.9 * (epoch / up_epochs) 
            if epoch < max(up_epochs, 1) else 1.0)

        const_scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=max(stay_epochs, 1), gamma=1.0)

        cosine_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=max(down_epochs, 1),
            eta_min=self.ARCH["train"]["lr"] * 0.00001)

        self.scheduler = SequentialLR(
            optimizer=self.optimizer,
            schedulers=[warmup_scheduler, const_scheduler, cosine_scheduler],
            milestones=[up_epochs, up_epochs + stay_epochs])

        # (optional) resume
        self.start_epoch = 0
        if self.resume is not None:
            try:
                try:
                    torch.serialization.add_safe_globals([torch.optim.lr_scheduler.CyclicLR])
                except AttributeError:
                    pass
                w_dict = torch.load(self.resume, weights_only=False)
                
                self.model.load_state_dict(w_dict['model'])
                self.optimizer.load_state_dict(w_dict['optim'])
                self.scheduler.load_state_dict(w_dict['scheduler'])
                self.start_epoch = w_dict.get('epoch', 0) + 1
                
                if self.use_ema:
                    self._build_ema_model()
                    if 'model_ema' in w_dict:
                        self.ema_model.load_state_dict(w_dict['model_ema'])
                self.logger.info(f"✅ Successfully RESUMED training from {self.resume}. Starting at epoch {self.start_epoch}")
            except Exception as e:
                self.logger.warning(f"❌ Couldn't resume from {self.resume}. Error: {e}")

        elif self.pretrained is not None:
            try:
                w_dict = torch.load(self.pretrained, weights_only=False)
                state_dict = w_dict['model'] if 'model' in w_dict else w_dict
                
                self.model.load_state_dict(state_dict, strict=False)
                
                if self.use_ema:
                    self._build_ema_model()
                self.logger.info(f"✅ Successfully loaded PRETRAINED weights from {self.pretrained}. Starting from scratch.")
            except Exception as e:
                self.logger.warning(f"❌ Couldn't load pretrained parameters. Error: {e}")

        # Loss weights (tunable)
        self.w_aux2 = 0.30
        self.w_aux4 = 0.15
        self.w_aux8 = 0.10
        self.w_lovasz = 0.50
        self.w_boundary = 0.20  

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

    def train(self):
        best_train_iou = 0.0
        best_val_iou = 0.0

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                self.logger.info(f"Ignoring class {i} in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

        # train & validate
        for epoch in range(self.start_epoch,
                           self.ARCH["train"]["max_epochs"]):
            
            # Epoch 60 に到達したら Auxiliary Loss を無効化する
            if epoch == 60:
                self.logger.info("*" * 80)
                self.logger.info(f"[Epoch {epoch}] Disabling Auxiliary Losses (w_aux2, w_aux4 -> 0.0) for final fine-tuning!")
                self.logger.info("*" * 80)
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

                if iou_v > best_val_iou:
                    self.logger.info(
                        "Best mean IoU on validation so far, saving!")
                    self.logger.info("*" * 80)
                    best_val_iou = iou_v

                    # checkpoint: EMA を保存（あれば）、下流は 'model' を使う
                    state = {
                        'epoch': epoch,
                        'optim': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'model': self.model.state_dict(),
                    }
                    if self.use_ema and self.ema_model is not None:
                        state['model_ema'] = self.ema_model.state_dict()
                        # test/infer 側は 'model' を読むので、そこも EMA にすり替える
                        state['model'] = self.ema_model.state_dict()

                    torch.save(state, osp.join(self.log, 'best_val.path'))
                self.logger.info("*" * 80)

        self.logger.info('Finished Training')
        return

    def _mix_losses(self, outs, labels, boundary_gt, proj_mask):
        # outs is dict from JunNet / ChatNet4
        logits = outs['logits']
        loss = self.criterion_main(logits, labels)

        # aux
        if 'aux2' in outs:
            loss = loss + self.w_aux2 * self.criterion_main(
                outs['aux2'], labels)
        if 'aux4' in outs:
            loss = loss + self.w_aux4 * self.criterion_main(
                outs['aux4'], labels)
        if 'aux8' in outs:
            loss = loss + self.w_aux8 * self.criterion_main(
                outs['aux8'], labels)

        # Lovasz on probs (ignore=0)
        probs = torch.softmax(logits, dim=1)
        loss = loss + self.w_lovasz * self.lovasz(probs, labels)

        # boundary: proj_mask で有効画素のみ平均
        if self.w_boundary > 0 and 'boundary' in outs and boundary_gt is not None: # ★ 条件を追加
            bmap = self.boundary_criterion(
                outs['boundary'], boundary_gt)  # (B,1,H,W), reduction='none'

             # ★ マスクの次元合わせ
            if proj_mask.dim() == 3:
                pmask = proj_mask.unsqueeze(1).float()
            else:
                pmask = proj_mask.float()
           
            loss_b = (bmap * pmask).sum() / (pmask.sum() + 1e-6)
            loss = loss + self.w_boundary * loss_b

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
                _, _, proj_range, _, _, _, _, _, _, edge) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(
                    non_blocking=True).long()
            
            if np.random.rand() < 0.5: 
                in_vol, proj_mask, proj_labels = apply_polarmix(in_vol, proj_mask, proj_labels)

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
     
            in_vol8 = torch.cat([in_vol, proj_mask_exp], dim=1)

            train_labels = proj_labels.clone()
            train_labels[train_labels == 2] = 7  # bicycle(2) を bicyclist(7) に統合
            train_labels[train_labels == 3] = 8  # motorcycle(3) を motorcyclist(8) に統合
        
            optimizer.zero_grad()
            # autocastブロックで囲む
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outs = model(in_vol8)
                loss = self._mix_losses(outs, proj_labels, boundary_gt, proj_mask)

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
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(
                            value.data.detach().cpu().numpy().reshape((-1)))
                        upd = np.linalg.norm(
                            (-max(lr, 1e-10) * value.grad).detach().cpu().numpy().reshape((-1)))
                        update_ratios.append(upd / max(w, 1e-10))
            update_ratios = np.array(update_ratios) if len(
                update_ratios) else np.array([0.0])
            update_mean = float(np.nanmean(update_ratios)) if len(update_ratios) else 0.0
            update_std = float(np.nanstd(update_ratios)) if len(update_ratios) > 1 else 0.0
            update_ratio_meter.update(update_mean)

            if i % report == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    'Lr: {lr:.3e} | Update: {umean:.3e} mean,{ustd:.3e} std | '
                    'Epoch: [{ep}][{it}/{tot}] | Loss {lcur:.4f} ({lavg:.4f}) | '
                    'acc {acur:.3f} ({aavg:.3f}) | IoU {icur:.3f} ({iavg:.3f})'
                    .format(
                        lr=lr, umean=update_mean, ustd=update_std,
                        ep=epoch, it=i, tot=len(train_loader),
                        lcur=losses.val, lavg=losses.avg,
                        acur=acc.val, aavg=acc.avg,
                        icur=iou.val, iavg=iou.avg))

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

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                    _, _, proj_range, _, _, _, _, _, _, edge) in enumerate(val_loader):
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

                in_vol8 = torch.cat([in_vol, proj_mask_exp], dim=1)

                outs = model(in_vol8)    

                loss = self._mix_losses(outs, proj_labels, boundary_gt, proj_mask)
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
        return acc_v.item(), iou_v.item(), losses.avg