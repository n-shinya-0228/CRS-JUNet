import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

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

class C2Recipe:
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

