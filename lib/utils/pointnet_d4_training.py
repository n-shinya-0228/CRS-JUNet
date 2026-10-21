"""Unmodified C.2 StableLR scheduling/stopping helpers (historical reference)."""

import math
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR


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
