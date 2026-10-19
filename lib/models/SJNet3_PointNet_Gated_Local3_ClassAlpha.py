"""D.2 evaluation: external, non-learnable residual scales; unchanged C.2 keys."""

import torch

from .SJNet3_PointNet_Gated_Local3_FixedAlpha import (
    PointNetLocal3FixedAlpha, SJNet3_PointNet_Gated_Local3_FixedAlpha,
)

CLASS_NAMES = (
    "unlabeled", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
    "pole", "traffic-sign",
)
ALPHA_TABLE = (1.2, 1.0, 1.6, 1.2, 1.6, 1.6, .6, .8, 1.2, .8,
               .6, .8, 1.6, 1.1, 1.6, 1.6, 1.0, 1.6, 1.3, 1.3)


def build_alpha_maps(stage1_logits, observation_mask):
    """Both assignments use the same RAW logits; no labels/postprocessing."""
    if stage1_logits.ndim != 4 or stage1_logits.shape[1] != 20:
        raise ValueError("Expected main semantic logits [B,20,H,W]")
    if not torch.isfinite(stage1_logits).all():
        raise ValueError("Non-finite stage1 logits")
    if observation_mask.ndim == 3:
        observation_mask = observation_mask.unsqueeze(1)
    shape = (stage1_logits.shape[0], 1, *stage1_logits.shape[2:])
    if observation_mask.shape != shape or observation_mask.device != stage1_logits.device:
        raise ValueError("Observation mask must match [B,1,H,W] and logits device")
    if not torch.isfinite(observation_mask).all():
        raise ValueError("Non-finite observation mask")
    table = stage1_logits.new_tensor(ALPHA_TABLE)
    hard = table[stage1_logits.argmax(dim=1)].unsqueeze(1)
    soft = (stage1_logits.softmax(dim=1) * table.view(1, 20, 1, 1)).sum(dim=1, keepdim=True)
    observed = observation_mask > 0
    return torch.where(observed, hard, 1.2), torch.where(observed, soft, 1.2)


class PointNetLocal3ClassAlpha(PointNetLocal3FixedAlpha):
    def __init__(self, local, alpha=1.2):
        super().__init__(local, alpha)
        self._alpha_map = None

    def forward(self, point_base):
        scale = self.fixed_alpha if self._alpha_map is None else self._alpha_map
        return point_base + scale * self.local(point_base)


class SJNet3_PointNet_Gated_Local3_ClassAlpha(SJNet3_PointNet_Gated_Local3_FixedAlpha):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, alpha=1.2, **kwargs)
        self.point_context = PointNetLocal3ClassAlpha(self.point_context.local)

    def forward(self, x, alpha_map=None):
        assert x.ndim == 4 and x.shape[1] == 24, "Expected [B,24,H,W]"
        if self.point_context._alpha_map is not None:
            raise RuntimeError("Overlapping external-alpha forwards are not supported")
        if alpha_map is not None:
            if (alpha_map.shape != (x.shape[0], 1, *x.shape[2:]) or
                    alpha_map.device != x.device or alpha_map.dtype != x.dtype or
                    alpha_map.requires_grad or not torch.isfinite(alpha_map).all()):
                raise ValueError("Expected finite, non-learnable [B,1,H,W] alpha map matching input")
        self.point_context._alpha_map = alpha_map
        try:
            return super().forward(x)
        finally:
            self.point_context._alpha_map = None
