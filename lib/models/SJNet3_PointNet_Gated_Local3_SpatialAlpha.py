"""Step C.4: global residual scale plus a feature-dependent scalar per cell."""

import torch
from torch import nn

from .SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3


class SpatialAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_global = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        # The zero-initialized predictor must not advance C.2's random stream.
        with torch.random.fork_rng(devices=[]):
            self.alpha_predictor = nn.Conv2d(16, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.alpha_predictor.weight)

    def forward(self, point_base):
        return self.alpha_global + self.alpha_predictor(point_base)


class PointNetLocal3SpatialAlpha(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        self.alpha = SpatialAlpha()

    def forward(self, point_base):
        local_residual = self.local(point_base)
        alpha_map = self.alpha(point_base)
        return point_base + alpha_map * local_residual


class SJNet3_PointNet_Gated_Local3_SpatialAlpha(SJNet3_PointNet_Gated_Local3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_context = PointNetLocal3SpatialAlpha(self.point_context.local)

    @property
    def alpha_global(self):
        return self.point_context.alpha.alpha_global

    @property
    def alpha_predictor(self):
        return self.point_context.alpha.alpha_predictor
