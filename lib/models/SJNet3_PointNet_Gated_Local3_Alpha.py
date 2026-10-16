"""Step C.3: the C.2 residual with one unconstrained, learnable scalar."""

import torch
from torch import nn

from .SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3


class PointNetLocal3Alpha(nn.Module):
    def __init__(self, local):
        super().__init__()
        # Reuse the initialized C.2 block without drawing additional random weights.
        self.local = local
        self.local_alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, point_base):
        local_residual = self.local(point_base)
        return point_base + self.local_alpha * local_residual


class SJNet3_PointNet_Gated_Local3_Alpha(SJNet3_PointNet_Gated_Local3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_context = PointNetLocal3Alpha(self.point_context.local)

    @property
    def local_alpha(self):
        return self.point_context.local_alpha
