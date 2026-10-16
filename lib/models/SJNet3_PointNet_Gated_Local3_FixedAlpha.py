"""D.1 evaluation wrapper; exactly the C.2 state_dict, with a plain float scale."""

import math
from torch import nn

from .SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3


class PointNetLocal3FixedAlpha(nn.Module):
    def __init__(self, local, alpha=1.0):
        super().__init__()
        self.local = local
        self.set_alpha(alpha)

    def set_alpha(self, alpha):
        alpha = float(alpha)
        if not math.isfinite(alpha):
            raise ValueError("fixed_alpha must be finite")
        self.fixed_alpha = alpha

    def forward(self, point_base):
        local_residual = self.local(point_base)
        return point_base + self.fixed_alpha * local_residual


class SJNet3_PointNet_Gated_Local3_FixedAlpha(SJNet3_PointNet_Gated_Local3):
    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_context = PointNetLocal3FixedAlpha(self.point_context.local, alpha)

    def set_alpha(self, alpha):
        self.point_context.set_alpha(alpha)

    @property
    def fixed_alpha(self):
        return self.point_context.fixed_alpha
