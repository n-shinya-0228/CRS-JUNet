"""Step C.2: one residual Local 3x3 block on the PointNet branch only."""

import torch
from torch import nn
import torch.nn.functional as F

from .SJNet3_PointNet import RingConv2d
from .SJNet3_PointNet_Gated import SJNet3_PointNet_Gated


class PointNetLocal3(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(
            RingConv2d(16, 16, kernel_size=3, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, point_base):
        return point_base + self.local(point_base)


class SJNet3_PointNet_Gated_Local3(SJNet3_PointNet_Gated):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the new block after the unchanged C.1b parameters.
        self.point_context = PointNetLocal3()

    def forward(self, x):
        assert x.ndim == 4, f"Expected [B, 24, H, W], got {tuple(x.shape)}"
        assert x.shape[1] == 24, f"Expected 24 channels, got {x.shape[1]}"
        hand = x[:, 0:7]
        point = x[:, 7:23]
        m = x[:, 23:24]

        hand_feat = self.hand_stem(hand)
        point_base = self.point_stem(point)
        point_local = self.point_context(point_base)
        gate = self.point_gate(torch.cat([hand_feat, point_local], dim=1))
        point_gated = point_local * gate
        s0 = self.stem_feat(torch.cat([hand_feat, point_gated], dim=1))

        # Preserve Step C's observation gate and skip paths exactly. Its
        # gate_scale parameter is unused, and no extra hard mask is applied.
        g = torch.tanh(self.stem_mask(m))
        s0 = s0 * (1.0 + g)

        B, C, H, W = x.shape
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.aspp(s4)
        b = self.bottleneck_proj(b)
        b = self.lka(b)
        d4 = self.up4(b, s3)
        d3 = self.up3(d4, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)

        aux8 = F.interpolate(self.aux8_head(s4), size=(H, W), mode="bilinear", align_corners=False)
        aux4 = F.interpolate(self.aux4_head(d3), size=(H, W), mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2_head(d2), size=(H, W), mode="bilinear", align_corners=False)

        feat = self.fuse(d1)
        sh = F.interpolate(self.strip_h(feat), size=(H, W), mode="bilinear", align_corners=False)
        sw = F.interpolate(self.strip_w(feat), size=(H, W), mode="bilinear", align_corners=False)
        feat = self.strip_fuse(torch.cat([feat, sh, sw], dim=1))

        boundary = F.interpolate(self.boundary_head(feat), size=(H, W), mode="bilinear", align_corners=False)
        logits = self.final_logits(feat)
        return {"logits": logits, "aux2": aux2, "aux4": aux4, "aux8": aux8, "boundary": boundary}
