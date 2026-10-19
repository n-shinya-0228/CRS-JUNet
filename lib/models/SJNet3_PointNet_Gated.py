"""Step C.1: separate feature stems with gated PointNet fusion."""

import torch
from torch import nn
import torch.nn.functional as F

from .SJNet3_PointNet import RingConv2d, SJNet3_PointNet


class SJNet3_PointNet_Gated(SJNet3_PointNet):
    def __init__(
        self, in_channels=24, nclasses=20, drop=0.5, base_ch=48,
        aspp_out=256, swa_heads=4, swa_window=(8, 8), return_aux=True,
    ):
        super().__init__(
            in_channels=in_channels, nclasses=nclasses, drop=drop,
            base_ch=base_ch, aspp_out=aspp_out, swa_heads=swa_heads,
            swa_window=swa_window, return_aux=return_aux,
        )
        self.hand_stem = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
        )
        self.point_stem = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
        )
        self.point_gate = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Sigmoid(),
        )
        # Keep the default Conv2d weight initialization; start with no gate bias.
        nn.init.zeros_(self.point_gate[0].bias)
        self.stem_feat = nn.Sequential(
            RingConv2d(32, base_ch, kernel_size=3, bias=False),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        assert x.ndim == 4, f"Expected [B, 24, H, W], got {tuple(x.shape)}"
        assert x.shape[1] == 24, f"Expected 24 channels, got {x.shape[1]}"
        hand = x[:, 0:7]
        point = x[:, 7:23]
        m = x[:, 23:24]

        hand_feat = self.hand_stem(hand)
        point_feat = self.point_stem(point)
        gate = self.point_gate(torch.cat([hand_feat, point_feat], dim=1))
        point_gated = point_feat * gate
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
