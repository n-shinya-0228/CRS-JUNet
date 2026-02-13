# -*- coding: utf-8 -*-
# ChatNet5: Light FPN + Boundary-Guided Refinement
# in:  (B, 6, H, W)  = [range, x, y, z, remission, obs_mask]
# out: dict {'logits','aux2','aux4','boundary'} with HxW resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -------------------- building blocks --------------------
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True, d=1):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0, use_se=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.se = SE(out_ch) if use_se else nn.Identity()

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        idt = x
        x = self.act(self.bn1(x))
        x = self.conv1(x)
        x = self.act(self.bn2(x))
        x = self.drop(x)
        x = self.conv2(x)
        x = self.se(x)
        if self.down is not None:
            idt = self.down(idt)
        x = x + idt
        x = self.act(x)
        return x


class ResStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2, pool=True, drop=0.0, use_se=True):
        super().__init__()
        blocks = []
        stride = 2 if pool else 1
        blocks.append(BasicBlock(in_ch, out_ch, stride=stride, drop=drop, use_se=use_se))
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1, drop=drop, use_se=use_se))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class LKA(nn.Module):
    def __init__(self, ch, k=7, d=3):
        super().__init__()
        self.dw1 = nn.Conv2d(ch, ch, kernel_size=k, padding=k // 2, groups=ch)
        self.dw2 = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        u = x
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.pw(x)
        return x * u


# Swin-like window attention (no shift)
class WindowAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: (B,H,W,C) window
        B, H, W, C = x.shape
        N = H * W
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (q.shape[-1] ** 0.5))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out).reshape(B, H, W, C)
        return out


class SWABlock(nn.Module):
    def __init__(self, ch, window: Tuple[int, int] = (8, 16), heads=8):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(ch)
        self.attn = WindowAttention(ch, heads=heads)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(
            nn.Linear(ch, 4 * ch),
            nn.GELU(),
            nn.Linear(4 * ch, ch),
        )

    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        Ph, Pw = self.window

        # pad to window
        pad_h = (Ph - H % Ph) % Ph
        pad_w = (Pw - W % Pw) % Pw
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        # window partition
        x = (
            x.view(B, Hp // Ph, Ph, Wp // Pw, Pw, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )  # (B,nH,nW,Ph,Pw,C)
        x = x.view(-1, Ph, Pw, C)

        # attn
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        # merge
        x = (
            x.view(B, Hp // Ph, Wp // Pw, Ph, Pw, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        x = x.view(B, Hp, Wp, C)[:, :H, :W, :]
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=1, s=1, p=0))
            else:
                self.branches.append(
                    conv_bn_act(in_ch, out_ch, k=3, s=1, p=r, d=r)
                )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_bn_act(in_ch, out_ch, k=1, s=1, p=0),
        )
        self.proj = conv_bn_act(
            out_ch * (len(rates) + 1), out_ch, k=1, s=1, p=0
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(img)
        x = torch.cat(feats, dim=1)
        return self.proj(x)


class AttnGate(nn.Module):
    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, skip, gate):
        g = F.interpolate(
            self.phi(gate),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s = self.theta(skip)
        a = torch.sigmoid(self.psi(F.relu(s + g, inplace=True)))
        return skip * a


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.gate = AttnGate(
            skip_ch, in_ch, inter_ch=min(skip_ch, in_ch) // 2
        )
        self.conv1 = conv_bn_act(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = conv_bn_act(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        skip = self.gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


# -------------------- main net --------------------
class JunNet11(nn.Module):
    """
    in_channels: 6  (range, x, y, z, remission, obs_mask)
    nclasses:     20 (SemanticKITTI xentropy IDs)
    """

    def __init__(
        self,
        in_channels: int = 6,
        nclasses: int = 20,
        drop: float = 0.2,
        base_ch: int = 48,
        aspp_out: int = 384,
        swa_blocks: int = 2,
        swa_heads: int = 8,
        swa_window: Tuple[int, int] = (8, 16),
        return_aux: bool = True,
    ):
        super().__init__()
        self.return_aux = return_aux

        # ----- Mask-Aware Stem (ChatNet4 と同じ) -----
        self.stem_feat = conv_bn_act(5, base_ch, 3, 1, 1)
        self.stem_mask = nn.Sequential(
            conv_bn_act(1, base_ch // 2, 1, 1, 0),
            conv_bn_act(base_ch // 2, base_ch, 3, 1, 1),
            nn.Conv2d(base_ch, base_ch, 1),
        )

        # encoder
        self.enc1 = ResStage(
            base_ch, base_ch * 2, num_blocks=2, pool=True, drop=drop
        )  # 1/2
        self.enc2 = ResStage(
            base_ch * 2, base_ch * 4, num_blocks=2, pool=True, drop=drop
        )  # 1/4
        self.enc3 = ResStage(
            base_ch * 4, base_ch * 8, num_blocks=2, pool=True, drop=drop
        )  # 1/8

        # bottleneck (ASPP + Swin-like + LKA はほぼそのまま)
        self.aspp = ASPP(base_ch * 8, aspp_out)
        # self.swa = nn.Sequential(
        #     *[SWABlock(aspp_out, window=swa_window, heads=swa_heads) for _ in range(swa_blocks)]
        # )
        self.swa = conv_bn_act(aspp_out, aspp_out)
        self.bottleneck_proj = conv_bn_act(aspp_out, base_ch * 8, 1, 1, 0)
        self.lka = LKA(base_ch * 8, k=7, d=3)

        # decoder
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, drop=drop)  # 1/4
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, drop=drop)  # 1/2
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, drop=drop)          # 1/1

        # ----- New: 軽量 FPN で d2(1/2) と d1(1/1) を融合 -----
        # d1: base_ch, d2: base_ch*2 -> concat で 3*base_ch
        self.fpn_fuse = conv_bn_act(base_ch + base_ch * 2, base_ch, 3, 1, 1)

        # aux heads (そのまま保持)
        self.aux4_head = nn.Conv2d(base_ch * 4, nclasses, 1)
        self.aux2_head = nn.Conv2d(base_ch * 2, nclasses, 1)

        # strip pooling (ChatNet4 と同じ)
        self.fuse = conv_bn_act(base_ch, base_ch, 3, 1, 1)
        self.strip_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)), conv_bn_act(base_ch, base_ch, 1, 1, 0)
        )
        self.strip_w = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), conv_bn_act(base_ch, base_ch, 1, 1, 0)
        )
        self.strip_fuse = conv_bn_act(base_ch * 3, base_ch, 1, 1, 0)

        # ----- New: Boundary-guided refinement -----
        self.boundary_head = nn.Conv2d(base_ch, 1, 1)
        # boundary map から注意マップを作り、特徴を再スケーリングする
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),  # 0~1 の係数
        )

        self.final_logits = nn.Conv2d(base_ch, nclasses, 1)

    def forward(self, x):
        # x: (B,6,H,W)  -> split
        feat_in = x[:, :5, :, :]
        m = x[:, 5:6, :, :]
        B, C_in, H, W = x.shape

        # ----- mask-aware stem -----
        s0 = self.stem_feat(feat_in)       # (B, base_ch, H, W)
        g = torch.tanh(self.stem_mask(m))  # (B, base_ch, H, W)
        s0 = s0 * (1.0 + g)                # 観測マスクでゲート

        # ----- encoder -----
        s1 = self.enc1(s0)   # 1/2
        s2 = self.enc2(s1)   # 1/4
        s3 = self.enc3(s2)   # 1/8

        # ----- bottleneck -----
        b = self.aspp(s3)
        # b = self.swa(b)
        b = self.bottleneck_proj(b)
        b = self.lka(b)

        # ----- decoder -----
        d3 = self.up3(b, s2)  # 1/4
        d2 = self.up2(d3, s1) # 1/2
        d1 = self.up1(d2, s0) # 1/1

        # Aux heads
        aux4 = F.interpolate(
            self.aux4_head(d3),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        aux2 = F.interpolate(
            self.aux2_head(d2),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # ----- New: 軽量 FPN で d2, d1 を融合 -----
        d2_up = F.interpolate(
            d2, size=d1.shape[-2:], mode="bilinear", align_corners=False
        )
        fpn = self.fpn_fuse(torch.cat([d1, d2_up], dim=1))  # (B, base_ch, H, W)

        # strip pooling
        feat = self.fuse(fpn)
        sh = F.interpolate(
            self.strip_h(feat), size=(H, W), mode="bilinear", align_corners=False
        )
        sw = F.interpolate(
            self.strip_w(feat), size=(H, W), mode="bilinear", align_corners=False
        )
        feat = self.strip_fuse(torch.cat([feat, sh, sw], dim=1))

        # ----- boundary + refinement -----
        boundary_logit = self.boundary_head(feat)
        # 少し平滑にした boundary をゲートに使う
        b_gate = self.boundary_refine(boundary_logit)      # (B,1,H,W) 0~1
        feat_refined = feat * (1.0 + b_gate)               # 境界付近をやや強調

        # final logits
        logits = self.final_logits(feat_refined)

        # アウトプットは ChatNet4 と同じ形式
        return {
            "logits": logits,
            "aux2": aux2,
            "aux4": aux4,
            "boundary": F.interpolate(
                boundary_logit, size=(H, W), mode="bilinear", align_corners=False
            ),
        }
