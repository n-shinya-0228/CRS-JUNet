# lib/models/JunNet12.py
# -*- coding: utf-8 -*-
# JunNet12: 8ch 入力 / Encoder-Decoder + Swin-like bottleneck + Boundary head
# in:  (B, 8, H, W) = [range, x, y, z, remission, obs_mask, height, groundness]
# out: dict {'logits','aux2','aux4','boundary'} with HxW resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0, use_se=False):
        super().__init__()
        self.conv1 = conv_bn_act(in_ch, out_ch, 3, stride, 1)
        self.conv2 = conv_bn_act(out_ch, out_ch, 3, 1, 1, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.se = SE(out_ch) if use_se else nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.down = conv_bn_act(in_ch, out_ch, 1, stride, 0, act=False)
        else:
            self.down = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.se(out)
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        out = self.relu(out)
        return out


class ResStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2, pool=True, drop=0.0, use_se=False):
        super().__init__()
        layers = []
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        layers.append(BasicBlock(in_ch, out_ch, stride=1, drop=drop, use_se=use_se))
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1, drop=drop, use_se=use_se))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


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


class WindowAttention(nn.Module):
    # シンプルな全ウィンドウ self-attention（実装を軽くするためウィンドウ=全体扱い）
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: (B,H,W,C)
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
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

    def forward(self, x):  # x: (B,C,H,W)
        B, C, H, W = x.shape
        Ph, Pw = self.window
        pad_h = (Ph - H % Ph) % Ph
        pad_w = (Pw - W % Pw) % Pw
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.permute(0, 2, 3, 1).contiguous()   # (B,Hp,Wp,C)

        x_reshaped = x.view(B, Hp // Ph, Ph, Wp // Pw, Pw, C)
        x_reshaped = x_reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x_reshaped.view(-1, Ph, Pw, C)

        windows = self.norm1(windows)
        windows = self.attn(windows)
        windows = self.norm2(windows)
        windows = self.mlp(windows)

        x_reshaped = windows.view(B, Hp // Ph, Wp // Pw, Ph, Pw, C)
        x_reshaped = x_reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x_reshaped.view(B, Hp, Wp, C)
        x = x[:, :H, :W, :].contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ASPP(nn.Module):
    """
    ASPP (dilation rates を控えめにして「高さが小さい」特徴マップでも safe)
    rates=(1,2,4) 固定。
    """
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=1, s=1, p=0))
            else:
                # k=3, dilation=r, padding=r
                self.branches.append(conv_bn_act(in_ch, out_ch, k=3, s=1, p=r, d=r))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_bn_act(in_ch, out_ch, k=1, s=1, p=0),
        )
        self.proj = conv_bn_act(out_ch * (len(rates) + 1), out_ch, k=1, s=1, p=0)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(img)
        x = torch.cat(feats, dim=1)
        x = self.proj(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = conv_bn_act(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.conv2 = conv_bn_act(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BoundaryHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_act(in_ch, in_ch, 3, 1, 1),
            nn.Conv2d(in_ch, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class JunNet12(nn.Module):
    """
    Encoder-Decoder + Swin-like bottleneck + Boundary head
    入力:  (B, 8, H, W) = [range, x, y, z, remission, mask, height, groundness]
    出力:  dict(logits, aux2, aux4, boundary)
    """

    def __init__(self, nclasses: int = 20, base_ch: int = 32, swa_blocks: int = 2):
        super().__init__()
        self.nclasses = nclasses

        # stem (8ch -> base_ch)
        self.stem = conv_bn_act(10, base_ch, k=3, s=1, p=1)

        # encoder
        self.enc1 = ResStage(base_ch, base_ch * 2, num_blocks=2, pool=True, drop=0.0, use_se=False)
        self.enc2 = ResStage(base_ch * 2, base_ch * 4, num_blocks=2, pool=True, drop=0.0, use_se=False)
        self.enc3 = ResStage(base_ch * 4, base_ch * 8, num_blocks=2, pool=True, drop=0.0, use_se=True)

        # bottleneck
        self.aspp = ASPP(base_ch * 8, base_ch * 8, rates=(1, 2, 4))
        self.swa = nn.Sequential(*[SWABlock(base_ch * 8, window=(8, 16), heads=8) for _ in range(swa_blocks)])
        self.lka = LKA(base_ch * 8, k=7, d=3)

        # decoder
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # heads
        self.aux4_head = nn.Conv2d(base_ch * 4, nclasses, 1)
        self.aux2_head = nn.Conv2d(base_ch * 2, nclasses, 1)

        # strip pooling
        self.strip_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            conv_bn_act(base_ch, base_ch, 1, 1, 0),
        )
        self.strip_w = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            conv_bn_act(base_ch, base_ch, 1, 1, 0),
        )
        self.strip_fuse = conv_bn_act(base_ch * 3, base_ch, 1, 1, 0)

        self.boundary_head = BoundaryHead(base_ch)
        self.final_head = nn.Conv2d(base_ch, nclasses, 1)

    def forward(self, x):
        """
        x: (B,8,H,W)
        returns dict: logits, aux2, aux4, boundary
        """
        B, C, H, W = x.shape

        xs = self.stem(x)          # (B,base,H,W)

        e1 = self.enc1(xs)         # (B,2b,H/2,W/2)
        e2 = self.enc2(e1)         # (B,4b,H/4,W/4)
        e3 = self.enc3(e2)         # (B,8b,H/8,W/8)

        b = self.aspp(e3)          # (B,8b,H/8,W/8)
        b = self.swa(b)
        b = self.lka(b)

        d3 = self.up3(b, e2)       # (B,4b,H/4,W/4)
        d2 = self.up2(d3, e1)      # (B,2b,H/2,W/2)
        d1 = self.up1(d2, xs)      # (B,b,H,W)

        aux4 = F.interpolate(self.aux4_head(d3), size=(H, W),
                             mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2_head(d2), size=(H, W),
                             mode="bilinear", align_corners=False)

        feat = d1
        sh = F.interpolate(self.strip_h(feat), size=(H, W),
                           mode="bilinear", align_corners=False)
        sw = F.interpolate(self.strip_w(feat), size=(H, W),
                           mode="bilinear", align_corners=False)
        feat_cat = torch.cat([feat, sh, sw], dim=1)
        feat_fused = self.strip_fuse(feat_cat)

        boundary = F.interpolate(self.boundary_head(feat_fused), size=(H, W),
                                 mode="bilinear", align_corners=False)
        logits = self.final_head(feat_fused)

        return {
            "logits": logits,
            "aux2": aux2,
            "aux4": aux4,
            "boundary": boundary,
        }
