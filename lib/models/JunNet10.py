# -*- coding: utf-8 -*-
# JunNet10 (Compat with chat3.py)
# in:  (B, 6, H, W) = [range, x, y, z, remission, obs_mask]
# out: dict {'logits','logits_raw','aux2','aux4','boundary'} with HxW resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# -------------------- small utils --------------------
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True, d=1):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---- ICM: Inception-like Context Module ----
class ICM(nn.Module):
    """ Multi-scale context extractor (lightweight).
        branches: 3x3 (d=1), 5x5, 3x3 (d=2) + image-level pooling
    """
    def __init__(self, in_ch=5, c1=24, c2=32, c3=64):
        super().__init__()
        self.b1 = conv_bn_act(in_ch, c1, 3, 1, 1, d=1)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 5, 1, 2, bias=False, groups=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.b3 = conv_bn_act(in_ch, c1, 3, 1, 2, d=2)
        self.mix1 = conv_bn_act(c1*3, c2, 1, 1, 0)
        self.ms1a = conv_bn_act(c2, c2, 3, 1, 1, d=1)
        self.ms1b = conv_bn_act(c2, c2, 3, 1, 2, d=2)
        self.img = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_act(in_ch, c2, 1, 1, 0))
        self.mix2 = conv_bn_act(c2*3, c3, 1, 1, 0)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        h = torch.cat([b1, b2, b3], dim=1)
        h = self.mix1(h)
        h = torch.cat([
            self.ms1a(h),
            self.ms1b(h),
            F.interpolate(self.img(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        ], dim=1)
        h = self.mix2(h)
        return h


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
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                                      nn.BatchNorm2d(out_ch))
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


# ---- CAM: Context Aggregation Module ----
class CAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        )
        self.gate = nn.Sigmoid()
    def forward(self, x):
        a = self.gate(self.conv(x))
        return x * (1.0 + a)  # residual gate


# ---- CBAM (Channel + Spatial attention) ----
class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel attention
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) +
                           self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        # Spatial attention
        sa_in = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        sa = torch.sigmoid(self.spatial(sa_in))
        return x * sa


class LKA(nn.Module):
    def __init__(self, ch, k=7, d=3):
        super().__init__()
        self.dw1 = nn.Conv2d(ch, ch, kernel_size=k, padding=k//2, groups=ch)
        self.dw2 = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d, groups=ch)
        self.pw  = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        u = x
        x = self.dw1(x); x = self.dw2(x); x = self.pw(x)
        return x * u


# ---- Swin-like Window Attention (no shift) ----
class WindowAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):  # x: (B,H,W,C)
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
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
        self.mlp = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))
    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        Ph, Pw = self.window
        pad_h = (Ph - H % Ph) % Ph
        pad_w = (Pw - W % Pw) % Pw
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, Hp//Ph, Ph, Wp//Pw, Pw, C).permute(0,1,3,2,4,5).contiguous()
        x = x.view(-1, Ph, Pw, C)
        h = self.norm1(x); h = self.attn(h); x = x + h
        h = self.norm2(x); h = self.mlp(h); x = x + h
        x = x.view(B, Hp//Ph, Wp//Pw, Ph, Pw, C).permute(0,1,3,2,4,5).contiguous()
        x = x.view(B, Hp, Wp, C)[:, :H, :W, :].permute(0,3,1,2).contiguous()
        return x


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=1, s=1, p=0))
            else:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=3, s=1, p=r, d=r))
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_act(in_ch, out_ch, k=1, s=1, p=0))
        self.proj = conv_bn_act(out_ch * (len(rates) + 1), out_ch, k=1, s=1, p=0)
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=x.shape[-2:], mode='bilinear', align_corners=False)
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
        g = F.interpolate(self.phi(gate), size=skip.shape[-2:], mode='bilinear', align_corners=False)
        s = self.theta(skip)
        a = torch.sigmoid(self.psi(F.relu(s + g, inplace=True)))
        return skip * a


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.gate = AttnGate(skip_ch, in_ch, inter_ch=min(skip_ch, in_ch) // 2)
        self.conv1 = conv_bn_act(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = conv_bn_act(out_ch, out_ch, 3, 1, 1)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip = self.gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


# ---- MCSPN: Multi-class Spatial Propagation Network ----
class MCSPN(nn.Module):
    def __init__(self, ch_in, nclasses, T=4):
        super().__init__()
        self.T = T
        self.guidance = nn.Sequential(
            conv_bn_act(ch_in, max(64, ch_in // 2), 3, 1, 1),
            nn.Conv2d(max(64, ch_in // 2), nclasses * 4, 1)
        )

    def forward(self, feats, logits):
        # feats: (B,Cf,Hf,Wf), logits: (B,K,H,W)
        B, K, H, W = logits.shape

        if feats.shape[-2:] != (H, W):
            feats = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)

        g = self.guidance(feats).view(B, K, 4, H, W)  # [left, right, up, down]
        g = torch.softmax(g, dim=2)

        h = logits

        def shift_left(x):
            return F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        def shift_right(x):
            return F.pad(x, (1, 0, 0, 0))[:, :, :, :W]
        def shift_up(x):
            return F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        def shift_down(x):
            return F.pad(x, (0, 0, 1, 0))[:, :, :H, :]

        for _ in range(self.T):
            left  = shift_right(h)
            right = shift_left(h)
            up    = shift_down(h)
            down  = shift_up(h)
            agg = (g[:, :, 0] * left +
                   g[:, :, 1] * right +
                   g[:, :, 2] * up +
                   g[:, :, 3] * down)
            self_w = 1.0 - g.sum(dim=2)
            h = self_w * h + agg

        return h


# ---- FPN-HR Fuse ----
class FPNFuse(nn.Module):
    def __init__(self, in_chs, out_ch):
        super().__init__()
        self.proj = nn.ModuleList([conv_bn_act(c, out_ch, 1, 1, 0) for c in in_chs])
        self.mix = conv_bn_act(out_ch * len(in_chs), out_ch, 3, 1, 1)

    def forward(self, feats, size_hw):
        ups = []
        for i, f in enumerate(feats):
            x = self.proj[i](f)
            x = F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)
            ups.append(x)
        return self.mix(torch.cat(ups, dim=1))


# -------------------- main net --------------------
class JunNet10(nn.Module):
    """
    in_channels: 6  (range, x, y, z, remission, obs_mask)
    nclasses:    20 (SemanticKITTI xentropy IDs)
    * JunNet9 と I/O 互換、内部だけ強化（Range-aware, CBAM, FPN融合）
    """
    def __init__(self, in_channels: int = 6, nclasses: int = 20, drop: float = 0.2,
                 base_ch: int = 48, aspp_out: int = 384, swa_blocks: int = 2,
                 swa_heads: int = 8, swa_window: Tuple[int, int] = (8, 16),
                 return_aux: bool = True, spn_steps: int = 4, fpn_ch: int = 96):
        super().__init__()
        self.return_aux = return_aux
        self.nclasses = nclasses

        # ----- Range-aware + Mask-aware Stem -----
        self.icm = ICM(in_ch=5, c1=24, c2=32, c3=base_ch)
        self.stem_mask = nn.Sequential(
            conv_bn_act(1, base_ch//2, 1, 1, 0),
            conv_bn_act(base_ch//2, base_ch, 3, 1, 1),
            nn.Conv2d(base_ch, base_ch, 1)
        )
        self.range_gate = nn.Sequential(
            nn.Conv2d(1, base_ch//2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, base_ch, 3, 1, 1)
        )

        # encoder + CAM
        self.enc1 = ResStage(base_ch, base_ch * 2, num_blocks=2, pool=True, drop=drop)
        self.cam1 = CAM(base_ch * 2)
        self.enc2 = ResStage(base_ch * 2, base_ch * 4, num_blocks=2, pool=True, drop=drop)
        self.cam2 = CAM(base_ch * 4)
        self.enc3 = ResStage(base_ch * 4, base_ch * 8, num_blocks=2, pool=True, drop=drop)
        self.cam3 = CAM(base_ch * 8)

        # bottleneck: ASPP -> SWA -> CBAM -> LKA
        self.aspp = ASPP(base_ch * 8, aspp_out)
        self.swa = nn.Sequential(*[SWABlock(aspp_out, window=swa_window, heads=swa_heads) for _ in range(swa_blocks)])
        self.bot_proj = conv_bn_act(aspp_out, base_ch * 8, 1, 1, 0)
        self.cbam = CBAM(base_ch * 8, r=16)
        self.lka = LKA(base_ch * 8, k=7, d=3)

        # decoder
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, drop=drop)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, drop=drop)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, drop=drop)

        # High-res FPN fusion
        self.fpn = FPNFuse(in_chs=[base_ch, base_ch*2, base_ch*4, base_ch], out_ch=fpn_ch)

        # heads
        self.aux4_head = nn.Conv2d(base_ch * 4, nclasses, 1)
        self.aux2_head = nn.Conv2d(base_ch * 2, nclasses, 1)
        self.fuse = conv_bn_act(base_ch, base_ch, 3, 1, 1)
        self.strip_h = nn.Sequential(nn.AdaptiveAvgPool2d((None, 1)), conv_bn_act(base_ch, base_ch, 1, 1, 0))
        self.strip_w = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), conv_bn_act(base_ch, base_ch, 1, 1, 0))
        self.strip_fuse = conv_bn_act(base_ch * 3, base_ch, 1, 1, 0)
        self.boundary_head = nn.Conv2d(base_ch, 1, 1)

        # final
        self.final_mix = conv_bn_act(base_ch + fpn_ch, base_ch, 3, 1, 1)
        self.final_logits = nn.Conv2d(base_ch, nclasses, 1)

        # MCSPN (use feat_cat that includes FPN)
        self.mcspn = MCSPN(ch_in=base_ch + fpn_ch, nclasses=nclasses, T=spn_steps)

    def forward(self, x):
        # x: (B,6,H,W)
        feat_in = x[:, :5, :, :]
        range_only = x[:, 0:1, :, :]
        m = x[:, 5:6, :, :]

        s0 = self.icm(feat_in)
        g_mask  = torch.tanh(self.stem_mask(m))
        g_range = torch.tanh(self.range_gate(range_only))
        s0 = s0 * (1.0 + g_mask) * (1.0 + g_range)

        B, C, H, W = x.shape
        s1 = self.cam1(self.enc1(s0))     # 1/2
        s2 = self.cam2(self.enc2(s1))     # 1/4
        s3 = self.cam3(self.enc3(s2))     # 1/8

        b = self.aspp(s3)
        b = self.swa(b)
        b = self.bot_proj(b)
        b = self.cbam(b)
        b = self.lka(b)

        d3 = self.up3(b, s2)              # 1/4
        d2 = self.up2(d3, s1)             # 1/2
        d1 = self.up1(d2, s0)             # 1/1

        aux4 = F.interpolate(self.aux4_head(d3), size=(H, W), mode='bilinear', align_corners=False)
        aux2 = F.interpolate(self.aux2_head(d2), size=(H, W), mode='bilinear', align_corners=False)

        feat = self.fuse(d1)
        sh = F.interpolate(self.strip_h(feat), size=(H, W), mode='bilinear', align_corners=False)
        sw = F.interpolate(self.strip_w(feat), size=(H, W), mode='bilinear', align_corners=False)
        feat = self.strip_fuse(torch.cat([feat, sh, sw], dim=1))

        boundary = F.interpolate(self.boundary_head(feat), size=(H, W), mode='bilinear', align_corners=False)

        # FPN 高解像融合
        fpn_feat = self.fpn([s0, s1, s2, d1], size_hw=(H, W))

        # 最終予測
        feat_cat = torch.cat([feat, fpn_feat], dim=1)   # (B, base_ch + fpn_ch, H, W)
        feat_mix = self.final_mix(feat_cat)
        logits_raw = self.final_logits(feat_mix)
        logits = self.mcspn(feat_cat, logits_raw)

        return {
            "logits": logits,
            "logits_raw": logits_raw,
            "aux2": aux2,
            "aux4": aux4,
            "boundary": boundary
        }
