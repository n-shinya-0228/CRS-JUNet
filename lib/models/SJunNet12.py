# -*- coding: utf-8 -*-
# JunNet13 (Light) : Shifted Window Attention (Swin-like) + Cached Mask + Reduced Cost
# in:  (B, 6, H, W)  = [range, x, y, z, remission, obs_mask
# Cartesian

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True, d=1):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=groups, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 4), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 4), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0, use_se=True):
        super().__init__()
        self.bn1 = nn.InstanceNorm2d(in_ch, affine=True)
        self.act = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_ch, affine=True)

        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.se = SE(out_ch) if use_se else nn.Identity()

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )

    def forward(self, x, mask=None):
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

        # ★ 追加: マスクが与えられた場合、点がない場所の特徴量をゼロにリセット
        if mask is not None:
            # テンソルのサイズが違う場合（プーリング後など）は、マスクも縮小する
            if x.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
            x = x * mask

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

    def forward(self, x, mask=None):
        for block in self.net:
            x = block(x, mask) # ★ 各ブロックにマスクを渡す
        return x


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


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=1, s=1, p=0))
            else:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=3, s=1, p=r, d=r))

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # ★ conv_bn_act を使わず、直接 Conv2d と ReLU を書く（InstanceNormを回避）
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.proj = conv_bn_act(out_ch * (len(rates) + 1), out_ch, k=1, s=1, p=0)

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
        inter_ch = max(inter_ch, 8)
        self.theta = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, skip, gate):
        g = F.interpolate(self.phi(gate), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        s = self.theta(skip)
        a = torch.sigmoid(self.psi(F.relu(s + g, inplace=True)))
        return skip * a


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.gate = AttnGate(skip_ch, in_ch, inter_ch=min(skip_ch, in_ch) // 2)
        self.conv1 = conv_bn_act(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = conv_bn_act(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


# ---- Shifted Window Attention (Swin-like, minimal) ----
class RelPosBias(nn.Module):
    def __init__(self, window: Tuple[int, int], heads: int):
        super().__init__()
        Wh, Ww = window
        self.window = window
        self.heads = heads
        num_rel = (2 * Wh - 1) * (2 * Ww - 1)
        self.table = nn.Parameter(torch.zeros(num_rel, heads))

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2,Wh,Ww
        coords_flat = torch.flatten(coords, 1)  # 2,N
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2,N,N
        rel = rel.permute(1, 2, 0).contiguous()  # N,N,2
        rel[:, :, 0] += Wh - 1
        rel[:, :, 1] += Ww - 1
        rel[:, :, 0] *= (2 * Ww - 1)
        rel_index = rel.sum(-1)  # N,N
        self.register_buffer("rel_index", rel_index, persistent=False)

        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self):
        N = self.rel_index.shape[0]
        bias = (
            self.table[self.rel_index.view(-1)]
            .view(N, N, self.heads)
            .permute(2, 0, 1)
            .contiguous()
        )
        return bias  # (heads,N,N)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, window=(8, 8)):
        super().__init__()
        assert dim % heads == 0, f"dim({dim}) must be divisible by heads({heads})"
        self.dim = dim
        self.heads = heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rpb = RelPosBias(window, heads)

    def forward(self, x, attn_mask=None):  # x: (B,N,C)
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,heads,N,dimh)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (q.shape[-1] ** 0.5))
        attn = attn + self.rpb().unsqueeze(0)  # (1,heads,N,N)

        if attn_mask is not None:
            # attn_mask: (Bwin,1,N,N) where Bwin = nW*B
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


def window_partition(x, window: Tuple[int, int]):
    # x: (B,H,W,C) -> (num_windows*B, Wh*Ww, C)
    B, H, W, C = x.shape
    Wh, Ww = window
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, Wh * Ww, C)
    return x


def window_reverse(x, window: Tuple[int, int], H, W, B):
    # x: (num_windows*B, Wh*Ww, C) -> (B,H,W,C)
    Wh, Ww = window
    C = x.shape[-1]
    x = x.view(B, H // Wh, W // Ww, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x


class ShiftedSWABlock(nn.Module):
    def __init__(self, ch, window: Tuple[int, int] = (8, 8), heads=4, shift: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(ch)
        self.attn = WindowAttention(ch, heads=heads, window=window)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))

        # mask cache: key=(Hp,Wp,device_type,dtype) -> (nW,1,N,N)
        self._attn_mask_cache = {}

    @torch.no_grad()
    def _build_attn_mask(self, Hp: int, Wp: int, device, dtype):
        """Return (nW,1,N,N) mask for a single image (batch展開はしない)."""
        Wh, Ww = self.window
        Sh, Sw = self.shift

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        cnt = 0
        h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None))
        w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None))
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window).squeeze(-1)  # (nW, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        attn_mask = attn_mask.unsqueeze(1).to(dtype=dtype)  # (nW,1,N,N)
        return attn_mask

    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        Wh, Ww = self.window
        Sh, Sw = self.shift

        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.permute(0, 2, 3, 1).contiguous()  # (B,Hp,Wp,C)

        # shift
        if Sh != 0 or Sw != 0:
            x = torch.roll(x, shifts=(-Sh, -Sw), dims=(1, 2))

        attn_mask = None
        if Sh != 0 or Sw != 0:
            key = (Hp, Wp, x.device.type, x.dtype)
            cached = self._attn_mask_cache.get(key, None)
            if cached is None or cached.device != x.device or cached.dtype != x.dtype:
                cached = self._build_attn_mask(Hp, Wp, device=x.device, dtype=x.dtype)
                self._attn_mask_cache[key] = cached

            # (nW,1,N,N) -> (nW*B,1,N,N)
            attn_mask = cached.repeat_interleave(B, dim=0)

        # window partition
        xw = window_partition(x, self.window)  # (nW*B, N, C)
        if attn_mask is not None:
            assert attn_mask.shape[0] == xw.shape[0], (attn_mask.shape, xw.shape)

        h = self.norm1(xw)
        h = self.attn(h, attn_mask=attn_mask)
        xw = xw + h

        h = self.norm2(xw)
        h = self.mlp(h)
        xw = xw + h

        # reverse
        x = window_reverse(xw, self.window, Hp, Wp, B)

        # reverse shift
        if Sh != 0 or Sw != 0:
            x = torch.roll(x, shifts=(Sh, Sw), dims=(1, 2))

        x = x[:, :H, :W, :].contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class CartesianCSWinBlock(nn.Module):
    """
    Cross-Shaped Window Attention (CSWin) for Cartesian BEV
    直交座標系用の十字窓アテンション（円環シフトなし）。
    """
    def __init__(self, ch, window_r=(16, 4), window_a=(4, 16), heads=4):
        super().__init__()
        assert ch % 2 == 0, "Channel must be divisible by 2 for CSWin"
        self.ch_half = ch // 2
        self.heads_half = max(1, heads // 2)

        self.window_r = window_r  # 縦長窓
        self.window_a = window_a  # 横長窓

        self.norm1 = nn.LayerNorm(ch)

        # 半分ずつ処理する独立したWindow Attention (シフトなし)
        self.attn_r = WindowAttention(self.ch_half, heads=self.heads_half, window=window_r)
        self.attn_a = WindowAttention(self.ch_half, heads=self.heads_half, window=window_a)

        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))

    def _process_branch(self, x, window, attn_module):
        B, Hp, Wp, C = x.shape
        xw = window_partition(x, window)
        h = attn_module(xw, attn_mask=None) # Cartesianでシフトしない場合はマスク不要
        xw = xw + h
        x = window_reverse(xw, window, Hp, Wp, B)
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        max_Wh = max(self.window_r[0], self.window_a[0])
        max_Ww = max(self.window_r[1], self.window_a[1])
        pad_h = (max_Wh - H % max_Wh) % max_Wh
        pad_w = (max_Ww - W % max_Ww) % max_Ww
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.permute(0, 2, 3, 1).contiguous()
        idt = x

        x = self.norm1(x)

        x_r = x[..., :self.ch_half]
        x_a = x[..., self.ch_half:]

        out_r = self._process_branch(x_r, self.window_r, self.attn_r)
        out_a = self._process_branch(x_a, self.window_a, self.attn_a)

        x = torch.cat([out_r, out_a], dim=-1)
        
        x = idt + x
        x = x + self.mlp(self.norm2(x))

        x = x[:, :H, :W, :].contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class SJunNet12(nn.Module):
    """
    Cartesian BEV 専用最強モデル。
    通常のCNN ＋ 十字窓アテンション(CSWin) 搭載。
    """
    def __init__(
        self,
        in_channels: int = 7,     # ★ 5 から 7 に変更
        nclasses: int = 20,
        drop: float = 0.3,
        base_ch: int = 48,
        aspp_out: int = 256,
        swa_heads: int = 4,
        swa_window: Tuple[int, int] = (16, 16),
        return_aux: bool = True,
    ):
        super().__init__()
        self.return_aux = return_aux

        assert aspp_out % swa_heads == 0

        # 直交座標系の通常の CNN
        self.stem_feat = conv_bn_act(7, base_ch, 3, 1, 1)  # ★ ここを 7 にする
        self.stem_mask = nn.Sequential(
            conv_bn_act(1, base_ch // 2, 1, 1, 0),
            conv_bn_act(base_ch // 2, base_ch, 3, 1, 1),
            nn.Conv2d(base_ch, base_ch, 1),
        )
        self.gate_scale = nn.Parameter(torch.tensor(1.0))  # ★ 真のゲート機構用のスケールを追加

        self.enc1 = ResStage(base_ch, base_ch * 2, num_blocks=2, pool=True, drop=drop)
        self.enc2 = ResStage(base_ch * 2, base_ch * 4, num_blocks=2, pool=True, drop=drop)
        self.enc3 = ResStage(base_ch * 4, base_ch * 8, num_blocks=2, pool=True, drop=drop)

        self.aspp = ASPP(base_ch * 8, aspp_out)

        self.swa = nn.Sequential(
            CartesianCSWinBlock(aspp_out, window_r=(16, 4), window_a=(4, 16), heads=swa_heads),
            CartesianCSWinBlock(aspp_out, window_r=(16, 4), window_a=(4, 16), heads=swa_heads),
        )

        self.bottleneck_proj = conv_bn_act(aspp_out, base_ch * 8, 1, 1, 0)
        self.lka = LKA(base_ch * 8, k=7, d=3)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, drop=drop)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, drop=drop)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, drop=drop)

        self.aux4_head = nn.Conv2d(base_ch * 4, nclasses, 1)
        self.aux2_head = nn.Conv2d(base_ch * 2, nclasses, 1)

        self.fuse = conv_bn_act(base_ch, base_ch, 3, 1, 1)
        
        self.strip_h = nn.Sequential(nn.AdaptiveAvgPool2d((None, 1)), conv_bn_act(base_ch, base_ch, 1, 1, 0))
        self.strip_w = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), conv_bn_act(base_ch, base_ch, 1, 1, 0))
        self.strip_fuse = conv_bn_act(base_ch * 3, base_ch, 1, 1, 0)

        self.boundary_head = nn.Conv2d(base_ch, 1, 1)
        self.final_logits = nn.Conv2d(base_ch, nclasses, 1)

    def forward(self, x):
        feat_in = x[:, :7, :, :]  
        m = x[:, 7:8, :, :]       

        # mask-aware gate
        s0 = self.stem_feat(feat_in)              # (B, C, H, W)
        g  = torch.tanh(self.stem_mask(m))        # [-1,1]に制限
        s0 = s0 * (1.0 + g)                       # ゲート（観測=正、欠測=抑制）  ← 改良点

        # encoder-decoder as before
        B, C, H, W = x.shape
        s1 = self.enc1(s0)  
        s2 = self.enc2(s1)  
        s3 = self.enc3(s2) 
        b = self.aspp(s3) 
        b = self.bottleneck_proj(b)
        b = self.lka(b)
        d3 = self.up3(b, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)

        aux4 = F.interpolate(self.aux4_head(d3), size=(H, W), mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2_head(d2), size=(H, W), mode="bilinear", align_corners=False)

        feat = self.fuse(d1)
        sh = F.interpolate(self.strip_h(feat), size=(H, W), mode="bilinear", align_corners=False)
        sw = F.interpolate(self.strip_w(feat), size=(H, W), mode="bilinear", align_corners=False)
        feat = self.strip_fuse(torch.cat([feat, sh, sw], dim=1))

        boundary = F.interpolate(self.boundary_head(feat), size=(H, W), mode="bilinear", align_corners=False)
        logits = self.final_logits(feat)
        return {"logits": logits, "aux2": aux2, "aux4": aux4, "boundary": boundary}