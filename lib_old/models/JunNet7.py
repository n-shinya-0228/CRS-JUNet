# lib/models/ChatNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small utils ----------
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, bn=True, relu=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not bn)]  # groupsは使わない（=1）
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, drop=0, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
        super(BasicBlock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(drop)
        
        if inplanes != planes:
            self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm2d(inplanes)
        else:
            self.skip_conv = None
            self.skip_bn = None # Noneにしておくことで、forwardで分岐処理ができる

        
    def forward(self, x):
        residual1 = x

        out1 = self.bn1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out1 = self.conv2(out1)


# スキップ接続の処理
        if self.skip_conv is not None:
            residual1 = self.skip_bn(residual1) # スキップパスにもBNを適用することが多い
            residual1 = self.skip_conv(residual1)
            

        out1 += residual1
        out1 = self.relu(out1)                    #スキップ接続して、ReLU活性化 → 出力。

        return out1
    

class Downblock(nn.Module):
    def __init__(self, inplanes, planes, drop, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
        super(Downblock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=(1, 2), stride=(1, 2), bias=False)     #ダウンサンプリング
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        # down_x = x[:,:,:,::2]
        residual = self.bn3(x)
        residual = self.relu(residual)
        residual = self.conv3(residual)                                  #ダウンサンプリング
        
        out = self.bn4(residual)
        out = self.relu(out)
        out = self.conv4(out) 

        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.dropout(out)


        out += residual
        # out2 += down_x
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2, pool=True, drop=0.0):
        super().__init__()
        blocks = []
        blocks.append(BasicBlock(in_ch, out_ch, drop=drop))
        
        blocks.append(Downblock(out_ch, out_ch, drop=drop))
        self.block = nn.Sequential(*blocks)
    def forward(self, x):
        return self.block(x)

# ---------- ViT bottleneck ----------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop1(x)
        x = self.fc2(x); x = self.drop2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.1, drop_path_prob=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
    def forward(self, x):  # (B,N,C)
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.proj_drop(x)
        x = h + self.drop_path(x)
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + self.drop_path(x)
        return x

# def get_2d_sincos_pos_embed(h: int, w: int, dim: int, device):
#     """
#     2D sin-cos positional encoding (parameter-free)
#     return: (1, N, C) where N = H*W, C = dim
#     """
#     assert dim % 2 == 0, "dim must be even"
#     dim_h = dim // 2
#     dim_w = dim - dim_h

#     # 1D grids
#     gh = torch.arange(h, dtype=torch.float32, device=device).view(1, 1, h, 1)  # (1,1,H,1)
#     gw = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, 1, w)  # (1,1,1,W)

#     def pe_1d(pos, channels):
#         # pos: (1,1,H,1) or (1,1,1,W)
#         omega = torch.arange(channels, dtype=torch.float32, device=device) / channels
#         omega = 1.0 / (10000 ** omega)                     # (C,)
#         out = pos * omega.view(1, channels, 1, 1)          # (1,C,H,1) or (1,C,1,W)
#         return torch.cat([out.sin(), out.cos()], dim=1)    # (1,2C,H,1) or (1,2C,1,W)

#     # height and width encodings, then broadcast to (H,W)
#     pe_h = pe_1d(gh, dim_h)                                # (1,2*dim_h, H, 1)
#     pe_h = pe_h.expand(1, pe_h.shape[1], h, w)             # (1,2*dim_h, H, W)
#     pe_h = pe_h[:, :dim_h, :, :]                           # (1,dim_h,H,W)  (half sine, half cosine already mixed)

#     pe_w = pe_1d(gw, dim_w)                                # (1,2*dim_w, 1, W)
#     pe_w = pe_w.expand(1, pe_w.shape[1], h, w)             # (1,2*dim_w, H, W)
#     pe_w = pe_w[:, :dim_w, :, :]                           # (1,dim_w,H,W)

#     pe2d = torch.cat([pe_h, pe_w], dim=1)                  # (1,dim,H,W)
#     pe2d = pe2d.flatten(2).transpose(1, 2)                 # (1,N,dim)
#     return pe2d

def get_2d_sincos_pos_embed(h: int, w: int, dim: int, device):
    """
    2D sin-cos positional encoding
    return: (1, H*W, dim)
    """
    assert dim % 4 == 0, "dim should be divisible by 4 (half for H/W, then half for sin/cos)"
    dim_h = dim // 2
    dim_w = dim - dim_h
    ch_h = dim_h // 2   # sin/cos で半々
    ch_w = dim_w // 2

    gh = torch.arange(h, dtype=torch.float32, device=device).view(1, 1, h, 1)  # (1,1,H,1)
    gw = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, 1, w)  # (1,1,1,W)

    def pe_1d(pos, ch):
        # ch は「sin 用のチャンネル数」=「cos 用のチャンネル数」
        omega = torch.arange(ch, dtype=torch.float32, device=device) / max(ch, 1)
        omega = 1.0 / (10000 ** omega)                   # (ch,)
        out = pos * omega.view(1, ch, 1, 1)              # (1,ch,H,1) or (1,ch,1,W)
        return torch.cat([out.sin(), out.cos()], dim=1)  # (1,2*ch,...)

    pe_h = pe_1d(gh, ch_h).expand(1, 2 * ch_h, h, w)     # -> (1, dim_h, H, W)
    pe_w = pe_1d(gw, ch_w).expand(1, 2 * ch_w, h, w)     # -> (1, dim_w, H, W)

    pe2d = torch.cat([pe_h, pe_w], dim=1)                # (1, dim, H, W)
    pe2d = pe2d.flatten(2).transpose(1, 2)               # (1, H*W, dim)
    return pe2d



class ViTBottleneck(nn.Module):
    def __init__(self, patch_size: int, in_ch: int, embed_dim: int = 256, depth: int = 3, num_heads: int = 8,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.input_layer = nn.Linear(self.patch_size * self.patch_size * in_ch, embed_dim)
        # self.proj_in = nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, attn_drop, proj_drop, dpr[i], mlp_ratio)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_ch, kernel_size=1, bias=False)
    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        #順番を変える
        x = x.permute(0, 2, 4, 1, 3, 5)
        #分割したパッチごとにフラット化する
        #(バッチサイズ, 全パッチ数2*2, １パッチ当たりのパッチデータ16*16*3)
        x = x.reshape(B, (H // self.patch_size)*(W // self.patch_size), -1)     #パッチサイズ4の場合(B,16*16,4*4*256)

        x = self.input_layer(x)  #出力(バッチサイズ, 全パッチ数2*2, 512)
        D = x.shape[-1]
        # x = self.proj_in(x)                       # (B,D,H,W)
        # D = x.shape[1]
        # x = x.flatten(2).transpose(1, 2)          # (B,N,D)
        pos = get_2d_sincos_pos_embed(H // self.patch_size, W // self.patch_size, D, x.device)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).transpose(1, 2).reshape(B, D, H // self.patch_size, W // self.patch_size)
        x = self.proj_out(x)
        return x

# ---------- UNet up block ----------
class UpBlock(nn.Module):
    def __init__(self, x, y, in_ch: int, skip_ch=0, out_ch=0, drop: float = 0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(x,y), mode='bilinear', align_corners=False)
        self.conv1 = conv_bn_relu(in_ch + skip_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = conv_bn_relu(out_ch, out_ch, k=3, s=1, p=1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2] != skip.shape[-2] or x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x); x = self.drop(x); x = self.conv2(x)
        return x

# ---------- Main Model ----------
class JunNet7(nn.Module):
    """
    UNet + ViT bottleneck with deep supervision & boundary head.
    入力: (B, in_ch, H, W)
    出力: dict {'logits','aux2','aux4','boundary'} （全て入力と同じ解像度）
    """
    def __init__(self, in_channels: int = 6, nclasses: int = 20, drop: float = 0.2,
                 base_ch: int = 32, vit_dim: int = 256, vit_depth: int = 3, vit_heads: int = 8,
                 drop_path: float = 0.1, return_aux: bool = True):
        super().__init__()
        self.return_aux = return_aux
        # Encoder
        self.stem = conv_bn_relu(in_channels, base_ch, k=3, s=1, p=1)
        self.enc1 = ResBlock(base_ch, base_ch * 2, num_blocks=2, pool=True,  drop=drop)  # 1/2
        self.enc2 = ResBlock(base_ch*2, base_ch * 4, num_blocks=2, pool=True,  drop=drop)  # 1/4
        self.enc3 = ResBlock(base_ch*4, base_ch * 8, num_blocks=2, pool=True,  drop=drop)  # 1/8
        # Bottleneck
        self.vit_in  = conv_bn_relu(base_ch*8, base_ch*8, k=1, s=1, p=0)
        self.vit     = ViTBottleneck(patch_size=4, in_ch=base_ch*8, embed_dim=vit_dim, depth=vit_depth, num_heads=vit_heads,
                                     mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.1, drop_path=drop_path)
        # self.vit_out = conv_bn_relu(base_ch*8, base_ch*8, k=3, s=1, p=1)
        self.up4 = UpBlock(x=4,y=4,in_ch=base_ch*8, out_ch=base_ch*8, drop=drop) 
        # Decoder
        
        self.up3 = UpBlock(x=1,y=2,in_ch=base_ch*8, skip_ch=base_ch*4, out_ch=base_ch*4, drop=drop)  # ->1/4
        self.up2 = UpBlock(x=1,y=2,in_ch=base_ch*4, skip_ch=base_ch*2, out_ch=base_ch*2, drop=drop)  # ->1/2
        self.up1 = UpBlock(x=1,y=2,in_ch=base_ch*2, skip_ch=base_ch,   out_ch=base_ch,   drop=drop)  # ->1/1
        # Heads
        self.fuse         = conv_bn_relu(base_ch, base_ch, k=3, s=1, p=1)
        self.final_logits = nn.Conv2d(base_ch, nclasses, kernel_size=1)
        self.aux2 = nn.Conv2d(base_ch*2, nclasses, kernel_size=1)  # 1/2
        self.aux4 = nn.Conv2d(base_ch*4, nclasses, kernel_size=1)  # 1/4
        self.boundary_head = nn.Sequential(
            conv_bn_relu(base_ch, base_ch, k=3, s=1, p=1),
            nn.Conv2d(base_ch, 1, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # 追加
        mask = x[:, -1:, :, :]         # [B,1,H,W]
        x_img = x[:, :-1, :, :] * mask # 有効画素のみ通す
        x = torch.cat([x_img, mask], dim=1)  # 形は [B,6,H,W] のまま


        x0 = self.stem(x)    # (32, 64, 512)
        x1 = self.enc1(x0)   # (64, 64, 256)
        x2 = self.enc2(x1)   # (128, 64, 128)
        x3 = self.enc3(x2)   # (256, 64, 64)

        b  = self.vit_in(x3)    # (256, 64, 64)
        b  = self.vit(b)        # (256, 16, 16)
        b  = self.up4(b)        # (256, 64, 64)
        # b  = self.vit_out(b)    # (256, 64, 64)

        d3 = self.up3(b,  x2)    # 1/4  (128, 64, 128)
        d2 = self.up2(d3, x1)    # 1/2  (64, 64, 256)
        d1 = self.up1(d2, x0)    # 1/1  (32, 64, 512)

        feat   = self.fuse(d1)   #(32, 64, 512)
        logits = self.final_logits(feat)   #(19, 64, 512)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        # 追加：mask==0 の場所はクラス0以外のlogitを強制的に低くする
        invalid = (mask < 0.5)
        if invalid.any():
            # クラス0以外を強く抑制
            logits[:, 1:, :, :][invalid.expand_as(logits[:, 1:, :, :])] = -1e4

        if not self.return_aux:
            return logits

        aux2 = F.interpolate(self.aux2(d2), size=(H, W), mode='bilinear', align_corners=False)
        aux4 = F.interpolate(self.aux4(d3), size=(H, W), mode='bilinear', align_corners=False)
        boundary = F.interpolate(self.boundary_head(feat), size=(H, W), mode='bilinear', align_corners=False)
        return {"logits": logits, "aux2": aux2, "aux4": aux4, "boundary": boundary}
