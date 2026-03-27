import torch
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm


BN_MOMENTUM = 0.1                                               #バッチ正規化のモーメンタム（平均の移動平均係数）を定義


class first_layer(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        if inplanes != planes:
            self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        else:
            self.skip_conv = None
            self.skip_bn = None # Noneにしておくことで、forwardで分岐処理ができる
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        # メインパス
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # 最初のReLUはConv1の後

        

        out = self.conv2(out)
        out = self.bn2(out) # 2番目のConvの後にBN
        out = self.dropout(out)

        # スキップ接続の処理
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity) # スキップパスにもBNを適用することが多い

        # 残差接続: メインパスの出力にスキップパスの出力を加算
        out += identity
        out = self.relu(out) # 残差接続の後にReLU

        # out = self.pool(x)

        return out
        


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dropout, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
        super(BasicBlock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), dilation=(2, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(dropout)
        # self.downsample = downsample                                    #スキップ接続用のダウンサンプルが必要かを保存。

        
    def forward(self, x):
        residual1 = self.conv0(x)                                        #入力を残しておく（スキップ接続用）。
        residual1 = self.bn0(residual1)
        # residual1 = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)


        # if self.downsample is not None:
        #     residual = self.downsample(x)         #入出力のチャネル数などが合わない場合、residual も変換。

        out1 += residual1
        out1 = self.relu(out1)                    #スキップ接続して、ReLU活性化 → 出力。

        return out1
    

class Downblock(nn.Module):
    def __init__(self, inplanes, planes, dropout, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
        super(Downblock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=(1, 2), stride=(1, 2), bias=False)     #ダウンサンプリング
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.bn5 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # down_x = x[:,:,:,::2]

        residual2 = self.conv3(x)                                  #ダウンサンプリング
        residual2 = self.bn3(residual2)

        out2 = self.conv4(residual2)
        out2 = self.bn4(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)

        out2 = self.conv5(out2)
        out2 = self.bn5(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)


        out2 += residual2
        # out2 += down_x
        out2 = self.relu(out2)

        return out2
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        # 1x1 畳み込み
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 異なるレートのアトラス畳み込み
        self.aspp_convs = nn.ModuleList()
        for rate in rates:
            self.aspp_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # グローバル平均プーリング
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # グローバル平均プーリング
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 全てのASPP出力を結合した後の最終1x1畳み込み
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, kernel_size=1), # 1x1, aspp_convs, global_avg_poolの合計
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _s = x.shape[2:] # 後のアップサンプリングのためにサイズを保存

        x1 = self.conv1x1(x)
        aspp_outs = [x1]

        for aspp_conv in self.aspp_convs:
            aspp_outs.append(aspp_conv(x))

        x_pool = self.global_avg_pool(x)
        x_pool = F.interpolate(x_pool, size=_s, mode='bilinear', align_corners=True) # 元のサイズに戻す
        aspp_outs.append(x_pool)

        x_concat = torch.cat(aspp_outs, dim=1) # 全ての分岐を結合
        return self.final_conv(x_concat)

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, num_inputlayer_units: int, num_heads: int, dropout: float=0.3):   #num_inputlayer_units:全結合層のユニット数, num_heads:マルチヘッドアテンションのヘッド数
        super().__init__()


        if num_inputlayer_units % num_heads != 0:
            raise ValueError("num_inputlayer_units must be divisible by num_heads")
        
        self.num_heads = num_heads
        #ヘッドごとの特徴量の次元の次元を求める(128)
        dim_head = num_inputlayer_units // num_heads
        #q,k,vを作るための全結合層
        self.expansion_layer = nn.Linear(num_inputlayer_units, num_inputlayer_units * 3, bias=False)
        #softmax関数のオーバーフロー対策
        self.scale = 1 / (dim_head ** 0.5)

        self.headjoin_layer =  nn.Linear(num_inputlayer_units, num_inputlayer_units)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):

        #入力token(32, 5, 512)からバッチサイズとパッチ数を取得
        bs, ns = x.shape[:2]
        #入力(32, 5, 512)から出力(32, 5, 512*3), query,key,valueを作る。
        qkv = self.expansion_layer(x)
        #(32, 5, 3(query,key,value), 4(ヘッド数), 128(特徴量))に変換し、(3, 32, 4, 5, 128)
        qkv = qkv.view(bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        #query,key,valueに分割する
        q, k, v = qkv.unbind(0)

        #qと転置したkを内積。(5, 128)・(128, 5)=(5, 5)
        attn = q.matmul(k.transpose(-2, -1))
        #アテンションスコアの行方向ソフトマックス関数を適用
        attn = (attn * self.scale).softmax(dim=-1)
        attn = self.dropout(attn)
        #アテンションスコアとvを内積(5, 128)
        x = attn.matmul(v)
        #(32, 4, 5, 128)から(32, 5, 4, 128).flattenで(32, 5, 512)
        x = x.permute(0, 2, 1, 3).flatten(2)
        #(32, 5, 512)から(32, 5, 512)
        x = self.headjoin_layer(x)

        return x

#multiheadattention終わった後に、MLPに通す
class MLP(nn.Module):
    def __init__(self, num_inputlayer_units: int, num_mlp_units: int, dropout: float=0.1):
        super().__init__()

        self.linear1 = nn.Linear(num_inputlayer_units, num_mlp_units)   #今回num_inputlayer_unitsとnum_mlp_unitsは同じ数
        self.linear2 = nn.Linear(num_mlp_units, num_inputlayer_units)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
#正規化層、アテンション層、正規化層、MLP
class EncorderBlock(nn.Module):
    def __init__(self, num_inputlayer_units: int, num_heads: int, num_mlp_units: int):
        super().__init__()

        self.attention = MultiHeadSelfAttention(num_inputlayer_units, num_heads)
        self.mlp = MLP(num_inputlayer_units, num_mlp_units)
        self.norm1 = nn.LayerNorm(num_inputlayer_units)
        self.norm2 = nn.LayerNorm(num_inputlayer_units)

    
    def forward(self, x: torch.Tensor):

        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x

        return x


class VisionTransformer(nn.Module):

    def __init__(self, img_size_h: int, img_size_w: int, in_channel: int, patch_size: int, num_inputlayer_units: int, num_heads: int, num_mlp_units: int, num_layers: int):
        super().__init__()

        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size

        num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)     #全パッチ数
        input_dim = in_channel * patch_size ** 2                 #１パッチあたりの特徴量 

        #特徴次元削減のため768から512にする全結合層を定義
        self.input_layer = nn.Linear(input_dim, num_inputlayer_units)
        #nn.Parameterでtensorを作ると、勾配の計算が可能になる
        #クラストークンの生成、3階テンソル(1, 1, 512)を平均0, 標準偏差1でランダムに値を生成
        self.class_token = nn.Parameter(torch.zeros(1, 1, num_inputlayer_units))  
        #位置情報の生成、3階テンソル(1, パッチ数+1(クラストークン), 512)、これはtorch.catではなく加算用
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_inputlayer_units))
        #num_layersの数だけエンコーダーブロックを作成(今回は３)
        self.encoder_layer = nn.ModuleList([EncorderBlock(num_inputlayer_units, num_heads, num_mlp_units) for _ in range(num_layers)])

        #出力時
        self.normalize = nn.LayerNorm(num_inputlayer_units)
        # self.output_layer = nn.Linear(num_inputlayer_units, num_classes)

    def forward(self, x: torch.Tensor):
        #ミニバッチサイズ32, チャネル3, 縦32, 横32
        bs, c, h, w = x.shape
        #(32, 3, 32, 32)を(32, 3, 2, 16, 2, 16)の6階テンソルに変換させる。これはパッチサイズ16*16に分割するコード、計４つのパッチができる
        x = x.view(bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)

        #順番を変える
        x = x.permute(0, 2, 4, 1, 3, 5)
        #分割したパッチごとにフラット化する
        #(バッチサイズ, 全パッチ数2*2, １パッチ当たりのパッチデータ16*16*3)
        x = x.reshape(bs, (h // self.patch_size)*(w // self.patch_size), -1)

        x = self.input_layer(x)  #出力(バッチサイズ, 全パッチ数2*2, 512)

        #バッチサイズ分クラストークンを拡張
        class_token = self.class_token.expand(bs, -1, -1)
        #dimで連結する次元を指定する。このときほかの次元は数があってないといけない。(32, 5, 512)になる。
        x = torch.cat((class_token, x), dim=1)
        #位置情報の付与
        x += self.pos_embed
        
        #encoderブロックに三回通す
        for layer in self.encoder_layer:
            x = layer(x)

        return x
    
    #最終の全結合層を処理中のデバイスを返す関数
    def get_device(self):
        return self.output_layer.weight.device    

class Upblock(nn.Module):
    def __init__(self, inplanes, planes, dropout):
        super().__init__()

        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(dropout)

        if inplanes != planes:
            self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity # Residual connection
        out = self.relu(out)
        return out


class DecoderUpBlock(nn.Module):
    def __init__(self, x, y, enc_channels, down_channels, out_filters):
        super().__init__()
        
        self.enc_channels = enc_channels
        self.down_channels = down_channels
        self.out_filters = out_filters
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(x, y), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
            nn.Conv2d(self.enc_channels, self.down_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.down_channels),
            nn.ReLU(inplace=True)
        )
        # スキップコネクション後の畳み込みブロック
        # 結合後のチャンネル数: self.enc_2_channels (up) + self.enc_2_channels (skip)
        self.dec_block2 = Upblock(self.down_channels, self.out_filters, 0.2) # -> 64x128, 32
        self.dropout = nn.Dropout(0.3)

    def forward(self, in_img):
        # アップサンプリング
        dec2_up = self.upconv2(in_img) # -> (B, 64, 64, 128)

        dec2_out = self.dec_block2(dec2_up) # -> (B, 32, 64, 128)
        dec2_out = self.dropout(dec2_out)

        return dec2_out


class JunNet4(nn.Module):
    def __init__(self, in_channels, nclasses, drop=0, use_mps=True):
        super(JunNet4, self).__init__()
        self.nclasses = nclasses
        num_blocks = 3
        current_in_channels = in_channels # 最初の入力チャンネル数 (5)
        intermediate1_channels = [16, 32, 32] # 5つのブロックの出力チャンネル数
        intermediate2_channels = [32, 64, 64]

        first_block_layers1 = []
        first_block_layers2 = []
        for i in range(num_blocks):
            # 最後のブロックで最終的な出力チャンネル数に合わせる
            out_ch = intermediate1_channels[i]
            first_block_layers1.append(first_layer(current_in_channels, out_ch))
            current_in_channels = out_ch # 次のブロックの入力は現在のブロックの出力
        

        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        nextchannel = 32


        for i in range(3):
            # 最後のブロックで最終的な出力チャンネル数に合わせる
            out_ch = intermediate2_channels[i]
            first_block_layers2.append(first_layer(nextchannel, out_ch))
            nextchannel = out_ch # 次のブロックの入力は現在のブロックの出力
        
        self.first_blocks_sequential1 = nn.Sequential(*first_block_layers1)
        self.first_blocks_sequential2 = nn.Sequential(*first_block_layers2)


        self.transformer0 = VisionTransformer(img_size_h=64, img_size_w=512, in_channel=5, patch_size=16, num_inputlayer_units=512, num_heads=4, num_mlp_units=512, num_layers=2)  #出力(bs, 128+1, 512)
        self.transformer1 = VisionTransformer(img_size_h=64, img_size_w=256, in_channel=32, patch_size=8, num_inputlayer_units=512, num_heads=4, num_mlp_units=512, num_layers=2)  #出力(bs, 256+1, 512)
        self.transformer2 = VisionTransformer(img_size_h=64, img_size_w=128, in_channel=64, patch_size=4, num_inputlayer_units=512, num_heads=4, num_mlp_units=512, num_layers=2)  #出力(bs, 512+1, 512)


        self.upconv0 = DecoderUpBlock(x=4, y=8, enc_channels=512, down_channels=512, out_filters=256)
        self.upconv1 = DecoderUpBlock(x=4, y=4, enc_channels=512, down_channels=512, out_filters=256)
        self.upconv2 = DecoderUpBlock(x=4, y=2, enc_channels=512, down_channels=512, out_filters=256)

        self.resBlock1 = Upblock(256*3, 256, drop)

        self.final_upsample = DecoderUpBlock(x=1, y=8, enc_channels=256, down_channels=256, out_filters=128)
        self.final_conv = nn.Conv2d(128, nclasses, kernel_size=1) # 1x1 Convでクラス数に変換

        
    def forward(self, x):

        out0 = self.transformer0(x)
        vit_spatial0 = out0[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial0 = (512, 16, 8)
            out0.size(0), 512, 16, 8
        )


        res1 = self.first_blocks_sequential1(x)             #res1 = (32, 64, 512)
        down1 = self.pool(res1)                            #down1 = (32, 64, 256)

        out1 = self.transformer1(down1)
        vit_spatial1 = out1[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial1 = (512, 16, 16)
            out1.size(0), 512, 16, 16
        )


        res2 = self.first_blocks_sequential2(down1)             #res2 = (64, 64, 256)
        down2 = self.pool(res2)                            #down1 = (64, 64, 128)   

        out2 = self.transformer2(down2)
        vit_spatial2 = out2[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial2 = (512, 16, 32)
            out2.size(0), 512, 16, 32
        )



        dec0_up = self.upconv0(vit_spatial0)         #dec0_up = (256, 64, 64)
        dec1_up = self.upconv1(vit_spatial1)         #dec1_up = (256, 64, 64)
        dec2_up = self.upconv2(vit_spatial2)         #dec2_up = (256, 64, 64)

        dec_concat = torch.cat([dec0_up, dec1_up, dec2_up], dim=1) # -> (B, 256*3, 64, 64)
        dec1_out = self.resBlock1(dec_concat) # -> (B, 256, 64, 64)

        final_output = self.final_upsample(dec1_out) # -> (B, 128, 64, 512)
        logits = self.final_conv(final_output)

        return logits, {}