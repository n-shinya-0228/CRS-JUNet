# import torch
# import torch.nn as nn 
# from torch import optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms as transforms
# import timm


# BN_MOMENTUM = 0.1                                               #バッチ正規化のモーメンタム（平均の移動平均係数）を定義
# def get_group_mask(xyz, threshold=0.6, kernel_size=3, dilation=1, padding=1, stride=1):
#     N,C,H,W = xyz.size()                                                                    #入力xyzは[N, C=3, H, W]のテンソルで、C=3は点群のx, y, z座標。テンソルのサイズを取得。

#     center = xyz    #各ピクセル（点）の中心座標を保持。
#     #unfoldにより、各点の局所パッチを展開（展開後は[C, K, H, W]にreshape）。これは空間近傍を抽出している。
#     xyz = F.unfold(xyz, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride).view(N, C, kernel_size*kernel_size, H, W)  
#     group_xyz = xyz - center.unsqueeze(2)                                            #各近傍点から中心点を引き、局所ベクトルに変換。center.unsqueeze(2) で [N,3,1,H,W] にしている
#     dists = torch.sqrt(torch.sum(group_xyz*group_xyz, 1))                            #L2ノルム（ユークリッド距離）で距離を計算。5次元から4次元に(C=1になる)

#     mask_valid = (torch.sum(center*center, 1)>0).unsqueeze(1).repeat(1, kernel_size*kernel_size, 1, 1).float()          # 入力がゼロではない有効な点（NaN除外など）をマスクとして検出。
#     mask = (dists < threshold).float()                      #距離がしきい値以下の点のみを有効とするマスク。

#     dists = 1.0 / (dists + 1e-4)                    #小さい距離ほど重みが大きくなるように反転。
#     dists *= mask
#     dists *= mask_valid                               #無効な点（しきい値以上、またはゼロ）を除外。

#     norm = torch.sum(dists, dim=2, keepdim=True)+1e-4
#     weight = dists / norm                                               # 重みを正規化（各パッチ内で合計1になるように）

#     return weight, group_xyz            # 出力は2つ：重みテンソル（[N, K, H, W]）と、局所ベクトル（[N, 3, K, H, W]） これがのちのgroup_mask(l_xyz)とgroup_xyz


# class WeightNet(nn.Module):                     #このWeightNetは、後で紹介するPConvの中で使われ、点群の空間的な構造に基づいた重みづけを実現

#     def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):   #in_channelは入力チャネル（通常は3次元xyz）、out_channelは最終出力チャネル数。hidden_unitは中間層のノード数リスト
#         super(WeightNet, self).__init__()

#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()                          #MLP層とバッチ正規化層のリストを用意
#         if hidden_unit is None or len(hidden_unit) == 0:
#             self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))                # 隠れ層が指定されていない場合は1層だけ。
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
#         else:
#             self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))             #最初の隠れ層。
#             self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
#             for i in range(1, len(hidden_unit)):
#                 self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))             #残りの隠れ層を順に追加。
#                 self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
#             self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))         #最後の出力層を追加。
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
#     def forward(self, localized_xyz):               #forwardメソッドは特別でmodel=WeighNet()、output=model.forward()ではなく、output=model()で呼び出せる。
#         #xyz : BxCxKxN

#         weights = localized_xyz                                             #localized_xyzは [B, 3, K, H*W] のテンソル（近傍の相対座標）。
#         for i, conv in enumerate(self.mlp_convs):                           #それをMLPで処理し、重みとして出力。
#             bn = self.mlp_bns[i]
#             weights =  F.relu(bn(conv(weights)))                            #各近傍点の座標ベクトルに基づく学習可能な距離的重みを生成。

#         return weights


# class PConv(nn.Module):
#     def __init__(self, in_channel, mlp, kernel_size=3, dilation=1, padding=1, stride=1):    #mlp: MLPの出力チャネルのリスト（例：[64, 128]）
#         super(PConv, self).__init__()                                                                                   #group_operation: PMaxpooling等の関数を渡すことで集約操作を切り替え可能。
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.padding = padding
#         self.stride = stride

        
#         self.mlp_convs = nn.ModuleList()             # 入力特徴を変換するMLP（1×1 Conv）層を構築。PointNetで点ごとにMLPをかけるイメージと同じ。
#         self.mlp_bns = nn.ModuleList()
#         self.dropout = nn.Dropout(0.3)
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
#             last_channel = out_channel

#         self.linear = nn.Conv2d(16 * mlp[-1], mlp[-1], 1)            #group-wise weighted sum後に、チャネル圧縮のための1×1畳み込み
#         self.bn_linear = nn.BatchNorm2d(mlp[-1])

#         self.weightnet = WeightNet(3, 16)       # 先ほどのWeightNetを使って、近傍座標情報から16チャネルの空間的重みを生成

#     def forward(self, x, group_mask, group_xyz):
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             x = F.relu(bn(conv(x)))                       #入力特徴 x（形状: [B, C, H, W]）に対してMLP（1×1 Conv）で変換を施す。
#             x = self.dropout(x)


#         B, C, N, H, W = group_xyz.shape                   #group_xyz は [B, 3, K, H, W] の形で、近傍点の空間的な相対座標。
#         group_xyz = self.weightnet(group_xyz.view(B, C, N, -1)).view(B, -1, N, H, W)            #WeightNet を通して、各点の空間的重みを計算し、チャネル数を16倍に増やす。

#         B,C,H,W = x.size()
        
#         #入力特徴マップ x を近傍（パッチ）単位に展開する。これで x は [B, C, K, H, W] になり、各画素ごとの近傍が揃う。
#         x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride).view(B, C, self.kernel_size*self.kernel_size, group_mask.size(-2), group_mask.size(-1))

#         #x（特徴）と group_xyz（空間的重み）を内積して重み付け。結果は [B, H, W, C×16] の形に。
#         x = torch.matmul(input=x.permute(0, 3, 4, 1, 2), other=group_xyz.permute(0, 3, 4, 2, 1)).view(B, group_mask.size(-2), group_mask.size(-1), -1)
#         x = self.linear(x.permute(0, 3, 1, 2))
#         x = self.bn_linear(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         #x = self.group_operation(x, group_mask.unsqueeze(1))

#         return x


# class first_layer(nn.Module):
#     def __init__(self, inplanes, planes):
#         super().__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.dropout = nn.Dropout(0.3)
#         self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#         if inplanes != planes:
#             self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#             self.skip_bn = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         else:
#             self.skip_conv = None
#             self.skip_bn = None # Noneにしておくことで、forwardで分岐処理ができる
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         identity = x

#         # メインパス
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out) # 最初のReLUはConv1の後

        

#         out = self.conv2(out)
#         out = self.bn2(out) # 2番目のConvの後にBN
#         out = self.dropout(out)

#         # スキップ接続の処理
#         if self.skip_conv is not None:
#             identity = self.skip_conv(identity)
#             identity = self.skip_bn(identity) # スキップパスにもBNを適用することが多い

#         # 残差接続: メインパスの出力にスキップパスの出力を加算
#         out += identity
#         out = self.relu(out) # 残差接続の後にReLU

#         # out = self.pool(x)

#         return out
        


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, dropout, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
#         super(BasicBlock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
#         self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         self.bn0 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), dilation=(2, 2), bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.dropout = nn.Dropout(dropout)
#         # self.downsample = downsample                                    #スキップ接続用のダウンサンプルが必要かを保存。

        
#     def forward(self, x):
#         residual1 = self.conv0(x)                                        #入力を残しておく（スキップ接続用）。
#         residual1 = self.bn0(residual1)
#         # residual1 = x

#         out1 = self.conv1(x)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#         out1 = self.dropout(out1)

#         out1 = self.conv2(out1)
#         out1 = self.bn2(out1)
#         out1 = self.relu(out1)
#         out1 = self.dropout(out1)


#         # if self.downsample is not None:
#         #     residual = self.downsample(x)         #入出力のチャネル数などが合わない場合、residual も変換。

#         out1 += residual1
#         out1 = self.relu(out1)                    #スキップ接続して、ReLU活性化 → 出力。

#         return out1
    

# class Downblock(nn.Module):
#     def __init__(self, inplanes, planes, dropout, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
#         super(Downblock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
#         self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=(1, 2), stride=(1, 2), bias=False)     #ダウンサンプリング
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv4 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.bn4 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv5 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
#         self.bn5 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # down_x = x[:,:,:,::2]

#         residual2 = self.conv3(x)                                  #ダウンサンプリング
#         residual2 = self.bn3(residual2)

#         out2 = self.conv4(residual2)
#         out2 = self.bn4(out2)
#         out2 = self.relu(out2)
#         out2 = self.dropout(out2)

#         out2 = self.conv5(out2)
#         out2 = self.bn5(out2)
#         out2 = self.relu(out2)
#         out2 = self.dropout(out2)


#         out2 += residual2
#         # out2 += down_x
#         out2 = self.relu(out2)

#         return out2
    

# class MultiHeadSelfAttention(nn.Module):

#     def __init__(self, num_inputlayer_units: int, num_heads: int, dropout: float=0.3):   #num_inputlayer_units:全結合層のユニット数, num_heads:マルチヘッドアテンションのヘッド数
#         super().__init__()


#         if num_inputlayer_units % num_heads != 0:
#             raise ValueError("num_inputlayer_units must be divisible by num_heads")
        
#         self.num_heads = num_heads
#         #ヘッドごとの特徴量の次元の次元を求める(128)
#         dim_head = num_inputlayer_units // num_heads
#         #q,k,vを作るための全結合層
#         self.expansion_layer = nn.Linear(num_inputlayer_units, num_inputlayer_units * 3, bias=False)
#         #softmax関数のオーバーフロー対策
#         self.scale = 1 / (dim_head ** 0.5)

#         self.headjoin_layer =  nn.Linear(num_inputlayer_units, num_inputlayer_units)
#         self.dropout = nn.Dropout(dropout)


#     def forward(self, x: torch.Tensor):

#         #入力token(32, 5, 512)からバッチサイズとパッチ数を取得
#         bs, ns = x.shape[:2]
#         #入力(32, 5, 512)から出力(32, 5, 512*3), query,key,valueを作る。
#         qkv = self.expansion_layer(x)
#         #(32, 5, 3(query,key,value), 4(ヘッド数), 128(特徴量))に変換し、(3, 32, 4, 5, 128)
#         qkv = qkv.view(bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         #query,key,valueに分割する
#         q, k, v = qkv.unbind(0)

#         #qと転置したkを内積。(5, 128)・(128, 5)=(5, 5)
#         attn = q.matmul(k.transpose(-2, -1))
#         #アテンションスコアの行方向ソフトマックス関数を適用
#         attn = (attn * self.scale).softmax(dim=-1)
#         attn = self.dropout(attn)
#         #アテンションスコアとvを内積(5, 128)
#         x = attn.matmul(v)
#         #(32, 4, 5, 128)から(32, 5, 4, 128).flattenで(32, 5, 512)
#         x = x.permute(0, 2, 1, 3).flatten(2)
#         #(32, 5, 512)から(32, 5, 512)
#         x = self.headjoin_layer(x)

#         return x

# #multiheadattention終わった後に、MLPに通す
# class MLP(nn.Module):
#     def __init__(self, num_inputlayer_units: int, num_mlp_units: int, dropout: float=0.1):
#         super().__init__()

#         self.linear1 = nn.Linear(num_inputlayer_units, num_mlp_units)   #今回num_inputlayer_unitsとnum_mlp_unitsは同じ数
#         self.linear2 = nn.Linear(num_mlp_units, num_inputlayer_units)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         x = self.dropout(x)
#         return x
    
# #正規化層、アテンション層、正規化層、MLP
# class EncorderBlock(nn.Module):
#     def __init__(self, num_inputlayer_units: int, num_heads: int, num_mlp_units: int):
#         super().__init__()

#         self.attention = MultiHeadSelfAttention(num_inputlayer_units, num_heads)
#         self.mlp = MLP(num_inputlayer_units, num_mlp_units)
#         self.norm1 = nn.LayerNorm(num_inputlayer_units)
#         self.norm2 = nn.LayerNorm(num_inputlayer_units)

    
#     def forward(self, x: torch.Tensor):

#         x = self.norm1(x)
#         x = self.attention(x) + x
#         x = self.norm2(x)
#         x = self.mlp(x) + x

#         return x


# class VisionTransformer(nn.Module):

#     def __init__(self, img_size_h: int, img_size_w: int, patch_size: int, num_inputlayer_units: int, num_heads: int, num_mlp_units: int, num_layers: int):
#         super().__init__()

#         self.img_size_h = img_size_h
#         self.img_size_w = img_size_w
#         self.patch_size = patch_size

#         num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)     #全パッチ数
#         input_dim = 128 * patch_size ** 2                 #１パッチあたりの特徴量 

#         #特徴次元削減のため768から512にする全結合層を定義
#         self.input_layer = nn.Linear(input_dim, num_inputlayer_units)
#         #nn.Parameterでtensorを作ると、勾配の計算が可能になる
#         #クラストークンの生成、3階テンソル(1, 1, 512)を平均0, 標準偏差1でランダムに値を生成
#         self.class_token = nn.Parameter(torch.zeros(1, 1, num_inputlayer_units))  
#         #位置情報の生成、3階テンソル(1, パッチ数+1(クラストークン), 512)、これはtorch.catではなく加算用
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_inputlayer_units))
#         #num_layersの数だけエンコーダーブロックを作成(今回は３)
#         self.encoder_layer = nn.ModuleList([EncorderBlock(num_inputlayer_units, num_heads, num_mlp_units) for _ in range(num_layers)])

#         #出力時
#         self.normalize = nn.LayerNorm(num_inputlayer_units)
#         # self.output_layer = nn.Linear(num_inputlayer_units, num_classes)

#     def forward(self, x: torch.Tensor):
#         #ミニバッチサイズ32, チャネル3, 縦32, 横32
#         bs, c, h, w = x.shape
#         #(32, 3, 32, 32)を(32, 3, 2, 16, 2, 16)の6階テンソルに変換させる。これはパッチサイズ16*16に分割するコード、計４つのパッチができる
#         x = x.view(bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)

#         #順番を変える
#         x = x.permute(0, 2, 4, 1, 3, 5)
#         #分割したパッチごとにフラット化する
#         #(バッチサイズ, 全パッチ数2*2, １パッチ当たりのパッチデータ16*16*3)
#         x = x.reshape(bs, (h // self.patch_size)*(w // self.patch_size), -1)

#         x = self.input_layer(x)  #出力(バッチサイズ, 全パッチ数2*2, 512)

#         #バッチサイズ分クラストークンを拡張
#         class_token = self.class_token.expand(bs, -1, -1)
#         #dimで連結する次元を指定する。このときほかの次元は数があってないといけない。(32, 5, 512)になる。
#         x = torch.cat((class_token, x), dim=1)
#         #位置情報の付与
#         x += self.pos_embed
        
#         #encoderブロックに三回通す
#         for layer in self.encoder_layer:
#             x = layer(x)

#         return x
    
#     #最終の全結合層を処理中のデバイスを返す関数
#     def get_device(self):
#         return self.output_layer.weight.device    

# class Upblock(nn.Module):
#     def __init__(self, inplanes, planes):
#         super().__init__()

#         self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         self.bn0 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

#         if inplanes != planes:
#             self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1)
#         else:
#             self.skip_conv = None

#     def forward(self, x):
#         identity = x
#         if self.skip_conv is not None:
#             identity = self.skip_conv(identity)

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity # Residual connection
#         out = self.relu(out)
#         return out


# class Decoder(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         # エンコーダの各段階からのスキップコネクションの特徴量数を定義
#         self.enc_3_channels = 128 # 3ブロック目(ResNet) 64*64, 128
#         self.enc_2_channels = 64  # 2ブロック目(ResNet) 64*128, 64
#         self.enc_1_channels = 32  # 1ブロック目(ResNet) 64*256, 32
#         self.input_channels = 5   # 元の入力画像の特徴量

#         self.dropout = nn.Dropout(0.3)
#         # --- デコーダ 1ブロック目 ---
#         # ViTの出力 (64トークン, 512特徴量) を空間的な特徴マップ (8x8, 512) に変換
#         # 後でアップサンプリングして 64x64 にする
#         self.vit_channels = 768 # ViTの出力特徴量

#         # 8x8 から 64x64 へアップサンプリング (スケールファクター 8)
#         # Transposed Convを使う場合
#         # self.upconv1 = nn.ConvTranspose2d(self.vit_channels, self.enc_3_channels, kernel_size=8, stride=8)
#         # Upsample + Convを使う場合 (推奨)
#         self.upconv1 = nn.Sequential(
#             nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
#             nn.Conv2d(self.vit_channels, self.enc_3_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.enc_3_channels),
#             nn.ReLU(inplace=True)
#         )
#         # スキップコネクション後の畳み込みブロック
#         # 結合後のチャンネル数: self.enc_3_channels (ViTup) + self.enc_3_channels (skip)
#         self.dec_block1 = Upblock(self.enc_3_channels + self.enc_3_channels, self.enc_2_channels) # -> 64x64, 64

#         # --- デコーダ 2ブロック目 ---
#         # 64x64 から 64x128 へアップサンプリング (スケールファクター 2, 幅方向)
#         self.upconv2 = nn.Sequential(
#             nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
#             nn.Conv2d(self.enc_2_channels, self.enc_2_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.enc_2_channels),
#             nn.ReLU(inplace=True)
#         )
#         # スキップコネクション後の畳み込みブロック
#         # 結合後のチャンネル数: self.enc_2_channels (up) + self.enc_2_channels (skip)
#         self.dec_block2 = Upblock(self.enc_2_channels + self.enc_3_channels, self.enc_1_channels) # -> 64x128, 32

#         # --- デコーダ 3ブロック目 ---
#         # 64x128 から 64x256 へアップサンプリング (スケールファクター 2, 幅方向)
#         self.upconv3 = nn.Sequential(
#             nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
#             nn.Conv2d(self.enc_1_channels, self.enc_1_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.enc_1_channels),
#             nn.ReLU(inplace=True)
#         )
#         # スキップコネクション後の畳み込みブロック
#         # 結合後のチャンネル数: self.enc_1_channels (up) + self.enc_1_channels (skip)
#         self.dec_block3 = Upblock(self.enc_1_channels + self.enc_2_channels, self.input_channels) # -> 64x256, 5

#         # --- 最終出力層 ---
#         # 64x256 から 64x512 へアップサンプリング (スケールファクター 2, 幅方向)
#         self.final_upsample = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
#         self.final_conv = nn.Conv2d(self.input_channels, num_classes, kernel_size=1) # 1x1 Convでクラス数に変換

#     def forward(self, enc_outputs):
#         # enc_outputs はタプルまたはリストで、エンコーダの各段階の出力を含む
#         # 例: (enc1_out, enc2_out, enc3_out, vit_out)
#         enc1_out, enc2_out, enc3_out, vit_out = enc_outputs

#         # --- デコーダ 1ブロック目 ---
#         # ViT出力の空間再構成
#         # vit_out 形状: (batch_size, 65, 512)
#         # CLSトークン (vit_out[:, 0, :]) を無視し、残りの64トークンを使用
#         # (batch_size, 64, 512) -> (batch_size, 512, 8, 8)
#         # sqrt(64) = 8
#         vit_spatial = vit_out[:, 1:, :].transpose(1, 2).reshape(
#             vit_out.size(0), self.vit_channels, 8, 8
#         )
#         # アップサンプリング
#         dec1_up = self.upconv1(vit_spatial) # -> (B, 128, 64, 64)

#         # スキップコネクションと結合
#         # enc3_out 形状: (B, 128, 64, 64)
#         dec1_concat = torch.cat([dec1_up, enc3_out], dim=1) # -> (B, 128+128, 64, 64)
#         dec1_out = self.dec_block1(dec1_concat) # -> (B, 64, 64, 64)

#         # --- デコーダ 2ブロック目 ---
#         # アップサンプリング
#         dec2_up = self.upconv2(dec1_out) # -> (B, 64, 64, 128)

#         # スキップコネクションと結合
#         # enc2_out 形状: (B, 64, 64, 128)
#         dec2_concat = torch.cat([dec2_up, enc2_out], dim=1) # -> (B, 64+64, 64, 128)
#         dec2_out = self.dec_block2(dec2_concat) # -> (B, 32, 64, 128)
#         dec2_out = self.dropout(dec2_out)

#         # --- デコーダ 3ブロック目 ---
#         # アップサンプリング
#         dec3_up = self.upconv3(dec2_out) # -> (B, 32, 64, 256)

#         # スキップコネクションと結合
#         # enc1_out 形状: (B, 32, 64, 256)
#         dec3_concat = torch.cat([dec3_up, enc1_out], dim=1) # -> (B, 32+32, 64, 256)
#         dec3_out = self.dec_block3(dec3_concat) # -> (B, 5, 64, 256)
#         dec3_out = self.dropout(dec3_out)

#         # --- 最終出力層 ---
#         final_output = self.final_upsample(dec3_out) # -> (B, 5, 64, 512)
#         logits = self.final_conv(final_output)       # -> (B, num_classes, 64, 512)

#         return logits

# class DecoderUpBlock(nn.Module):
#     def __init__(self, enc_channels, down_channels, skip_rate, out_filters):
#         super().__init__()
        
#         self.enc_channels = enc_channels
#         self.down_channels = down_channels
#         self.out_filters = out_filters
#         self.skip_rate = skip_rate
#         self.upconv2 = nn.Sequential(
#             nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
#             nn.Conv2d(self.enc_channels, self.down_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.down_channels),
#             nn.ReLU(inplace=True)
#         )
#         # スキップコネクション後の畳み込みブロック
#         # 結合後のチャンネル数: self.enc_2_channels (up) + self.enc_2_channels (skip)
#         self.dec_block2 = Upblock(self.down_channels + self.skip_rate, self.out_filters) # -> 64x128, 32
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, in_img, skip):
#         # アップサンプリング
#         dec2_up = self.upconv2(in_img) # -> (B, 64, 64, 128)

#         # スキップコネクションと結合
#         # enc2_out 形状: (B, 64, 64, 128)
#         dec2_concat = torch.cat([dec2_up, skip], dim=1) # -> (B, 64+64, 64, 128)
#         dec2_out = self.dec_block2(dec2_concat) # -> (B, 32, 64, 128)
#         dec2_out = self.dropout(dec2_out)

#         return dec2_out


# class JunNet(nn.Module):
#     def __init__(self, in_channels, nclasses, drop=0, use_mps=True):
#         super(JunNet, self).__init__()
#         self.nclasses = nclasses
#         num_blocks = 8
#         current_in_channels = in_channels # 最初の入力チャンネル数 (5)
#         intermediate1_channels = [8, 8, 16, 16, 24, 24, 32, 32] # 5つのブロックの出力チャンネル数
#         intermediate2_channels = [32, 32, 32]

#         first_block_layers1 = []
#         first_block_layers2 = []
#         for i in range(num_blocks):
#             # 最後のブロックで最終的な出力チャンネル数に合わせる
#             out_ch = intermediate1_channels[i]
#             first_block_layers1.append(first_layer(current_in_channels, out_ch))
#             current_in_channels = out_ch # 次のブロックの入力は現在のブロックの出力
        

#         self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#         nextchannel = 32


#         for i in range(3):
#             # 最後のブロックで最終的な出力チャンネル数に合わせる
#             out_ch = intermediate2_channels[i]
#             first_block_layers2.append(first_layer(nextchannel, out_ch))
#             nextchannel = out_ch # 次のブロックの入力は現在のブロックの出力
        
#         self.first_blocks_sequential1 = nn.Sequential(*first_block_layers1)
#         self.first_blocks_sequential2 = nn.Sequential(*first_block_layers2)

#         # self.first_block = first_layer(5, 32)        #BasicBlockを8個積んだ最初の浅い層（C=5 → C=32）
#         # self.pconv0 = PConv(32, [32,64])
#         self.resBlock1 = BasicBlock(32, 2 * 32, drop)
#         self.downBlock1 = Downblock(2*  32, 2 * 32, drop)
#         # self.pconv1 = PConv(64, [64,128])
#         self.resBlock2 = BasicBlock(2 * 32, 2 * 2 * 32, drop)
#         self.downBlock2 = Downblock(2 * 2 * 32, 2 * 2 * 32, drop)
#         self.resBlock3 = BasicBlock(2*2 * 32, 2 * 2 * 32, drop)
#         # self.pconv2 = PConv(128, 128)

#         self.transformer = VisionTransformer(img_size_h=64, img_size_w=64, patch_size=4, num_inputlayer_units=512, num_heads=4, num_mlp_units=512, num_layers=2)  #出力(bs, 256+1, 512)

#         # ViT モデル（timm）で事前学習済み
#         # self.conv128 = nn.Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         # self.vit = timm.create_model(
#         #     "vit_base_patch8_224",   # ← timm にある正式名称
#         #     pretrained=True,
#         #     img_size=64,            # ← 64×64 を明示
#         #     num_classes=0,           # ← 分類ヘッドを最初から外しておく書き方
#         #     # in_chans=128
#         # )
#         # self.vit.head = nn.Identity()  # 分類層を削除

#         # self.decoder = Decoder(nclasses)
#         self.upconv1 = nn.Sequential(
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
#             nn.Conv2d(512, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.dec_block1 = Upblock(128 + 128, 128) # -> 64x64, 64

#         self.upconv2 = DecoderUpBlock(enc_channels=128, down_channels=64, skip_rate=64, out_filters=64)
#         self.upconv3 = DecoderUpBlock(enc_channels=64, down_channels=32, skip_rate=32, out_filters=5)
#         self.final_upsample = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
#         self.final_conv = nn.Conv2d(in_channels, nclasses, kernel_size=1) # 1x1 Convでクラス数に変換

#         # --- 境界検出ヘッドを追加 ---
#         self.boundary_head = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=1) # 最終的に1チャネル（境界か否か）を出力
#         )
        
#     def forward(self, x):
#         # xyz_0 = x[:, 1:4, :, :]
#         # xyz_1 = xyz_0[:,:,:,::2]
#         # l0_xyz, group0_xyz= get_group_mask(xyz_1, 0.2, kernel_size=3, padding=1)
#         # xyz_2 = xyz_1[:,:,:,::2] 
#         # l1_xyz, group1_xyz= get_group_mask(xyz_2, 0.2, kernel_size=3, padding=1)       #入力xyzは[N, C=3, H, W]のテンソル
#         # xyz_3 = xyz_2[:,:,:,::2] 
#         # l2_xyz, group2_xyz= get_group_mask(xyz_3, 0.2, kernel_size=3, padding=1)       #入力xyzは[N, C=3, H, W]のテンソル

#         # downCntx = self.first_block(x[:, 1::, :, :])             #downCntx = (32, 64, 256)
#         downCntx = self.first_blocks_sequential1(x)             #downCntx = (32, 64, 512)
#         downCntx = self.pool(downCntx)
#         # downCntx = downCntx[:,:,:,::2]
#         # downCntx = self.first_blocks_sequential2(downCntx)      #downCntx = (32, 64, 256)
#         # weight0 = self.pconv0(downCntx, l0_xyz, group0_xyz)         #weight0 = (64, 64, 256)
#         skip0 = self.resBlock1(downCntx)           #skip0 = (64, 64, 256)
#         down0 = self.downBlock1(skip0)             #down0 = (64, 64, 128)  
#         # weight1 = self.pconv1(down0, l1_xyz, group1_xyz)            #weight1 = (128, 64, 128)
#         skip1 = self.resBlock2(down0)              #skip1 = (128, 64, 128)
#         down1 = self.downBlock2(skip1)             #down1 = (128, 64, 64)
#         # weight2 = self.pconv2(down1, l2_xyz, group2_xyz)
#         skip2 = self.resBlock3(down1)


#         down2 = self.transformer(skip2)
#         vit_spatial = down2[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial = (512, 16, 16)
#             down2.size(0), 512, 16, 16
#         )
#         # a = self.conv128(down1)
#         # down2 = self.vit.forward_features(a)

#         # vit_spatial = down2[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial = (512, 16, 16)
#         #     down2.size(0), 768, 8, 8
#         # )





#         dec1_up = self.upconv1(vit_spatial)         #dec1_up = (128, 64, 64)
#         dec1_concat = torch.cat([dec1_up, down1], dim=1) # -> (B, 128+128, 64, 64)
#         dec1_out = self.dec_block1(dec1_concat) # -> (B, 128, 64, 64)
        
#         dec2_out = self.upconv2(dec1_out, down0) # -> (B, 64, 64, 128)
#         dec3_out = self.upconv3(dec2_out, downCntx)    # -> (B, 5, 64, 256)

#         final_output = self.final_upsample(dec3_out) # -> (B, 5, 64, 512)
#         logits = self.final_conv(final_output)


#         # --- 境界検出ヘッドのフォワードパスを追加 ---
#         boundary_logits = self.boundary_head(x)

#         # skips辞書に境界検出の出力を追加
#         skips = {}
#         skips["boundary"] = boundary_logits

#         # logitsと更新されたskips辞書を返す
#         return logits, skips

#         # logits = F.softmax(logits, dim=1)
#         # return logits, {}

import torch
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm


BN_MOMENTUM = 0.1                                               #バッチ正規化のモーメンタム（平均の移動平均係数）を定義
def get_group_mask(xyz, threshold=0.6, kernel_size=3, dilation=1, padding=1, stride=1):
    N,C,H,W = xyz.size()                                                                    #入力xyzは[N, C=3, H, W]のテンソルで、C=3は点群のx, y, z座標。テンソルのサイズを取得。

    center = xyz    #各ピクセル（点）の中心座標を保持。
    #unfoldにより、各点の局所パッチを展開（展開後は[C, K, H, W]にreshape）。これは空間近傍を抽出している。
    xyz = F.unfold(xyz, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride).view(N, C, kernel_size*kernel_size, H, W)  
    group_xyz = xyz - center.unsqueeze(2)                                            #各近傍点から中心点を引き、局所ベクトルに変換。center.unsqueeze(2) で [N,3,1,H,W] にしている
    dists = torch.sqrt(torch.sum(group_xyz*group_xyz, 1))                            #L2ノルム（ユークリッド距離）で距離を計算。5次元から4次元に(C=1になる)

    mask_valid = (torch.sum(center*center, 1)>0).unsqueeze(1).repeat(1, kernel_size*kernel_size, 1, 1).float()          # 入力がゼロではない有効な点（NaN除外など）をマスクとして検出。
    mask = (dists < threshold).float()                      #距離がしきい値以下の点のみを有効とするマスク。

    dists = 1.0 / (dists + 1e-4)                    #小さい距離ほど重みが大きくなるように反転。
    dists *= mask
    dists *= mask_valid                               #無効な点（しきい値以上、またはゼロ）を除外。

    norm = torch.sum(dists, dim=2, keepdim=True)+1e-4
    weight = dists / norm                                               # 重みを正規化（各パッチ内で合計1になるように）

    return weight, group_xyz            # 出力は2つ：重みテンソル（[N, K, H, W]）と、局所ベクトル（[N, 3, K, H, W]） これがのちのgroup_mask(l_xyz)とgroup_xyz


class WeightNet(nn.Module):                     #このWeightNetは、後で紹介するPConvの中で使われ、点群の空間的な構造に基づいた重みづけを実現

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):   #in_channelは入力チャネル（通常は3次元xyz）、out_channelは最終出力チャネル数。hidden_unitは中間層のノード数リスト
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()                          #MLP層とバッチ正規化層のリストを用意
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))                # 隠れ層が指定されていない場合は1層だけ。
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))             #最初の隠れ層。
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))             #残りの隠れ層を順に追加。
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))         #最後の出力層を追加。
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):               #forwardメソッドは特別でmodel=WeighNet()、output=model.forward()ではなく、output=model()で呼び出せる。
        #xyz : BxCxKxN

        weights = localized_xyz                                             #localized_xyzは [B, 3, K, H*W] のテンソル（近傍の相対座標）。
        for i, conv in enumerate(self.mlp_convs):                           #それをMLPで処理し、重みとして出力。
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))                            #各近傍点の座標ベクトルに基づく学習可能な距離的重みを生成。

        return weights


class PConv(nn.Module):
    def __init__(self, in_channel, mlp, kernel_size=3, dilation=1, padding=1, stride=1):    #mlp: MLPの出力チャネルのリスト（例：[64, 128]）
        super(PConv, self).__init__()                                                                                   #group_operation: PMaxpooling等の関数を渡すことで集約操作を切り替え可能。
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        
        self.mlp_convs = nn.ModuleList()             # 入力特徴を変換するMLP（1×1 Conv）層を構築。PointNetで点ごとにMLPをかけるイメージと同じ。
        self.mlp_bns = nn.ModuleList()
        self.dropout = nn.Dropout(0.3)
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.linear = nn.Conv2d(16 * mlp[-1], mlp[-1], 1)            #group-wise weighted sum後に、チャネル圧縮のための1×1畳み込み
        self.bn_linear = nn.BatchNorm2d(mlp[-1])

        self.weightnet = WeightNet(3, 16)       # 先ほどのWeightNetを使って、近傍座標情報から16チャネルの空間的重みを生成

    def forward(self, x, group_mask, group_xyz):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))                       #入力特徴 x（形状: [B, C, H, W]）に対してMLP（1×1 Conv）で変換を施す。
            x = self.dropout(x)


        B, C, N, H, W = group_xyz.shape                   #group_xyz は [B, 3, K, H, W] の形で、近傍点の空間的な相対座標。
        group_xyz = self.weightnet(group_xyz.view(B, C, N, -1)).view(B, -1, N, H, W)            #WeightNet を通して、各点の空間的重みを計算し、チャネル数を16倍に増やす。

        B,C,H,W = x.size()
        
        #入力特徴マップ x を近傍（パッチ）単位に展開する。これで x は [B, C, K, H, W] になり、各画素ごとの近傍が揃う。
        x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride).view(B, C, self.kernel_size*self.kernel_size, group_mask.size(-2), group_mask.size(-1))

        #x（特徴）と group_xyz（空間的重み）を内積して重み付け。結果は [B, H, W, C×16] の形に。
        x = torch.matmul(input=x.permute(0, 3, 4, 1, 2), other=group_xyz.permute(0, 3, 4, 2, 1)).view(B, group_mask.size(-2), group_mask.size(-1), -1)
        x = self.linear(x.permute(0, 3, 1, 2))
        x = self.bn_linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        #x = self.group_operation(x, group_mask.unsqueeze(1))

        return x


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
    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

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

        out += identity # Residual connection
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # エンコーダの各段階からのスキップコネクションの特徴量数を定義
        self.enc_3_channels = 128 # 3ブロック目(ResNet) 64*64, 128
        self.enc_2_channels = 64  # 2ブロック目(ResNet) 64*128, 64
        self.enc_1_channels = 32  # 1ブロック目(ResNet) 64*256, 32
        self.input_channels = 5   # 元の入力画像の特徴量

        self.dropout = nn.Dropout(0.3)
        # --- デコーダ 1ブロック目 ---
        # ViTの出力 (64トークン, 512特徴量) を空間的な特徴マップ (8x8, 512) に変換
        # 後でアップサンプリングして 64x64 にする
        self.vit_channels = 768 # ViTの出力特徴量

        # 8x8 から 64x64 へアップサンプリング (スケールファクター 8)
        # Transposed Convを使う場合
        # self.upconv1 = nn.ConvTranspose2d(self.vit_channels, self.enc_3_channels, kernel_size=8, stride=8)
        # Upsample + Convを使う場合 (推奨)
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(self.vit_channels, self.enc_3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.enc_3_channels),
            nn.ReLU(inplace=True)
        )
        # スキップコネクション後の畳み込みブロック
        # 結合後のチャンネル数: self.enc_3_channels (ViTup) + self.enc_3_channels (skip)
        self.dec_block1 = Upblock(self.enc_3_channels + self.enc_3_channels, self.enc_2_channels) # -> 64x64, 64

        # --- デコーダ 2ブロック目 ---
        # 64x64 から 64x128 へアップサンプリング (スケールファクター 2, 幅方向)
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
            nn.Conv2d(self.enc_2_channels, self.enc_2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.enc_2_channels),
            nn.ReLU(inplace=True)
        )
        # スキップコネクション後の畳み込みブロック
        # 結合後のチャンネル数: self.enc_2_channels (up) + self.enc_2_channels (skip)
        self.dec_block2 = Upblock(self.enc_2_channels + self.enc_3_channels, self.enc_1_channels) # -> 64x128, 32

        # --- デコーダ 3ブロック目 ---
        # 64x128 から 64x256 へアップサンプリング (スケールファクター 2, 幅方向)
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
            nn.Conv2d(self.enc_1_channels, self.enc_1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.enc_1_channels),
            nn.ReLU(inplace=True)
        )
        # スキップコネクション後の畳み込みブロック
        # 結合後のチャンネル数: self.enc_1_channels (up) + self.enc_1_channels (skip)
        self.dec_block3 = Upblock(self.enc_1_channels + self.enc_2_channels, self.input_channels) # -> 64x256, 5

        # --- 最終出力層 ---
        # 64x256 から 64x512 へアップサンプリング (スケールファクター 2, 幅方向)
        self.final_upsample = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(self.input_channels, num_classes, kernel_size=1) # 1x1 Convでクラス数に変換

    def forward(self, enc_outputs):
        # enc_outputs はタプルまたはリストで、エンコーダの各段階の出力を含む
        # 例: (enc1_out, enc2_out, enc3_out, vit_out)
        enc1_out, enc2_out, enc3_out, vit_out = enc_outputs

        # --- デコーダ 1ブロック目 ---
        # ViT出力の空間再構成
        # vit_out 形状: (batch_size, 65, 512)
        # CLSトークン (vit_out[:, 0, :]) を無視し、残りの64トークンを使用
        # (batch_size, 64, 512) -> (batch_size, 512, 8, 8)
        # sqrt(64) = 8
        vit_spatial = vit_out[:, 1:, :].transpose(1, 2).reshape(
            vit_out.size(0), self.vit_channels, 8, 8
        )
        # アップサンプリング
        dec1_up = self.upconv1(vit_spatial) # -> (B, 128, 64, 64)

        # スキップコネクションと結合
        # enc3_out 形状: (B, 128, 64, 64)
        dec1_concat = torch.cat([dec1_up, enc3_out], dim=1) # -> (B, 128+128, 64, 64)
        dec1_out = self.dec_block1(dec1_concat) # -> (B, 64, 64, 64)

        # --- デコーダ 2ブロック目 ---
        # アップサンプリング
        dec2_up = self.upconv2(dec1_out) # -> (B, 64, 64, 128)

        # スキップコネクションと結合
        # enc2_out 形状: (B, 64, 64, 128)
        dec2_concat = torch.cat([dec2_up, enc2_out], dim=1) # -> (B, 64+64, 64, 128)
        dec2_out = self.dec_block2(dec2_concat) # -> (B, 32, 64, 128)
        dec2_out = self.dropout(dec2_out)

        # --- デコーダ 3ブロック目 ---
        # アップサンプリング
        dec3_up = self.upconv3(dec2_out) # -> (B, 32, 64, 256)

        # スキップコネクションと結合
        # enc1_out 形状: (B, 32, 64, 256)
        dec3_concat = torch.cat([dec3_up, enc1_out], dim=1) # -> (B, 32+32, 64, 256)
        dec3_out = self.dec_block3(dec3_concat) # -> (B, 5, 64, 256)
        dec3_out = self.dropout(dec3_out)

        # --- 最終出力層 ---
        final_output = self.final_upsample(dec3_out) # -> (B, 5, 64, 512)
        logits = self.final_conv(final_output)       # -> (B, num_classes, 64, 512)

        return logits

class DecoderUpBlock(nn.Module):
    def __init__(self, enc_channels, down_channels, skip_rate, out_filters):
        super().__init__()
        
        self.enc_channels = enc_channels
        self.down_channels = down_channels
        self.out_filters = out_filters
        self.skip_rate = skip_rate
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False), # 高さ1倍、幅2倍
            nn.Conv2d(self.enc_channels, self.down_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.down_channels),
            nn.ReLU(inplace=True)
        )
        # スキップコネクション後の畳み込みブロック
        # 結合後のチャンネル数: self.enc_2_channels (up) + self.enc_2_channels (skip)
        self.dec_block2 = Upblock(self.down_channels + self.skip_rate, self.out_filters) # -> 64x128, 32
        self.dropout = nn.Dropout(0.3)

    def forward(self, in_img, skip):
        # アップサンプリング
        dec2_up = self.upconv2(in_img) # -> (B, 64, 64, 128)

        # スキップコネクションと結合
        # enc2_out 形状: (B, 64, 64, 128)
        dec2_concat = torch.cat([dec2_up, skip], dim=1) # -> (B, 64+64, 64, 128)
        dec2_out = self.dec_block2(dec2_concat) # -> (B, 32, 64, 128)
        dec2_out = self.dropout(dec2_out)

        return dec2_out


class JunNet(nn.Module):
    def __init__(self, in_channels, nclasses, drop=0, use_mps=True):
        super(JunNet, self).__init__()
        self.nclasses = nclasses
        num_blocks = 5
        current_in_channels = in_channels # 最初の入力チャンネル数 (5)
        intermediate1_channels = [8, 16, 24, 32, 32] # 5つのブロックの出力チャンネル数
        intermediate2_channels = [32, 32, 32]

        first_block_layers1 = []
        first_block_layers2 = []
        for i in range(num_blocks):
            # 最後のブロックで最終的な出力チャンネル数に合わせる
            out_ch = intermediate1_channels[i]
            first_block_layers1.append(first_layer(current_in_channels, out_ch))
            current_in_channels = out_ch # 次のブロックの入力は現在のブロックの出力
        

        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        nextchannel = 32


        for i in range(num_blocks-2):
            # 最後のブロックで最終的な出力チャンネル数に合わせる
            out_ch = intermediate2_channels[i]
            first_block_layers2.append(first_layer(nextchannel, out_ch))
            nextchannel = out_ch # 次のブロックの入力は現在のブロックの出力
        
        self.first_blocks_sequential1 = nn.Sequential(*first_block_layers1)
        self.first_blocks_sequential2 = nn.Sequential(*first_block_layers2)

        # self.first_block = first_layer(5, 32)        #BasicBlockを8個積んだ最初の浅い層（C=5 → C=32）
        # self.pconv0 = PConv(32, [32,64])
        self.resBlock1 = BasicBlock(32, 2 * 32, drop)
        self.downBlock1 = Downblock(2*  32, 2 * 32, drop)
        # self.pconv1 = PConv(64, [64,128])
        self.resBlock2 = BasicBlock(2 * 32, 2 * 2 * 32, drop)
        self.downBlock2 = Downblock(2 * 2 * 32, 2 * 2 * 32, drop)
        # self.pconv2 = PConv(128, 128)

        self.transformer = VisionTransformer(img_size_h=64, img_size_w=64, in_channel=128, patch_size=4, num_inputlayer_units=512, num_heads=4, num_mlp_units=512, num_layers=2)  #出力(bs, 256+1, 512)

        # ViT モデル（timm）で事前学習済み
        # self.vit = timm.create_model(
        #     "vit_base_patch8_224",   # ← timm にある正式名称
        #     pretrained=True,
        #     img_size=64,            # ← 64×64 を明示
        #     num_classes=0,           # ← 分類ヘッドを最初から外しておく書き方
        #     in_chans=128
        # )
        # self.vit.head = nn.Identity()  # 分類層を削除

        # self.decoder = Decoder(nclasses)
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec_block1 = Upblock(128 + 128, 128) # -> 64x64, 64

        self.upconv2 = DecoderUpBlock(enc_channels=128, down_channels=64, skip_rate=64, out_filters=64)
        self.upconv3 = DecoderUpBlock(enc_channels=64, down_channels=32, skip_rate=32, out_filters=5)
        self.final_upsample = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(in_channels, nclasses, kernel_size=1) # 1x1 Convでクラス数に変換

                # --- 境界検出ヘッドを追加 ---
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1) # 最終的に1チャネル（境界か否か）を出力
        )

        
    def forward(self, x):
        # xyz_0 = x[:, 1:4, :, :]
        # xyz_1 = xyz_0[:,:,:,::2]
        # l0_xyz, group0_xyz= get_group_mask(xyz_1, 0.2, kernel_size=3, padding=1)
        # xyz_2 = xyz_1[:,:,:,::2] 
        # l1_xyz, group1_xyz= get_group_mask(xyz_2, 0.2, kernel_size=3, padding=1)       #入力xyzは[N, C=3, H, W]のテンソル
        # xyz_3 = xyz_2[:,:,:,::2] 
        # l2_xyz, group2_xyz= get_group_mask(xyz_3, 0.2, kernel_size=3, padding=1)       #入力xyzは[N, C=3, H, W]のテンソル

        # downCntx = self.first_block(x[:, 1::, :, :])             #downCntx = (32, 64, 256)
        downCntx = self.first_blocks_sequential1(x)             #downCntx = (32, 64, 512)
        downCntx = self.pool(downCntx)
        # downCntx = downCntx[:,:,:,::2]
        downCntx = self.first_blocks_sequential2(downCntx)      #downCntx = (32, 64, 256)
        # weight0 = self.pconv0(downCntx, l0_xyz, group0_xyz)         #weight0 = (64, 64, 256)
        skip0 = self.resBlock1(downCntx)           #skip0 = (64, 64, 256)
        down0 = self.downBlock1(skip0)             #down0 = (64, 64, 128)  
        # weight1 = self.pconv1(down0, l1_xyz, group1_xyz)            #weight1 = (128, 64, 128)
        skip1 = self.resBlock2(down0)              #skip1 = (128, 64, 128)
        down1 = self.downBlock2(skip1)             #down1 = (128, 64, 64)
        # weight2 = self.pconv2(down1, l2_xyz, group2_xyz)


        down2 = self.transformer(down1)
        vit_spatial = down2[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial = (512, 16, 16)
            down2.size(0), 512, 16, 16
        )

        # down2 = self.vit.forward_features(down1)

        # vit_spatial = down2[:, 1:, :].transpose(1, 2).reshape(       #vit_spatial = (512, 16, 16)
        #     down2.size(0), 768, 8, 8
        # )




        dec1_up = self.upconv1(vit_spatial)         #dec1_up = (128, 64, 64)
        dec1_concat = torch.cat([dec1_up, down1], dim=1) # -> (B, 128+128, 64, 64)
        dec1_out = self.dec_block1(dec1_concat) # -> (B, 128, 64, 64)
        
        dec2_out = self.upconv2(dec1_out, down0) # -> (B, 64, 64, 128)
        dec3_out = self.upconv3(dec2_out, downCntx)    # -> (B, 5, 64, 256)

        final_output = self.final_upsample(dec3_out) # -> (B, 5, 64, 512)
        logits = self.final_conv(final_output)


        # skip = (skip0, skip1, down1, down2)
        # logits = self.decoder(skip)

        # logits = F.softmax(logits, dim=1)
        # --- 境界検出ヘッドのフォワードパスを追加 ---

        boundary_logits = self.boundary_head(x)

        # skips辞書に境界検出の出力を追加
        skips = {}
        skips["boundary"] = boundary_logits

        # logitsと更新されたskips辞書を返す
        return logits, skips
        # return logits, {}