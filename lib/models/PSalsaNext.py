import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


def PMaxpooling(feat, group_mask):
    feat = feat*(group_mask != 0).float()      #feat: [B, C, K, H, W] の特徴マップ, group_mask: 各点に対して有効なK個の近傍を持つマスク。
    feat, _ = torch.max(feat, 2)

    return feat                                 #無効な点を0にしてから K 次元（近傍）で最大値を取る → 出力は [B, C, H, W]。

def PAvgpooling(feat, group_mask):
    feat = feat*group_mask
    feat = torch.sum(feat, 2)

    return feat

def PSumpooling(feat, group_mask):
    feat = feat*(group_mask != 0).float()
    feat, _ = torch.sum(feat, 2)

    return feat

class PDownsample(nn.Module):                           #ダウンサンプリング（下の解像度へ縮小）用クラス。factor=2なら、幅・高さを半分にする
    def __init__(self, factor):
        super(PDownsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        N, C, W, H = x.size()
        new = torch.zeros([N, C, int(W/self.factor), int(H/self.factor)]).cuda()
        #new = torch.zeros([N, C, int(W/self.factor), int(H/self.factor)])
        new = x[:, :, ::self.factor, ::self.factor]                         #「縦方向（高さ）と横方向（幅）の両方で、factorおきにサンプリング

        return new


# class PUpsample(nn.Module):                             #アップサンプリング（画像サイズを拡大）するクラス。factor=2なら、2倍に拡大。このPUpsampleは、後のデコーダ部（UpBlock）で使われます。

#     def __init__(self, factor):
#         super(PUpsample, self).__init__()
#         self.factor = factor

#     def forward(self, x):
#         N, C, W, H = x.size()
#         new = torch.zeros([N, C, W*self.factor, H*self.factor]).cuda()    #ゼロテンソルを用意し、元のテンソルをそのままスライスにコピーする形で拡大。
#         #new = torch.zeros([N, C, W*self.factor, H*self.factor])
#         new[:, :, ::self.factor, ::self.factor] = x

#         return new

class PUpsample(nn.Module):
    """画像のアップサンプリングを F.interpolate で行うクラス"""
    def __init__(self, factor, mode='nearest', align_corners=None):
        super(PUpsample, self).__init__()
        self.factor = factor
        self.mode = mode
        # bilinear や bicubic を使う場合には align_corners の設定が必要になることがあります
        self.align_corners = align_corners

    def forward(self, x):
        # 例: Nearest Neighbor
        return F.interpolate(
            x,
            scale_factor=self.factor,
            mode=self.mode,
            align_corners=self.align_corners
        )


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
    def __init__(self, in_channel, mlp, kernel_size=3, dilation=1, padding=1, stride=1, group_operation=PMaxpooling):    #mlp: MLPの出力チャネルのリスト（例：[64, 128]）
        super(PConv, self).__init__()                                                                                   #group_operation: PMaxpooling等の関数を渡すことで集約操作を切り替え可能。
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.group_operation = group_operation                  #畳み込みのハイパーパラメータを保存。

        
        self.mlp_convs = nn.ModuleList()             # 入力特徴を変換するMLP（1×1 Conv）層を構築。PointNetで点ごとにMLPをかけるイメージと同じ。
        self.mlp_bns = nn.ModuleList()
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
        #x = self.group_operation(x, group_mask.unsqueeze(1))

        return x

class PDConv(nn.Module):                            #PConv の対になる、アップサンプリング版の畳み込みです。
    def __init__(self, in_channel, mlp, kernel_size=3, padding=1, factor=1, group_operation=PMaxpooling):     #factor: スケール倍率（通常2）
        super(PDConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding=padding
        # self.up = PUpsample(factor)               #解像度を上げるために PUpsample を内部に持ちます。
        self.up = PUpsample(factor, mode='nearest')
        self.group_operation = group_operation
        
        #self.mlp_convs = nn.ModuleList()
        #self.mlp_bns = nn.ModuleList()
        #last_channel = in_channel
        #for out_channel in mlp:
        #    self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #    self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        #    last_channel = out_channel

    def forward(self, x_pre, x, group_mask):               #入力 x を PUpsample により2倍に拡大（スパースな拡大）
        x = self.up(x)
        B, C, H, W = x.size()
        x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding).view(B, C, self.kernel_size*self.kernel_size, group_mask.size(-2), group_mask.size(-1))
        x = self.group_operation(x, group_mask.unsqueeze(1))

        if isinstance(x_pre, torch.Tensor):        #skip connection の入力 x_pre（エンコーダの出力）と結合。
            x = torch.cat([x, x_pre], 1)
        #for i, conv in enumerate(self.mlp_convs):
        #    bn = self.mlp_bns[i]
        #    x = F.relu(bn(conv(x)))

        return x                                  #集約された特徴＋skip結合結果を出力。

# class PDConv(nn.Module):
#     def __init__(self, in_channels, mlp, kernel_size=3, padding=1, factor=2, group_operation=None):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.up = PUpsample(factor, mode='nearest')
#         self.group_op = group_operation or (lambda x, mask: x.sum(dim=2))
#         # MLP for pre-unfold features
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_ch = in_channels
#         for out_ch in mlp:
#             self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
#             self.mlp_bns.append(nn.BatchNorm2d(out_ch))
#             last_ch = out_ch
#         # WeightNet and output compression
#         self.weightnet = WeightNet(3, 16)
#         self.linear = nn.Conv2d(last_ch * 16, last_ch, 1)
#         self.bn_linear = nn.BatchNorm2d(last_ch)

#     def forward(self, x_skip, x_low, group_mask, group_xyz):
#         # 1) Upsample
#         x = self.up(x_low)  # (B, C, H2, W2)
#         # 2) Pre-MLP on features
#         for conv, bn in zip(self.mlp_convs, self.mlp_bns):
#             x = F.relu(bn(conv(x)))  # still (B, C', H2, W2)
#         B, C, H2, W2 = x.shape
#         # 3) unfold into patches
#         patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
#         K = self.kernel_size * self.kernel_size
#         patches = patches.view(B, C, K, H2, W2)
#         # 4) mask invalid
#         valid = group_mask.unsqueeze(1)  # (B,1,K,H2,W2)
#         patches = patches * valid
#         # 5) compute dynamic weights
#         w = self.weightnet(group_xyz.view(B, 3, K, -1)).view(B, 16, K, H2, W2)
#         # 6) weighted sum
#         feat = patches.permute(0,3,4,1,2)  # (B,H2,W2,C,K)
#         wt   = w.permute(0,3,4,2,1)       # (B,H2,W2,K,16)
#         out = torch.matmul(feat, wt)      # (B,H2,W2,C,16)
#         out = out.view(B, C*16, H2, W2)
#         # 7) compress
#         out = F.relu(self.bn_linear(self.linear(out)))  # (B,C',H2,W2)
#         # 8) skip connection
#         return torch.cat([out, x_skip], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):   #inplanes: 入力チャネル数, planes: 出力チャネル数, stride: 畳み込みのストライド（通常1）
        super(BasicBlock, self).__init__()                              #downsample: スキップ接続でチャネルやサイズが合わない場合の補正モジュール
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample                                    #スキップ接続用のダウンサンプルが必要かを保存。
        self.stride = stride

    def forward(self, x):
        residual = x                                        #入力を残しておく（スキップ接続用）。

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)         #入出力のチャネル数などが合わない場合、residual も変換。

        out += residual
        out = self.relu(out)                    #スキップ接続して、ReLU活性化 → 出力。

        return out



# 使われていない
class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):         #入力チャネル in_filters から出力チャネル out_filters に変換。
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)      # 入力を圧縮し、活性化する初期処理（スキップ接続にも使われる）。
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)              #通常の畳み込み処理（局所特徴抽出）。
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)       #拡張畳み込みにより、受容野を広げた特徴抽出を行う。
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)            # 入力に対して1×1畳み込み＋活性化を適用（後で加算するためのスキップ接続）。
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)                 #通常畳み込みと拡張畳み込みを連続して適用。
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


#このブロックは UnpNet の resBlock1 ～ resBlock5 で使われ、深さ方向にセマンティックな情報を抽出・圧縮していく主役の構成要素です。
class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,          #in_filters, out_filters: 入出力のチャネル数, dropout_rate: Dropoutの割合
                 pooling=True, drop_out=True):                                                       #pooling: Downsamplingするか, drop_out: Dropoutを使うか
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)           #入力をそのまま出力チャネル数に合わせるための1×1畳み込み（スキップ接続用）
        self.act1 = nn.LeakyReLU()                                                        

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)               # 3種類の畳み込みを連続して適用（通常→拡張→さらに拡張）して、異なるスケールの特徴を取得。
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))          #異なる畳み込み結果（3種）をチャネル方向に結合し、1×1で圧縮。
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:                                                  # ダウンサンプリングしたい場合は、ドロップアウトをかけた後、stride=2のPConvを使って解像度を半分に。
            self.dropout = nn.Dropout2d(p=dropout_rate)
            #self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
            self.pconv_d = PConv(out_filters, [out_filters, out_filters, out_filters], kernel_size=5, dilation=1, padding=2, stride=2, group_operation=PMaxpooling)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)



    def forward(self, x, l_xyz, group_xyz):
        shortcut = self.conv1(x)                    # 入力を1×1で変換し、後で加算する準備。
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)                       #3段階の畳み込みを連続的に処理。
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)          #3つの中間層を結合し、1×1で圧縮 → shortcutと加算（residual learning）。
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:                                     # プーリングありなら、resA にドロップアウトをかけてからPConvでダウンサンプリング（stride=2）。
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            #resB = self.pool(resB)
            resB = self.pconv_d(resB, l_xyz, group_xyz)

            return resB, resA                               #出力は (ダウンサンプル後, ダウンサンプル前) の2つ（後者はスキップ接続用）
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True, in_filters_=512):      #in_filters: アップサンプル後の入力特徴チャネル数（= skip + up）, dropout_rate: Dropoutの割合
        super(UpBlock, self).__init__()                                                             #out_filters: 最終的に出力するチャネル数, in_filters_: Skip+Up後に畳み込みに入れるチャネル数（明示的指定）
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        

        #in_out_c = in_filters//4 + 2*out_filters
        #self.pconv_u = PDConv(in_filters, 512, [in_out_c, in_out_c, in_out_c], kernel_size=5, padding=2, factor=2, group_operation=PMaxpooling)
        #PDConvで：2倍アップサンプリング, skip特徴と結合, グループ操作で特徴を集約
        self.pconv_u = PDConv(in_filters, [in_filters, in_filters//4 + 2*out_filters], kernel_size=5, padding=2, factor=2, group_operation=PMaxpooling)

        self.conv1 = nn.Conv2d(in_filters_, out_filters, (3,3), padding=1)         #エンコーダと同様、受容野を拡張するための拡張畳み込みも使用。
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))      # 異なる畳み込み結果を結合 → 圧縮し、さらにDropout適用。
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        

    def forward(self, x, skip, l_xyz, group_xyz):
        upB = self.pconv_u(skip, x, l_xyz)                  #まず、x（前段のアップ結果）を2倍にアップサンプリングして、skip（スキップ接続）と融合。
        #upA = nn.PixelShuffle(2)(x)
        #if self.drop_out:
        #    upA = self.dropout1(upA)

        #upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)      #融合後にDropout。

        upE = self.conv1(upB)                   #通常 + 拡張 + 拡張畳み込みの3層で特徴を抽出。
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)          #3段階の特徴を結合し、1×1で圧縮。
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:                       #最終出力にDropoutをかけて返す。
            upE = self.dropout3(upE)

        return upE

# class UpBlock(nn.Module):
#     def __init__(self, low_chan: int, skip_chan: int, out_chan: int, dropout_rate: float, drop_out: bool = True):
#         super().__init__()
#         self.drop_out = drop_out
#         # PDConv: in_channels = low resolution feature channels
#         self.pconv_u = PDConv(
#             in_channels=low_chan,
#             mlp=[low_chan],
#             kernel_size=5,
#             padding=2,
#             factor=2,
#             group_operation=PMaxpooling
#         )
#         # conv1 takes concatenated channels: low + skip
#         concat_chan = low_chan + skip_chan
#         self.conv1 = nn.Conv2d(concat_chan, out_chan, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_chan)
#         self.act1 = nn.LeakyReLU()
#         self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, dilation=2, padding=2)
#         self.bn2 = nn.BatchNorm2d(out_chan)
#         self.act2 = nn.LeakyReLU()
#         self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=2, dilation=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_chan)
#         self.act3 = nn.LeakyReLU()
#         self.conv4 = nn.Conv2d(out_chan * 3, out_chan, kernel_size=1)
#         self.bn4 = nn.BatchNorm2d(out_chan)
#         self.act4 = nn.LeakyReLU()
#         self.dropout = nn.Dropout2d(dropout_rate)

#     def forward(self, x_low: torch.Tensor, x_skip: torch.Tensor, group_mask: torch.Tensor, group_xyz: torch.Tensor) -> torch.Tensor:
#         # 1) PDConv: upsample + dynamic conv + skip concat inside
#         upB = self.pconv_u(x_skip, x_low, group_mask, group_xyz)
#         if self.drop_out:
#             upB = self.dropout(upB)
#         # 2) Multi-scale convolution
#         upE1 = self.act1(self.bn1(self.conv1(upB)))
#         upE2 = self.act2(self.bn2(self.conv2(upE1)))
#         upE3 = self.act3(self.bn3(self.conv3(upE2)))
#         # 3) Concatenate and compress
#         concat = torch.cat([upE1, upE2, upE3], dim=1)
#         upE = self.act4(self.bn4(self.conv4(concat)))
#         if self.drop_out:
#             upE = self.dropout(upE)
#         return upE


# class UnpNet(nn.Module):
#     def __init__(self, in_channels, nclasses, drop=0, use_mps=True):
#         super(UnpNet, self).__init__()
#         self.nclasses = nclasses

#         self.block_in = self._make_layer(BasicBlock, in_channels, 32, 8, 1)

#         self.resBlock1 = ResBlock(32, 2 * 32, drop, pooling=True, drop_out=False)
#         self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, drop, pooling=True)
#         self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, drop, pooling=True)
#         self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, drop, pooling=True)
#         self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, drop, pooling=False)

#         self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, drop, in_filters_=512)
#         self.upBlock2 = UpBlock(4 * 32, 4 * 32, drop, in_filters_=384)
#         self.upBlock3 = UpBlock(4 * 32, 2 * 32, drop, in_filters_=256)
#         self.upBlock4 = UpBlock(2 * 32, 32, drop, drop_out=False, in_filters_=128)


#         self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
#             )

#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))

#         return nn.Sequential(*layers)



#     def forward(self, x):

#         xyz_0 = x[:, 1:4, :, :]
        
#         l0_xyz, group0_xyz= get_group_mask(xyz_0, 0.2, kernel_size=5, padding=2)
#         xyz_1 = F.interpolate(xyz_0, scale_factor=0.5, mode='nearest')
#         l1_xyz, group1_xyz= get_group_mask(xyz_1, 0.2, kernel_size=5, padding=2)
#         xyz_2 = F.interpolate(xyz_1, scale_factor=0.5, mode='nearest')
#         l2_xyz, group2_xyz = get_group_mask(xyz_2, 0.4, kernel_size=5, padding=2)
#         xyz_3 = F.interpolate(xyz_2, scale_factor=0.5, mode='nearest')
#         l3_xyz, group3_xyz = get_group_mask(xyz_3, 0.6, kernel_size=5, padding=2)
#         xyz_4 = F.interpolate(xyz_3, scale_factor=0.5, mode='nearest')
#         l4_xyz, group4_xyz = get_group_mask(xyz_4, 0.8, kernel_size=5, padding=2)

#         downCntx = self.block_in(x)    #BasicBlock による初段の畳み込み＋活性化＋残差接続で、チャネルを増やしつつ特徴抽出。

#         down0c, down0b = self.resBlock1(downCntx, l1_xyz, group1_xyz)
#         down1c, down1b = self.resBlock2(down0c, l2_xyz, group2_xyz)  #各ブロックの出力は (feature_map, skip_feature) のタプルで返され、後者がデコーダのスキップ接続に使われます
#         down2c, down2b = self.resBlock3(down1c, l3_xyz, group3_xyz)
#         down3c, down3b = self.resBlock4(down2c, l4_xyz, group4_xyz)
#         down5c = self.resBlock5(down3c, None, None)

#         up4e = self.upBlock1(down5c,down3b, l3_xyz, group3_xyz)
#         up3e = self.upBlock2(up4e, down2b, l2_xyz, group2_xyz)
#         up2e = self.upBlock3(up3e, down1b, l1_xyz, group1_xyz)
#         up1e = self.upBlock4(up2e, down0b, l0_xyz, group0_xyz)
#         logits = self.logits(up1e)

#         logits = logits
#         logits = F.softmax(logits, dim=1)
#         return logits, {}
    


    # === Transformer追加 ===
# class SimpleTransformerEncoder(nn.Module):
#     def __init__(self, embed_dim=256, num_heads=8, num_layers=4):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             batch_first=True,
#             dim_feedforward=embed_dim*2,
#             activation='gelu'
#         )
#         self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # ResBlock5 後の H×W = 4×64 = 256 トークン長
#         self.pos_embed = nn.Parameter(torch.randn(1, 4*64, embed_dim))

#     def forward(self, x):  # x: (B, N, C) where N should be 4*64
#         x = x + self.pos_embed[:, :x.size(1), :]
#         return self.encoder(x)


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        # 既存の設定に加え、dropout も明示指定
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 学習可能な 1D pos_embed に加え、2D sinusoidal PE も足し合わせる
        self.pos_embed_learned = nn.Parameter(torch.randn(1, 4*64, embed_dim))
        # 2D Sinusoidal PE を事前にバッファ化
        pe = self._build_2d_sin_pe(h=4, w=64, dim=embed_dim)  # (1, 4*64, embed_dim)
        self.register_buffer('pos_embed_sin', pe)

    def _build_2d_sin_pe(self, h, w, dim):
        pe = torch.zeros(h, w, dim)              # (h, w, dim)
        # 偶数インデックス用の係数ベクトル
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))  # (dim/2,)

        # 縦位置 y の PE
        y_pos = torch.arange(h).unsqueeze(1)      # (h, 1)
        y_pe  = torch.sin(y_pos * div_term)       # (h, dim/2)
        pe[:, :, 0::2] = y_pe.unsqueeze(1).expand(h, w, dim//2)

        # 横位置 x の PE
        x_pos = torch.arange(w).unsqueeze(1)      # (w, 1)
        x_pe  = torch.cos(x_pos * div_term)       # (w, dim/2)
        pe[:, :, 1::2] = x_pe.unsqueeze(0).expand(h, w, dim//2)

        return pe.view(1, h*w, dim)               # (1, N, dim)

    def forward(self, x):  # x: (B, N, C)
        # Learned PE + Sinusoidal PE
        x = x + self.pos_embed_learned[:, :x.size(1), :] + self.pos_embed_sin[:, :x.size(1), :]
        # 自己注意 → 出力
        return self.encoder(x)



# 既存のUnpNetクラスの末尾にTransformerを組み込む
class UnpNet(nn.Module):
    def __init__(self, in_channels, nclasses, drop=0, use_mps=True):
        super(UnpNet, self).__init__()
        self.nclasses = nclasses
        
        self.block_in = self._make_layer(BasicBlock, 5, 32, 8, 1)        #BasicBlockを8個積んだ最初の浅い層（C=5 → C=32）

        self.resBlock1 = ResBlock(32, 2 * 32, drop, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, drop, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, drop, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, drop, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, drop, pooling=False)

        # self.transformer = SimpleTransformerEncoder(embed_dim=2 * 4 * 32, num_heads=8, num_layers=4)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2, in_filters_=512)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2, in_filters_=384)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2, in_filters_=256)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False, in_filters_=128)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))            #最後に1×1 Convで、各ピクセルにクラスを割り当てる（ソフトマックス前）。

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        xyz_0 = x[:, 1:4, :, :]                                #入力xyzは[N, C=3, H, W]のテンソル

        l0_xyz, group0_xyz= get_group_mask(xyz_0, 0.2, kernel_size=5, padding=2)
        xyz_1 = F.interpolate(xyz_0, scale_factor=0.5, mode='nearest')
        l1_xyz, group1_xyz= get_group_mask(xyz_1, 0.2, kernel_size=5, padding=2)
        xyz_2 = F.interpolate(xyz_1, scale_factor=0.5, mode='nearest')
        l2_xyz, group2_xyz = get_group_mask(xyz_2, 0.4, kernel_size=5, padding=2)
        xyz_3 = F.interpolate(xyz_2, scale_factor=0.5, mode='nearest')
        l3_xyz, group3_xyz = get_group_mask(xyz_3, 0.6, kernel_size=5, padding=2)
        xyz_4 = F.interpolate(xyz_3, scale_factor=0.5, mode='nearest')
        l4_xyz, group4_xyz = get_group_mask(xyz_4, 0.8, kernel_size=5, padding=2)

        downCntx = self.block_in(x)
        down0c, down0b = self.resBlock1(downCntx, l1_xyz, group1_xyz)
        down1c, down1b = self.resBlock2(down0c, l2_xyz, group2_xyz)
        down2c, down2b = self.resBlock3(down1c, l3_xyz, group3_xyz)
        down3c, down3b = self.resBlock4(down2c, l4_xyz, group4_xyz)
        down5c = self.resBlock5(down3c, None, None)

        # Transformer適用 (Flatten → Transformer → Reshape)
        # B, C, H, W = down3c.shape
        # x_flat = down3c.flatten(2).transpose(1,2)  # (B, N, C)
        # x_trans = self.transformer(x_flat)
        # down5c = x_trans.transpose(1,2).view(B,C,H,W) 
        # down5c = down3c + down5c

        up4e = self.upBlock1(down5c,down3b, l3_xyz, group3_xyz)
        up3e = self.upBlock2(up4e, down2b, l2_xyz, group2_xyz)
        up2e = self.upBlock3(up3e, down1b, l1_xyz, group1_xyz)
        up1e = self.upBlock4(up2e, down0b, l0_xyz, group0_xyz)
        logits = self.logits(up1e)

        logits = F.softmax(logits, dim=1)
        return logits, {}

