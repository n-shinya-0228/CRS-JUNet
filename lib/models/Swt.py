import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.metrics
import ignite.cotrib.handlers

#パラメータ値
DATA_DIR = "./data"
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_WORKERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1

#使用可能なデバイス
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", DEVICE)

#トレーニングデータに適用する一連の変換操作をコンテナにまとめる
train_transform = transforms.Compose([
    #画像をランダムに左右反転
    transforms.RandomHorizontalFlip(),
    #画像の各辺に4ピクセルのパディングを追加し、ランダムな位置から32*32のサイズで切り抜く
    transforms.RandomCrop(32, padding=4),
    #PIL形式の画像をpytorchのテンソルに変換
    transforms.PILToTensor(),
    #画像のデータ型をtorch.floatに変換
    transforms.ConvertImageDtype(torch.float)
])

#CIFAR-10データセットのトレーニングデータを読み込み、データ拡張を適用
train_dset = datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=train_transform
)
#画像データをPILからpytorchのテンソルに変換する処理のみ行う
test_dset = datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
)


#データローダの作成
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)


#残差接続用
class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma*self.residual(x)


#グローバル平均プーリング用
class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=-2)


#(3,32,32)の画像を2*2のパッチに分割
class ToPatches(nn.Module):
    def __init__(self, in_channels, dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels*patch_size**2
        self.proj = nn.Linear(patch_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        #F.unfoldはkernel_sizeとstrideに基づいて分割し、(bs, 3, 32, 32)→(bs,12,256), movedim(a,b)でaの次元とbの次元を交換する
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).movedim(1, -1)  #(bs, 3, 32, 32) → (bs, 256, 12)

        x = self.proj(x)
        x = self.norm(x)
        return x


#位置情報の追加
class AddPositionEmbedding(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.Tensor(num_patches, dim))

    def forward(self, x):
        return x + self.pos_embedding


#ToPatchesクラス、AddPositionEmbeddingクラスを順次実行
class ToEmbedding(nn.Sequential):
    def __init__(self, in_channels, dim, patch_size, num_patches, p_drop=0.):
        super().__init__(ToPatches(in_channels, dim, patch_size), AddPositionEmbedding(dim, num_patches), nn.Dropout(p_drop))


#シフトウィンドウ(SW-MSA)
class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, head_dim, shape, window_size, shift_size=0):
        super().__init__()
        self.heads = dim // head_dim    #dim=各パッチの特徴量, head_dim=各headの特徴量, heads=headの数
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.shape = shape
        self.window_size = window_size    #4*4
        self.shift_size = shift_size      #右に2ピクセル、下に2ピクセル

        self.to_qkv = nn.Linear(dim, dim*3)  #q,k,v

        self.unifyheads = nn.Linear(dim, dim)
        self.pos_enc = nn.Parameter(torch.Tensor(self.heads, (2*window_size - 1)**2))
        self.register_buffer("relative_indices",self.get_indices(window_size))

        if shift_size > 0:
            self.register_buffer("mask", self.generate_mask(shape, window_size, shift_size))

    def forward(self, x):
        shift_size, window_size = self.shift_size, self.window_size
        x = self.to_window(x, self.shape, window_size, shift_size)
        qkv = self.to_qkv(x).unflatten(-1, (3, self.heads, self.head_dim)).transpose(-2, 1)
        queries, keys, values = qkv.unbind(dim=2)
        att = queries @ keys.transpose(-2, -1)
        att = att * self.scale + self.get_rel_pos_enc(window_size)
        if shift_size > 0:
            att = self.mask_attention(att)

        att = F.softmax(att, dim=-1)
        x = att @ values
        x = x.transpose(1, 2).contiguous().flatten(-2, -1)

        x = self.unifyheads(x)
        x = self.from_windows(x, self.shape, window_size, shift_size)
        return x
    
    def to_window(self, x, shape, window_size, shift_size):
        x = x.unflatten(1, shape)
        if shift_size > 0:
            x = x.roll((-shift_size, -shift_size), dims=(1, 2))
        x = self.split_windows(x, window_size)
        return x
    
    def get_rel_pos_enc(self, window_size):
        indices = self.relative_indice.expand(self.heads, -1)
        rel_pos_enc = self.pos_enc.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (window_size**2, window_size**2))

        return rel_pos_enc
    
    def mask_attention(self, att):
        num_win = self.mask.size(1)
        att = att.unflatten(0, (att.size(0) // num_win, num_win))
        att = att.masked_fill(self.mask, float("inf"))
        att = att.flatten(0, 1)
        return att
    
    def from_windows(self, x, shape, window_size, shift_size):
        x = self.merge_windows(x, shape, window_size)
        if shift_size > 0:
            x = x.roll((shift_size, shift_size), dims=(1, 2))
        x = x.flatten(1, 2)
        return x
    
    @staticmethod
    def get_indices(window_size):
        x = torch.arange(window_size, dtype=torch.long)     #1階テンソル[0,1,2,3]を作成
        y1, x1, y2, x2 = torch.meshgrid(x, x, x, x, indexing="ij")
        indices = ((y1 - y2 + window_size -1)*(2*window_size - 1)+(x1 - x2 + window_size - 1))
        indices = indices.flatten()
        return indices

    @staticmethod
    def generate_mask(shape, window_size, shift_size):
        region_mask = torch.zeros(1, *shape, 1)
        slices = [
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None)
        ]

        region_num = 0
        for i in slices:
            for j in slices:
                region_mask[:, i, j, :] = region_num
                region_num += 1

        mask_windows = ShiftedWindowAttention.split_windows(region_mask, window_size).squeeze(-1)
        diff_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        mask = diff_mask != 0
        mask = mask.unsqueeze(1).unsqueeze(0)
        return mask
    
    @staticmethod
    def split_windows(x, window_size):
        n_h, n_w = x.size(1) // window_size, x.size(2) // window_size

        x = x.unflatten(1, (n_h, window_size)).unflatten(-2, (n_w, window_size))
        x = x.transpose(2, 3).flatten(0, 2)
        x = x.flatten(-3, -2)
        return x

    @staticmethod
    def merge_windows(x, shape, window_size):
        n_h, n_w = shape[0] // window_size, shape[1] //window_size
        bs = x.size(0) // (n_h*n_w)
        x = x.unflatten(1, (window_size, window_size))
        x = x.unflatten(0, (bs, n_h, n_w)).transpose(2, 3)
        x = x.flatten(1, 2).flatten(-3, -2)
        return x


class FeedForward(nn.Sequential):
    def __init__(self, dim, mult=4):
        hidden_dim = dim*mult
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

class TransformerBlock(nn.Sequential):
    def __init__(self, dim, head_dim, shape, window_size, shift_size=0, p_drop=0.):
        super().__init__(
            Residual(
                nn.LayerNorm(dim),
                ShiftedWindowAttention(dim, head_dim, shape, window_size, shift_size),
                nn.Dropout(p_drop)
            ),
            Residual(
                nn.LayerNorm(dim),
                FeedForward(dim),
                nn.Dropout(p_drop)
            )
        )

class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, shape):
        super().__init()
        self.shape = shape
        self.norm = nn.LayerNorm(4*in_dim)
        self.reduction = nn.Linear(4*in_dim, out_dim, bias=False)
    
    def forward(self, x):
        x = x.unflatten(1, self.shape).movedim(-1, 1)
        x = F.unfold(x, kernel_size=2, stride=2).movedim(-1, 1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Stage(nn.Sequential):
    def __init__(self, num_blocks, in_dim, out_dim, head_dim, shape, window_size, p_drop=0.):
        if out_dim != in_dim:
            layers = [PatchMerging(in_dim, out_dim, shape)]
            shape = (shape[0] //2, shape[1] //2)
        else:
            layers = []
        
        shift_size = window_size // 2
        layers += [
            TransformerBlock(
                out_dim,
                head_dim,
                shape,
                window_size,
                0 if (num % 2 == 0) else shift_size,
                p_drop
            )
            for num in range(num_blocks)
        ]

        super().__init__(*layers)

class StageStack(nn.Sequential):
    def __init__(self, num_blocks_list, dims, head_dim, shape, window_size, p_drop=0.):
        layers = []
        in_dim = dims[0]
        for num, out_dim in zip(num_blocks_list, dims[1:]):
            layers.append(
                Stage(num, in_dim, out_dim, head_dim, shape, window_size, p_drop)
            )
            
            if in_dim != out_dim:
                shape = (shape[0] // 2, shape[1] // 2)
                in_dim = out_dim
        
        super().__init__(*layers)

class Head(nn.Sequential):
    def __init__(self, dim, classes, p_drop=0.):
        super().__init__(nn.LayerNorm(dim), nn.GELU(), GlobalAvgPool(), nn.Dropout(p_drop), nn.Linear(dim, classes))

class SwinTransformer(nn.Sequential):
    def __init__(
            self, classes, image_size, num_blocks_list, dims, head_dim, patch_size,
            window_size, in_channels=3, emb_p_drop=0., trans_p_drop=0., head_p_drop=0.    ):
        
        reduce_size = image_size // patch_size
        shape = (reduce_size, reduce_size)
        num_patches = shape[0]*shape[1]

        super().__init__(
            ToEmbedding(in_channels, dims[0], patch_size, num_patches, emb_p_drop),
            StageStack(num_blocks_list, dims, head_dim, shape, window_size, trans_p_drop),
            Head(dims[-1], classes, head_p_drop)
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
            elif isinstance(m, AddPositionEmbedding):
                nn.init.normal_(m.pos_embedding, mean=0.0, std=0.02)
            elif isinstance(m, ShiftedWindowAttention):
                nn.init.normal_(m.pos_enc, mean=0.0, std=0.02)
            elif isinstance(m, Residual):
                nn.init.zeros_(m.gamma)
    
    def separate_parameters(self):
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear,)
        modules_no_weight_decay = (nn.LayerNorm,)

        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = (f"{m_name}.{param_name}" if m_name else param_name)

                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, Residual) and param_name.endswith("gamma"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, AddPositionEmbedding) and param_name.endswith("pos_embedding"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, ShiftedWindowAttention) and param_name.endswith("pos_enc"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)
         
        assert len(parameters_decay & parameters_no_decay) == 0
        assert len(parameters_decay) + len(parameters_no_decay) == len(list(model.parameters()))

        return parameters_decay, parameters_no_decay





import torch
from torchsummary import summary

model = SwinTransformer(
    NUM_CLASSES,
    IMAGE_SIZE,
    num_blocks_list=[4,4],
    dims=[128, 128, 256],
    head_dim=32,
    patch_size=2,
    window_size=4,
    emb_p_drop=0.,
    trans_p_drop=0.,
    head_p_drop=0.3
)
model.to(DEVICE)
summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))

def get_optimizer(model, learninig_rate, weight_decay):

    param_dict = {pn: p for pn, p in model.named_parameters()}
    parameters_decay, parameters_no_decay = model.separate_parameters()

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in parameters_decay],
            "weight_decay": weight_decay
        },
        {
            "params": [param_dict[pn] for pn in parameters_no_decay],
            "weight_decay": 0.0            
        },
    ]

    optimizer = optim.AdamW(optim_groups, lr=learninig_rate)

    return optimizer


loss = nn.CrossEntropyLoss()
optimizer = get_optimizer(model, learninig_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
trainer = create_supervised_trainer(model, optimizer, loss, device=DEVICE)
lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)

trainer.add_event_handler(
    Events.ITERATION_COMPLETED,
    lambda engine: lr_scheduler.step()
)

ignite.metrics.RunningAverage(
    output_transform=lambda x: x
).attach(trainer, "loss")

val_metrics = {
    "accuracy": ignite.metrics.Accuracy(),
    "loss": ignite.metrics.Loss(loss)
}

train_evaluator = create_supervised_evaluator(
    model,
    metrics=val_metrics,
    device=DEVICE
)

evaluator = create_supervised_evaluator(
    model,
    metrics=val_metrics,
    device=DEVICE
)

history = defaultdict(list)

# %%time
trainer.run(train_loader, max_epochs=EPOCHS);


fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.arrange(1, len(history["train loss"]) + 1)
ax.plot(xs, history["train loss"], "._", label="train")
ax.plot(xs, history["val loss"], "._", lavel="val")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
ax.grid()
plt.show()

# グラフ描画用のFigureオブジェクトを作成
fig = plt.figure()
# Figureにサブプロット(1行1列の1つ目のプロット)を追加
ax = fig.add_subplot(111)
# x軸のデータをエポック数に基づいて作成（1からhistory['val acc']の長さまでの範囲）
xs = np.arange(1, len(history['val acc']) + 1)
# バリデーションデータの正解率をプロット
ax.plot(xs, history['val acc'], label='Validation Accuracy', linestyle='-')
# トレーニングデータの正解率をプロット
ax.plot(xs, history['train acc'], label='Training Accuracy', linestyle='--')

ax.set_xlabel('Epoch') # x軸のラベルを設定
ax.set_ylabel('Accuracy') # y軸のラベルを設定
ax.grid() # グリッドを表示
ax.legend()  # 凡例を追加
plt.show() # グラフを表示
