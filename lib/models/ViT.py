import torch
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
# from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np


class VisionTransformer(nn.Module):

    def __init__(self, num_classes: int, img_size: int, patch_size: int, num_inputlayer_units: int, num_heads: int, num_mlp_units: int, num_layers: int):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        num_patches = (img_size // patch_size) ** 2     #全パッチ数
        input_dim = 3 * patch_size ** 2                 #１パッチあたりの特徴量 

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
        self.output_layer = nn.Linear(num_inputlayer_units, num_classes)

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
        x = x[:, 0]   #クラストークンを抽出


        x = self.normalize(x)
        x = self.output_layer(x)
        return x
    
    #最終の全結合層を処理中のデバイスを返す関数
    def get_device(self):
        return self.output_layer.weight.device    

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
    
def evaluate(data_loader, model, loss_func):

    model.eval()    #評価モード
    losses =[]
    correct_preds = 0
    total_samples = 0

    for x, y in data_loader:     #データローダからバッチ単位でデータを抽出
        with torch.no_grad():    #勾配の計算をオフにして計算リソースを節約
            x = x.to(device=model.get_device())
            y = y.to(device=model.get_device())

            preds = model(x)   #モデルによる予測
            loss = loss_func(preds, y)    #予測結果predsとgtを使用してloss計算

            losses.append(loss.item())

            _, predicted = torch.max(preds, 1)   #各行ごとに最大値と各インデックスを取得(1と指定すると)
            correct_preds += (predicted == y).sum().item()
            total_samples += y.size(0)

    average_loss = sum(losses) / len(losses)
    accuracy = correct_preds / total_samples

    return average_loss, accuracy

class ModelConfig:
    def __init__(self):
        self.num_epochs = 50
        self.batch_size = 32
        self.lr = 0.01
        self.img_size = 32
        self.patch_size = 16
        self.num_inputlayer_units = 512
        self.num_heads = 4
        self.num_mlp_units = 512
        self.num_layers = 6
        self.batch_size = 32

config = ModelConfig()
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

train_transform = transforms.Compose([transforms.ToTensor(), normalize])    #transforms.Composeは変換するものをまとめてくれる
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

train_dataset = torchvision.datasets.CIFAR10()
test_dataset = torchvision.datasets.CIFAR10()

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

loss_func = F.cross_entropy
model = VisionTransformer(len(train_dataset.classes), config.img_size, config.patch_size, config.num_inputlayer_units, config.num_heads, config.num_mlp_units, config.num_layers)

optimizer = optim.SGD(model.parameters(), lr=config.lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def train_eval():
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = loss_func(preds, y)
            accuracy = (preds.argmax(dim=1) == y).float().mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy.item()

    avg_train_loss = total_loss / len(train_loader) 
    avg_train_accuracy = total_accuracy / len(train_loader)

    val_loss, val_accuracy = evaluate(test_loader, model, loss_func)

    print(f"Epoch{epoch+1}/{config.num_epochs}") 
    print(f"  Training: loss = {avg_train_loss:.3f}, accuracy = {avg_train_accuracy:.3f}") 
    print(f"  Validation: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")  

    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config.num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, config.num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.show