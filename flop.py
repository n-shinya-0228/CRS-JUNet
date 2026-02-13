# import torch
# from thop import profile
# from lib.models.ChatNet4 import ChatNet4   # ← あなたの実装

# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ===== モデル生成 =====
#     model = ChatNet4(
#         in_channels=6,
#         nclasses=20,
#         base_ch=48,
#         aspp_out=384,
#         swa_blocks=2,
#         swa_heads=8,
#         swa_window=(8, 16)
#     ).to(device)
#     model.eval()

#     # ===== ダミー入力 =====
#     # SemanticKITTI range view: (B, C, H, W) = (1, 6, 64, 1024)
#     dummy = torch.randn(1, 6, 64, 2048).to(device)

#     # ===== Params =====
#     params = count_params(model)
#     print(f"Params: {params/1e6:.2f} M")

#     # ===== FLOPs =====
#     macs, params_thop = profile(model, inputs=(dummy,), verbose=False)
#     flops = macs * 2   # MACs → FLOPs
#     print(f"FLOPs: {flops/1e9:.2f} G")

import torch
import torch.nn as nn
from thop import profile
from lib.models.ChatNet4 import ChatNet4

class LogitsOnly(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        if isinstance(out, dict):
            return out["logits"]
        return out

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = "cuda" if torch.cuda.is_available() else "cpu"

base = ChatNet4(
    in_channels=6, nclasses=20,
    base_ch=48, aspp_out=384,
    swa_blocks=2, swa_heads=8, swa_window=(8, 16)
).to(device).eval()

model = LogitsOnly(base).to(device).eval()

dummy = torch.randn(1, 6, 64, 2048).to(device)

with torch.no_grad():
    macs, _ = profile(model, inputs=(dummy,), verbose=False)

params = count_params(base)  # ← baseでOK（ラッパーはノーパラ）
flops = macs * 2

print(f"Params: {params/1e6:.2f} M")
print(f"FLOPs : {flops/1e9:.2f} G")
