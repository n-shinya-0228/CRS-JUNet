"""D.5: one C.2 parameter set, detached eval pass followed by a trainable pass."""

import torch

from .SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    ALPHA_TABLE, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps,
)


class SJNet3_PointNet_Gated_Local3_SharedSoftAlpha(SJNet3_PointNet_Gated_Local3_ClassAlpha):
    def forward_with_alpha(self, x, alpha=1.2, alpha_map=None):
        if alpha != 1.2:
            raise ValueError("D.5 Stage 1 uses fixed global alpha=1.2")
        return super().forward(x, alpha_map=alpha_map)

    @torch.no_grad()
    def stage1(self, x):
        if x.ndim != 4 or x.shape[1] != 24 or x.dtype != torch.float32:
            raise ValueError("Expected float32 [B,24,H,W]: hand 7, PointNet 16, mask 1")
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            # C.2 Stage 2 uses BF16; alpha generation retains D.2's FP32 precision.
            with torch.autocast(device_type=x.device.type, enabled=False):
                outputs = self.forward_with_alpha(x)
                logits = outputs.pop("logits")
                del outputs
                _, soft = build_alpha_maps(logits, x[:, 23:24])
            return logits.detach(), soft.detach()
        finally:
            for module, training in modes:
                module.training = training

    def forward(self, x):
        logits, soft = self.stage1(x)
        del logits
        return self.forward_with_alpha(x, alpha_map=soft)
