"""D.4: a fixed C.2 teacher conditions a separate, trainable C.2 student."""

import torch
from torch import nn

from .SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    ALPHA_TABLE, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps,
)
from .SJNet3_PointNet_Gated_Local3_FixedAlpha import SJNet3_PointNet_Gated_Local3_FixedAlpha


class SJNet3_PointNet_Gated_Local3_TeacherStudent(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.teacher = SJNet3_PointNet_Gated_Local3_FixedAlpha(**options, alpha=1.2)
        self.student = SJNet3_PointNet_Gated_Local3_ClassAlpha(**options)
        self.teacher.requires_grad_(False).eval()
        self.fixed_table = ALPHA_TABLE

    def initialize(self, state):
        self.teacher.load_state_dict(state, strict=True)
        self.student.load_state_dict(state, strict=True)
        self.teacher.requires_grad_(False).eval()
        self.student.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def alpha_maps(self, x):
        if x.ndim != 4 or x.shape[1] != 24 or x.dtype != torch.float32:
            raise ValueError("Teacher/Student require the same float32 [B,24,H,W] input")
        if self.fixed_table != ALPHA_TABLE:
            raise RuntimeError("The fixed D.2 table changed")
        self.teacher.eval()
        self.teacher.set_alpha(1.2)
        outputs = self.teacher(x)
        hard, soft = build_alpha_maps(outputs["logits"], x[:, 23:24])
        del outputs
        # no_grad, not inference_mode: autograd can safely save this constant.
        return hard.detach(), soft.detach()

    def forward(self, x, student=None):
        _, soft = self.alpha_maps(x)
        target = self.student if student is None else student
        return target(x, alpha_map=soft)
