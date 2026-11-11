import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value):
        value = F.pad(value, (1, 1, 0, 0), mode="circular")
        value = F.pad(value, (0, 0, 1, 1), mode="constant", value=0.0)
        return self.activation(self.norm(self.conv(value)))


class ProjectionEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = CircularConv(in_channels, 16)
        self.down1 = nn.Sequential(CircularConv(16, 24, stride=2), CircularConv(24, 24))
        self.down2 = nn.Sequential(CircularConv(24, 32, stride=2), CircularConv(32, 32))
        self.down3 = nn.Sequential(CircularConv(32, 48, stride=2), CircularConv(48, 48))

    def forward(self, value):
        stem = self.stem(value)
        level1 = self.down1(stem)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        return stem, level1, level2, level3


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = CircularConv(in_channels + skip_channels, out_channels)
        self.conv2 = CircularConv(out_channels, out_channels)

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = torch.cat((value, skip), dim=1)
        return self.conv2(self.conv1(value))


class ProjectionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode2 = DecoderBlock(48, 32, 32)
        self.decode1 = DecoderBlock(32, 24, 24)
        self.decode0 = DecoderBlock(24, 16, 16)
        self.features = nn.Conv2d(16, 128, 1)

    def forward(self, levels):
        stem, level1, level2, level3 = levels
        value = self.decode2(level3, level2)
        value = self.decode1(value, level1)
        return self.features(self.decode0(value, stem))


class SphericalBEVFusion(nn.Module):
    """Independent CNN branches; final aligned 128-channel feature addition."""
    def __init__(self, nclasses=20):
        super().__init__()
        self.bev_encoder = ProjectionEncoder(7)
        self.sphere_encoder = ProjectionEncoder(6)
        self.bev_decoder = ProjectionDecoder()
        self.sphere_decoder = ProjectionDecoder()
        self.head = nn.Conv2d(128, nclasses, 1)
        self.sphere_head = nn.Conv2d(128, nclasses, 1)

    def forward(self, bev, sphere, bev_to_sphere, bev_to_sphere_valid):
        bev_features = self.bev_decoder(self.bev_encoder(bev))
        sphere_features = self.sphere_decoder(self.sphere_encoder(sphere))
        sphere_on_bev = F.grid_sample(
            sphere_features, bev_to_sphere, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ) * bev_to_sphere_valid.unsqueeze(1)
        fused = bev_features + sphere_on_bev
        return {"bev_logits": self.head(fused),
                "sphere_logits": self.sphere_head(sphere_features)}
