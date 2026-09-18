import math
import torch
import torch.nn as nn


def build_polar_point_features(
    points,
    H=512,
    W=512,
    max_radius=51.2
):
    """
    points: [N,4]
        x, y, z, remission

    return
    -------
    point_features : [Nv,7]
    unique_cells   : [K]
    inverse_indices: [Nv]
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    remission = points[:, 3]

    # -------------------------
    # Polar coordinates
    # -------------------------
    rho = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)

    res_rho = max_radius / H
    res_theta = 2.0 * math.pi / W

    valid = (rho > 0.0) & (rho < max_radius)

    if not torch.any(valid):
        return (
            torch.empty(
                (0, 7),
                dtype=points.dtype,
                device=points.device
            ),
            torch.empty(
                (0,),
                dtype=torch.long,
                device=points.device
            ),
            torch.empty(
                (0,),
                dtype=torch.long,
                device=points.device
            )
        )

    x = x[valid]
    y = y[valid]
    z = z[valid]
    remission = remission[valid]
    rho = rho[valid]
    theta = theta[valid]

    # -------------------------
    # BEV cell
    # -------------------------
    grid_y = torch.floor(
        rho / res_rho
    ).long()

    grid_x = torch.floor(
        (theta + math.pi) / res_theta
    ).long()

    grid_y = torch.clamp(grid_y, 0, H - 1)
    grid_x = torch.clamp(grid_x, 0, W - 1)

    cell_id = grid_y * W + grid_x

    # occupied cellだけに番号を振り直す
    unique_cells, inverse_indices = torch.unique(
        cell_id,
        sorted=True,
        return_inverse=True
    )

    num_cells = unique_cells.numel()

    # -------------------------
    # cell mean z
    # -------------------------
    z_sum = torch.zeros(
        num_cells,
        device=z.device,
        dtype=z.dtype
    )

    count = torch.zeros(
        num_cells,
        device=z.device,
        dtype=z.dtype
    )

    z_sum.scatter_add_(
        0,
        inverse_indices,
        z
    )

    count.scatter_add_(
        0,
        inverse_indices,
        torch.ones_like(z)
    )

    mean_z = z_sum / count.clamp_min(1.0)

    dz = z - mean_z[inverse_indices]

    # -------------------------
    # cell中心
    # -------------------------
    rho_center = (
        grid_y.float() + 0.5
    ) * res_rho

    theta_center = (
        -math.pi
        + (grid_x.float() + 0.5) * res_theta
    )

    drho = rho - rho_center
    dtheta = theta - theta_center

    # -------------------------
    # normalization
    # -------------------------
    x_norm = x / max_radius
    y_norm = y / max_radius

    # 現在のBEV z正規化に近づける
    z_norm = torch.clamp(
        z,
        -5.0,
        15.0
    )

    z_norm = (z_norm - 1.0) / 5.0

    remission_norm = torch.clamp(
        remission,
        0.0,
        1.0
    )

    # cell内ではだいたい -0.5 ～ +0.5
    drho_norm = drho / res_rho
    dtheta_norm = dtheta / res_theta

    dz_norm = torch.clamp(
        dz,
        -5.0,
        5.0
    ) / 5.0

    # [Nv,7]
    point_features = torch.stack(
        [
            x_norm,
            y_norm,
            z_norm,
            remission_norm,
            drho_norm,
            dtheta_norm,
            dz_norm,
        ],
        dim=1
    )

    return (
        point_features,
        unique_cells,
        inverse_indices
    )

class PolarPointEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        out_channels=16
    ):
        super().__init__()

        self.out_channels = out_channels

        # 各点に共通して適用するMLP
        self.point_mlp = nn.Sequential(

            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # cellごとに集約した後
        self.cell_mlp = nn.Sequential(

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, out_channels)
        )

    def forward(
        self,
        point_features,
        inverse_indices
    ):
        if point_features.shape[0] == 0:
            return torch.empty(
                (0, self.out_channels),
                dtype=point_features.dtype,
                device=point_features.device
                )

        # [N,7]
        point_features = self.point_mlp(
            point_features
        )

        # [N,64]

        num_cells = (
            int(inverse_indices.max().item())
            + 1
        )

        cell_features = torch.full(
            (
                num_cells,
                point_features.shape[1]
            ),
            -1e9,
            dtype=point_features.dtype,
            device=point_features.device
        )

        index = inverse_indices.unsqueeze(1)
        index = index.expand_as(
            point_features
        )

        # PointNetのmax pooling
        cell_features.scatter_reduce_(
            0,
            index,
            point_features,
            reduce="amax",
            include_self=True
        )

        # [K,64]
        cell_features = self.cell_mlp(
            cell_features
        )

        # [K,16]
        return cell_features

class PolarPointPretrainModel(nn.Module):
    """
    Step Aの事前学習用モデル

    Point Encoder:
        [Nv, 7]
          ↓
        [K, 16]

    Classifier:
        [K, 16]
          ↓
        [K, num_classes]
    """

    def __init__(
        self,
        num_classes=20,
        in_channels=7,
        feature_dim=16
    ):
        super().__init__()

        self.encoder = PolarPointEncoder(
            in_channels=in_channels,
            out_channels=feature_dim
        )

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

    def forward(
        self,
        point_features,
        inverse_indices
    ):
        cell_features = self.encoder(
            point_features,
            inverse_indices
        )

        logits = self.classifier(
            cell_features
        )

        return {
            "logits": logits,
            "features": cell_features
        }