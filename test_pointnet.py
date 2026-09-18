import numpy as np
import torch

from lib.utils.laserscan_Pointnet import (
    build_polar_point_features,
    PolarPointEncoder
)


device = "cuda"

bin_path = \
    "/home/ns/CRS-JUNet/SemanticKitti/sequences/00/velodyne/000000.bin"

scan = np.fromfile(
    bin_path,
    dtype=np.float32
).reshape(-1, 4)

points = torch.from_numpy(
    scan
).float().to(device)


point_features, unique_cells, inverse_indices = \
    build_polar_point_features(points)


print("================================")
print("Raw points")
print(points.shape)

print("PointNet input")
print(point_features.shape)

print("Occupied cells")
print(unique_cells.shape)

print("Inverse indices")
print(inverse_indices.shape)

print("Cell ID min/max")
print(
    unique_cells.min().item(),
    unique_cells.max().item()
)

print("NaN:")
print(
    torch.isnan(point_features).any().item()
)


encoder = PolarPointEncoder(
    in_channels=7,
    out_channels=16
).to(device)

encoder.eval()

with torch.no_grad():

    cell_features = encoder(
        point_features,
        inverse_indices
    )


print("================================")
print("Cell features")
print(cell_features.shape)

print("Feature NaN")
print(
    torch.isnan(cell_features).any().item()
)


pn_bev = torch.zeros(
    16,
    512 * 512,
    device=device
)

pn_bev[:, unique_cells] = \
    cell_features.T

pn_bev = pn_bev.view(
    16,
    512,
    512
)

print("PointNet BEV")
print(pn_bev.shape)

from lib.utils.laserscan_Polar5 import LaserScan


# ==========================================
# Polar5とのmask比較
# ==========================================

ref_scan = LaserScan(
    project=True,
    H=512,
    W=512
)

ref_scan.open_scan(bin_path)

ref_mask = torch.from_numpy(
    ref_scan.proj_mask
).bool().to(device)


# PointNet側のmask
pn_mask = torch.zeros(
    512 * 512,
    dtype=torch.bool,
    device=device
)

pn_mask[unique_cells] = True

pn_mask = pn_mask.view(
    512,
    512
)


# ==========================================
# 比較
# ==========================================

agreement = (
    pn_mask == ref_mask
).float().mean()

different_cells = torch.logical_xor(
    pn_mask,
    ref_mask
).sum()


print("================================")
print("Mask comparison")

print(
    "PointNet occupied:",
    pn_mask.sum().item()
)

print(
    "Polar5 occupied:",
    ref_mask.sum().item()
)

print(
    "Mask agreement:",
    agreement.item()
)

print(
    "Different cells:",
    different_cells.item()
)