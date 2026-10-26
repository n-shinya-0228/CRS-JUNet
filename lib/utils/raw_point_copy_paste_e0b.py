"""Isolated E.0b raw-point augmentation. Never imported by production training."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from lib.utils.laserscan_Polar3 import SemLaserScan
from lib.utils.laserscan_Polar4 import PRIORITY_MAP
from lib.utils.laserscan_Pointnet import build_polar_point_features
from lib.utils.pointnet_copy_paste_e0 import normalize_handcrafted, sha256


# Conservative audit gates chosen after measuring the existing DB distributions.
# Dimensions use XY PCA axes, not orientation-dependent world-axis bounding boxes.
QUALITY_RULES = {
    2: dict(min_points=30, length=(.6, 3.), width=(.15, 1.5), height=(.45, 1.6)),
    3: dict(min_points=40, length=(.7, 3.3), width=(.2, 1.6), height=(.5, 1.8)),
    4: dict(min_points=150, length=(2.5, 12.5), width=(.8, 3.5), height=(1.2, 3.6)),
    5: dict(min_points=100, length=(1., 20.), width=(.4, 3.5), height=(.7, 3.5)),
    6: dict(min_points=30, length=(.2, 1.2), width=(.1, .9), height=(.8, 2.1)),
    7: dict(min_points=40, length=(.3, 1.5), width=(.15, 1.1), height=(.9, 2.1)),
    8: dict(min_points=40, length=(.4, 1.8), width=(.2, 1.3), height=(1., 2.1)),
}


def inspect_patch(array, filename, learning_map):
    match = re.fullmatch(r"class_(\d+)_(\d+)\.npy", Path(filename).name)
    row = {"filename_raw_id": int(match[1]) if match else -1,
           "dtype": str(array.dtype), "shape": list(array.shape), "reasons": []}
    reasons = row["reasons"]
    if not match:
        reasons.append("invalid_filename")
    if array.dtype != np.float32:
        reasons.append("dtype_not_float32")
    if array.ndim != 2 or array.shape[1] != 5:
        reasons.append("shape_not_N5")
        return row
    row["points"] = len(array)
    if not len(array):
        reasons.append("empty_patch")
        return row
    if not np.isfinite(array).all():
        reasons.append("nonfinite_values")
        return row
    xyz = array[:, :3]
    raw = np.unique(array[:, 4])
    row["raw_ids"] = raw.tolist()
    if not np.equal(raw, np.floor(raw)).all() or any(int(v) not in learning_map for v in raw):
        reasons.append("invalid_raw_semantic_id")
        return row
    if len(raw) != 1:
        reasons.append("mixed_raw_semantic_ids")
    if not np.all(array[:, 4] == row["filename_raw_id"]):
        reasons.append("filename_content_mismatch")
    cls = learning_map.get(row["filename_raw_id"], -1)
    row["learning_id"] = cls
    mean = xyz.mean(0, dtype=np.float64)
    lo, hi = xyz.min(0), xyz.max(0)
    centered = xyz[:, :2].astype(np.float64) - mean[:2]
    _, axes = np.linalg.eigh(centered.T @ centered / len(xyz))
    width, length = sorted(np.ptp(centered @ axes, axis=0))
    height = float(hi[2] - lo[2])
    row.update(xyz_mean=mean.tolist(), xyz_min=lo.tolist(), xyz_max=hi.tolist(),
               xyz_extent=(hi - lo).tolist(), length=float(length), width=float(width), height=height,
               remission_min=float(array[:, 3].min()), remission_max=float(array[:, 3].max()),
               duplicate_points=len(array) - len(np.unique(array, axis=0)),
               max_center_distance=float(np.linalg.norm(xyz, axis=1).max()))
    if np.abs(mean).max() > .001:
        reasons.append("relative_center_not_zero")
    if array[:, 3].min() < 0 or array[:, 3].max() > 1:
        reasons.append("remission_outside_0_1")
    rule = QUALITY_RULES.get(cls)
    if rule is None:
        reasons.append("unsupported_learning_class")
    else:
        if len(array) < rule["min_points"]:
            reasons.append("insufficient_class_specific_points")
        for name in ("length", "width", "height"):
            if row[name] < rule[name][0]:
                reasons.append("incomplete_" + name)
            if row[name] > rule[name][1]:
                reasons.append("oversized_or_merged_" + name)
    if row["duplicate_points"] > .05 * len(array):
        reasons.append("duplicate_point_fraction_gt_5pct")
    # A rider without the associated vehicle cannot be safely grounded as a person.
    if cls in (7, 8):
        reasons.append("rider_vehicle_alignment_metadata_missing")
    return row


def load_patch(path, row, learning_map, *, audit_unverified_provenance=False):
    if not row.get("provenance_verified", False) and not audit_unverified_provenance:
        raise ValueError("donor_provenance_unverified")
    if not row.get("usable_for_audit", False):
        raise ValueError("donor_not_quality_approved")
    if row.get("sha256") and sha256(path) != row["sha256"]:
        raise ValueError("donor_changed_since_audit")
    array = np.load(path, allow_pickle=False)
    check = inspect_patch(array, path, learning_map)
    if check["reasons"]:
        raise ValueError("invalid_donor:" + ";".join(check["reasons"]))
    return array


def read_train_scan(path, learning_map, train_sequences):
    path = Path(path)
    sequence = path.parent.parent.name
    allowed = {f"{int(s):02d}" for s in train_sequences}
    if sequence not in allowed or sequence == "08":
        raise ValueError("target_not_train_split")
    # Use the same official project loader as build_database; Polar3 does not sort.
    state = np.random.get_state()
    scan = SemLaserScan(project=False)
    np.random.set_state(state)
    scan.open_scan(str(path))
    scan.open_label(str(path.parent.parent / "labels" / (path.stem + ".label")))
    points = np.column_stack([scan.points, scan.remissions.reshape(-1)]).astype(np.float32)
    packed = (scan.sem_label.astype(np.uint32) | (scan.inst_label.astype(np.uint32) << 16)).reshape(-1)
    if len(points) != len(packed) or not np.isfinite(points).all():
        raise ValueError("invalid_target_points_or_labels")
    map_semantic(packed, learning_map)
    return points, packed


def map_semantic(packed, learning_map):
    raw = np.asarray(packed).astype(np.int64) & 0xFFFF
    if any(int(v) not in learning_map for v in np.unique(raw)):
        raise ValueError("invalid_target_raw_semantic_id")
    lut = np.zeros(max(learning_map) + 1, dtype=np.int64)
    for key, value in learning_map.items():
        lut[key] = value
    return lut[raw]


@dataclass(frozen=True)
class PlacementRules:
    min_center_range: float = 3.
    max_radius: float = 51.2
    ground_clearance: float = .03
    ground_min_points: int = 8
    ground_max_rmse: float = .08
    ground_max_slope: float = .15
    collision_margin_xy: float = .15
    collision_margin_z: float = .05
    azimuth_bins: int = 2048
    elevation_bins: int = 64
    fov_down_degrees: float = -23.
    fov_up_degrees: float = 3.
    depth_tolerance: float = .15
    min_visible_fraction: float = .8


class PlacementRejected(ValueError):
    pass


def rigid_place(relative_xyz, yaw, translation):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return (relative_xyz.astype(np.float64) @ rotation.T + translation).astype(np.float32), rotation


def ray_coordinates(points, rules):
    xyz = np.asarray(points)[:, :3].astype(np.float64)
    rho = np.hypot(xyz[:, 0], xyz[:, 1])
    distance = np.linalg.norm(xyz, axis=1)
    angle = np.arctan2(xyz[:, 1], xyz[:, 0])
    elevation = np.arctan2(xyz[:, 2], rho)
    down, up = np.radians([rules.fov_down_degrees, rules.fov_up_degrees])
    valid = (distance > 0) & (elevation >= down) & (elevation <= up)
    col = np.floor((angle + np.pi) / (2 * np.pi) * rules.azimuth_bins).astype(np.int64) % rules.azimuth_bins
    row = np.clip(np.floor((elevation - down) / (up - down) * rules.elevation_bins).astype(np.int64),
                  0, rules.elevation_bins - 1)
    return row * rules.azimuth_bins + col, distance, valid


def visibility_filter(target, donor, rules):
    """Finite angular-bin depth buffer, NOT a calibrated LiDAR simulator."""
    ti, tr, tv = ray_coordinates(target, rules)
    di, dr, dv = ray_coordinates(donor, rules)
    size = rules.azimuth_bins * rules.elevation_bins
    nearest_target = np.full(size, np.inf)
    nearest_donor = np.full(size, np.inf)
    np.minimum.at(nearest_target, ti[tv], tr[tv])
    np.minimum.at(nearest_donor, di[dv], dr[dv])
    donor_visible = dv & (dr <= nearest_target[di] + rules.depth_tolerance)
    donor_visible &= dr <= nearest_donor[di] + rules.depth_tolerance
    # Remove only points behind actually retained donor returns in the same ray bin.
    nearest_retained = np.full(size, np.inf)
    np.minimum.at(nearest_retained, di[donor_visible], dr[donor_visible])
    target_keep = ~(tv & (tr > nearest_retained[ti] + rules.depth_tolerance))
    return target_keep, donor_visible


class TargetScene:
    def __init__(self, points, packed_labels, learning_map):
        self.points = np.asarray(points)
        self.labels = np.asarray(packed_labels)
        if self.points.ndim != 2 or self.points.shape[1] != 4 or len(points) != len(packed_labels):
            raise ValueError("target_shape_or_label_count")
        if not np.isfinite(points).all():
            raise ValueError("nonfinite_target")
        self.learning = map_semantic(self.labels, learning_map)
        self.ground = np.isin(self.learning, [9, 10, 11, 12, 17])
        self.tree = cKDTree(self.points[:, :2])

    def sample_ground(self, learning_class, rng):
        classes = [11] if learning_class in (6, 7) else [9, 10]
        rho = np.hypot(self.points[:, 0], self.points[:, 1])
        indices = np.flatnonzero(np.isin(self.learning, classes) & (rho > 4) & (rho < 40))
        if not len(indices):
            raise PlacementRejected("no_allowed_observed_ground")
        return self.points[int(rng.choice(indices)), :2].copy()


def raw_copy_paste(scene, patch, learning_map, center_xy, yaw, rules=PlacementRules()):
    start = time.perf_counter()
    if patch.ndim != 2 or patch.shape[1] != 5 or not len(patch) or not np.isfinite(patch).all():
        raise PlacementRejected("invalid_patch_shape_or_values")
    check = inspect_patch(patch, f"class_{int(patch[0,4])}_000000.npy" if len(patch) else "invalid.npy", learning_map)
    if check["reasons"]:
        raise PlacementRejected("invalid_patch:" + ";".join(check["reasons"]))
    cls = check["learning_id"]
    center_xy = np.asarray(center_xy, dtype=np.float64)
    if center_xy.shape != (2,) or not np.isfinite(center_xy).all() or not np.isfinite(yaw):
        raise PlacementRejected("invalid_transform")
    if np.linalg.norm(center_xy) < rules.min_center_range:
        raise PlacementRejected("center_too_close")
    radius = float(np.linalg.norm(patch[:, :2], axis=1).max()) + .4
    nearby = np.asarray(scene.tree.query_ball_point(center_xy, radius), dtype=np.int64)
    ground_indices = nearby[scene.ground[nearby]]
    if len(ground_indices) < rules.ground_min_points:
        raise PlacementRejected("insufficient_ground_support")
    ground = scene.points[ground_indices, :3].astype(np.float64)
    design = np.column_stack([ground[:, :2] - center_xy, np.ones(len(ground))])
    fit = np.linalg.lstsq(design, ground[:, 2], rcond=None)[0]
    for _ in range(2):
        residual = ground[:, 2] - design @ fit
        keep = np.abs(residual) <= .15
        if keep.sum() < rules.ground_min_points:
            raise PlacementRejected("ground_outliers_or_discontinuity")
        fit = np.linalg.lstsq(design[keep], ground[keep, 2], rcond=None)[0]
    rmse = float(np.sqrt(np.mean((ground[keep, 2] - design[keep] @ fit) ** 2)))
    if rmse > rules.ground_max_rmse or np.linalg.norm(fit[:2]) > rules.ground_max_slope:
        raise PlacementRejected("ground_rough_or_steep")
    z = fit[2] + rules.ground_clearance - float(patch[:, 2].min())
    translation = np.r_[center_xy, z]
    moved, rotation = rigid_place(patch[:, :3], yaw, translation)
    rho = np.hypot(moved[:, 0], moved[:, 1])
    if (rho <= 0).any() or (rho >= rules.max_radius).any():
        raise PlacementRejected("donor_outside_polar_radius")
    floor = fit[2] + (moved[:, :2] - center_xy) @ fit[:2]
    bottom = moved[:, 2] <= np.quantile(moved[:, 2], .1)
    clearance = moved[:, 2] - floor
    if clearance.min() < -.08 or np.median(clearance[bottom]) > .25:
        raise PlacementRejected("buried_or_floating_bottom")
    nearest_ground = cKDTree(ground[keep, :2]).query(moved[bottom, :2])[0]
    if np.quantile(nearest_ground, .9) > 1.:
        raise PlacementRejected("bottom_ground_support_too_sparse")
    placement_seconds = time.perf_counter() - start
    collision_start = time.perf_counter()
    non_ground = nearby[~scene.ground[nearby]]
    local = (scene.points[non_ground, :3] - translation) @ rotation
    lo, hi = patch[:, :3].min(0), patch[:, :3].max(0)
    margin = np.array([rules.collision_margin_xy, rules.collision_margin_xy, rules.collision_margin_z])
    collision = ((local >= lo - margin) & (local <= hi + margin)).all(1)
    if collision.any():
        raise PlacementRejected("non_ground_obb_collision")
    donor = np.column_stack([moved, patch[:, 3]]).astype(np.float32)
    target_keep, donor_visible = visibility_filter(scene.points, donor, rules)
    fraction = float(donor_visible.mean())
    if fraction < rules.min_visible_fraction or donor_visible.sum() < QUALITY_RULES[cls]["min_points"]:
        raise PlacementRejected("insufficient_visible_donor_points")
    after = np.concatenate([scene.points[target_keep], donor[donor_visible]])
    labels = np.concatenate([scene.labels[target_keep], patch[donor_visible, 4].astype(np.uint32)])
    paste_points = np.r_[np.zeros(int(target_keep.sum()), dtype=bool), np.ones(int(donor_visible.sum()), dtype=bool)]
    collision_seconds = time.perf_counter() - collision_start
    metadata = dict(learning_id=cls, raw_id=int(patch[0, 4]), yaw=float(yaw),
                    rotation=rotation.tolist(), translation=translation.tolist(),
                    ground_plane=fit.tolist(), ground_rmse=rmse,
                    minimum_ground_clearance=float(clearance.min()), bottom_median_clearance=float(np.median(clearance[bottom])),
                    visible_fraction=fraction, added_points=int(donor_visible.sum()),
                    removed_target_points=int((~target_keep).sum()), collision_points=0,
                    placement_seconds=placement_seconds, collision_visibility_seconds=collision_seconds,
                    raw_cp_seconds=time.perf_counter() - start, rules=asdict(rules),
                    donor_instance_id=0, donor_source_instance_id="not stored", visibility_model="angular-bin depth buffer approximation")
    return dict(points=after, labels=labels, paste_points=paste_points, donor_transformed=donor,
                donor_visible=donor_visible, target_keep=target_keep, metadata=metadata)


def deterministic_labels(cells, inverse, packed, learning_map):
    raw = np.asarray(packed, dtype=np.int64) & 0xFFFF
    if any(int(v) not in learning_map for v in np.unique(raw)):
        raise ValueError("invalid_raw_semantic_id")
    score = PRIORITY_MAP[raw].astype(np.int64) * 65536 + raw
    maxima = np.full(len(cells), -1, dtype=np.int64)
    np.maximum.at(maxima, inverse, score)
    lut = np.zeros(max(learning_map) + 1, dtype=np.int64)
    for key, value in learning_map.items():
        lut[key] = value
    label = np.zeros(512 * 512, dtype=np.int64)
    label[cells] = lut[maxima % 65536]
    return torch.from_numpy(label.reshape(512, 512))


def regenerate_hybrid(points, packed, encoder, learning_map, paste_points=None):
    """One shared mapping for all 24 channels; no cached feature copying."""
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("encoder_must_be_frozen_eval")
    if len(points) != len(packed):
        raise ValueError("point_label_count_mismatch")
    device = next(encoder.parameters()).device
    if device.type == "cuda" and not torch.are_deterministic_algorithms_enabled():
        raise ValueError("CUDA_pointnet_regeneration_requires_deterministic_algorithms")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
        features, cells, inverse = build_polar_point_features(tensor)
        radius = torch.sqrt(tensor[:, 0] ** 2 + tensor[:, 1] ** 2)
        valid = ((radius > 0) & (radius < 51.2)).cpu().numpy()
        ids, inv = cells.cpu().numpy(), inverse.cpu().numpy()
        sync()
        preprocessing = time.perf_counter() - start
        forward_start = time.perf_counter()
        point_values = encoder(features, inverse).cpu()
        sync()
        forward_seconds = time.perf_counter() - forward_start
    hand_start = time.perf_counter()
    valid_points = points[valid]
    order = np.argsort(inv, kind="stable")
    grouped = valid_points[order]
    _, starts, counts = np.unique(inv[order], return_index=True, return_counts=True)
    dense = np.zeros((7, 512 * 512), dtype=np.float32)
    if len(ids):
        hi = np.maximum.reduceat(grouped, starts, axis=0)
        lo = np.minimum.reduceat(grouped, starts, axis=0)
        dense[:, ids] = np.stack([hi[:, 2], np.add.reduceat(grouped[:, 2], starts) / counts,
            hi[:, 3], counts / 100., hi[:, 2] - lo[:, 2], hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1]])
    raw7 = torch.from_numpy(dense.reshape(7, 512, 512))
    hand_seconds = time.perf_counter() - hand_start
    label_start = time.perf_counter()
    label = deterministic_labels(ids, inv, packed[valid], learning_map)
    mask = torch.zeros(1, 512 * 512)
    mask[:, cells.cpu()] = 1
    mask = mask.reshape(1, 512, 512)
    point = torch.zeros(16, 512 * 512)
    point[:, cells.cpu()] = point_values.T
    point = point.reshape(16, 512, 512)
    paste = torch.zeros(1, 512 * 512)
    if paste_points is not None:
        paste[:, torch.from_numpy(np.unique(ids[inv[np.asarray(paste_points)[valid]]]))] = 1
    paste = paste.reshape(1, 512, 512)
    hybrid = torch.cat([normalize_handcrafted(raw7.half().float()), point.half().float(), mask])
    return dict(raw7=raw7, point16=point, label=label, mask=mask, paste_mask=paste,
                hybrid24=hybrid, cell_ids=cells.cpu(), point_cell_ids=ids[inv], valid_points=valid,
                timing=dict(preprocessing_seconds=preprocessing, pointnet_forward_seconds=forward_seconds,
                    handcrafted7_seconds=hand_seconds, label_mask_assembly_seconds=time.perf_counter() - label_start,
                    total_seconds=time.perf_counter() - start,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0))


def apply_dropout(hybrid, label, paste_mask, keep, *, policy="ignore_dropped"):
    if policy not in ("ignore_dropped", "legacy_denoising"):
        raise ValueError("unknown_dropout_policy")
    keep = keep.bool()
    result, target, paste = hybrid.clone(), label.clone(), paste_mask.clone()
    result *= keep.unsqueeze(0)
    paste *= keep.unsqueeze(0)
    if policy == "ignore_dropped":
        target[~keep] = 0
    return result, target, paste


def compatibility_off(dataset, index):
    """No extra RNG draw, preprocessing or tie-policy change on the legacy OFF path."""
    return dataset[index]
