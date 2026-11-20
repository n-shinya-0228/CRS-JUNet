"""Append-only Basic CP: no plane fitting, collision tests, or visibility filtering."""

from collections import Counter
import numpy as np

from lib.utils.raw_point_copy_paste_e0b import PlacementRejected, map_semantic, rigid_place
from lib.utils.raw_point_copy_paste_e0c import CLASSES


def place_basic(points, patch, learning_map, center_xy, yaw):
    points, patch = np.asarray(points), np.asarray(patch)
    if (points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all()
            or patch.ndim != 2 or patch.shape[1] != 5 or not len(patch)
            or not np.isfinite(patch).all()):
        raise PlacementRejected("invalid_points_or_patch")
    raw = patch[:, 4]
    if (not np.equal(raw, np.floor(raw)).all() or len(np.unique(raw)) != 1
            or learning_map.get(int(raw[0])) not in CLASSES):
        raise PlacementRejected("invalid_donor_semantic")
    center_xy = np.asarray(center_xy, dtype=np.float64)
    if center_xy.shape != (2,) or not np.isfinite(center_xy).all() or not np.isfinite(yaw):
        raise PlacementRejected("invalid_transform")
    rho = np.hypot(points[:, 0], points[:, 1])
    support = points[(rho > 0) & (rho < 51.2)]
    if not len(support):
        raise PlacementRejected("empty_in_range_target")
    local = support[np.sum((support[:, :2] - center_xy) ** 2, axis=1) <= 2. ** 2]
    # A scalar low-z estimate only; no semantic ground mask, plane, slope, or support test.
    floor = float(np.quantile((local if len(local) else support)[:, 2], .1))
    translation = np.r_[center_xy, floor + .03 - float(patch[:, 2].min())]
    moved, rotation = rigid_place(patch[:, :3], yaw, translation)
    radius = np.hypot(moved[:, 0], moved[:, 1])
    elevation = np.degrees(np.arctan2(moved[:, 2], radius))
    if not np.isfinite(moved).all() or (radius <= 0).any() or (radius >= 51.2).any():
        raise PlacementRejected("outside_polar_radius")
    # Sensor FOV only, not depth-buffer visibility or a retained-fraction threshold.
    if (elevation < -23.).any() or (elevation > 3.).any():
        raise PlacementRejected("outside_sensor_fov")
    donor = np.column_stack((moved, patch[:, 3])).astype(np.float32)
    return donor, dict(yaw=float(yaw), translation=translation.tolist(), rotation=rotation.tolist(),
                       height_estimate=floor, height_source="local_10pct_z" if len(local) else "scan_10pct_z",
                       added_points=len(donor), removed_target_points=0)


def basic_copy_paste(points, packed, database, learning_map, rng, config):
    """Select at most three donors; bounded placement retries never force success."""
    points, packed = np.asarray(points), np.asarray(packed)
    if (points.ndim != 2 or points.shape[1] != 4 or packed.ndim != 1
            or packed.dtype != np.uint32 or len(points) != len(packed)
            or not np.isfinite(points).all()):
        raise ValueError("invalid_target_points_labels")
    map_semantic(packed, learning_map)
    limit, trials = config["max_objects_per_scan"], config["max_candidate_trials"]
    if not isinstance(limit, int) or not 1 <= limit <= 3 or not isinstance(trials, int) or trials < 1:
        raise ValueError("Basic CP requires 1..3 objects and positive bounded trials")
    occupied = set((packed >> 16).tolist())
    selected, additions, labels, placements = set(), [], [], []
    attempts, successes, rejects = Counter(), Counter(), Counter()
    candidate_trials = 0
    for _ in range(limit):
        row, patch = database.sample(rng)
        cid = int(row["learning_id"])
        if cid not in CLASSES or not np.all(map_semantic(patch[:, 4].astype(np.uint32), learning_map) == cid):
            raise ValueError("donor_class_mismatch")
        if row["donor_id"] in selected:
            rejects["duplicate_donor"] += 1
            continue
        selected.add(row["donor_id"])
        attempts[str(cid)] += 1
        margin = float(np.linalg.norm(patch[:, :2], axis=1).max()) + .05
        low, high = 3. + margin, 51.2 - margin
        if low >= high:
            rejects["object_too_large"] += 1
            continue
        for _ in range(trials):
            candidate_trials += 1
            angle = rng.uniform(-np.pi, np.pi)
            radius = np.sqrt(rng.uniform(low ** 2, high ** 2))
            center = radius * np.array([np.cos(angle), np.sin(angle)])
            try:
                donor, metadata = place_basic(points, patch, learning_map, center, rng.uniform(-np.pi, np.pi))
            except PlacementRejected as exc:
                rejects[str(exc)] += 1
                continue
            instance = next((i for i in range(1, 65536) if i not in occupied), None)
            if instance is None:
                rejects["instance_ids_exhausted"] += 1
                break
            occupied.add(instance)
            additions.append(donor)
            labels.append(patch[:, 4].astype(np.uint32) | np.uint32(instance << 16))
            successes[str(cid)] += 1
            placements.append(dict(metadata, learning_id=cid, synthetic_instance_id=instance,
                                   donor_id=row["donor_id"], donor_sha256=row["sha256"],
                                   source_sequence=row["source_sequence"], source_frame=row["source_frame"],
                                   source_instance_id=row["source_instance_id"]))
            break
    after = np.concatenate([points, *additions]) if additions else points
    after_labels = np.concatenate([packed, *labels]) if labels else packed
    paste = np.arange(len(after)) >= len(points)
    assert len(after) == len(after_labels) and len(placements) <= limit
    return dict(points=after, packed=after_labels, paste_points=paste, placements=placements,
                successful_objects=len(placements), object_attempts=sum(attempts.values()),
                candidate_trials=candidate_trials, class_attempts=dict(attempts),
                class_successes=dict(successes), rejection_reasons=dict(rejects), added_points=int(paste.sum()))


def object_counts(rows):
    counts = Counter({str(i): 0 for i in range(4)})
    for row in rows:
        count = int(row.get("successful_objects", sum(row["class_successes"].values())))
        if not 0 <= count <= 3:
            raise ValueError("invalid_success_count")
        counts[str(count)] += 1
    return dict(counts)
