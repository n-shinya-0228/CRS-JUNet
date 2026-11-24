"""Paper's selection gates on observed polar cells, not angular depth-buffer visibility."""
from collections import Counter

import numpy as np
import torch

from lib.utils.laserscan_Pointnet import build_polar_point_features
from lib.utils.raw_point_copy_paste_basic_e1c import place_basic
from lib.utils.raw_point_copy_paste_e0b import PlacementRejected, deterministic_labels, map_semantic
from lib.utils.raw_point_copy_paste_e0c import stable_seed


def visibility_reason(ratio, policy, rng):
    if ratio < policy["min_visibility"]:
        return "minimum_visibility"
    if ratio > policy["high_visibility"] and rng.random() > policy["high_visibility_keep"]:
        return "high_visibility_sampling"
    return None


def select_cells(point_cells, target_labels, cid, policy, rng):
    cells, inverse = np.unique(point_cells, return_inverse=True)
    if not cells.size:
        raise PlacementRejected("empty_mask")
    target = target_labels[cells]
    blocked = np.isin(target, policy["protected"])
    n = len(cells)
    n_blocked = int(blocked.sum())
    context = np.count_nonzero(np.isin(target, policy["allowed_background"][cid])) / n
    occ, vis = n_blocked / n, (n - n_blocked) / n
    if context <= policy["context_min_exclusive"]:
        raise PlacementRejected("context")
    if occ > policy["max_overlap"][cid]:
        raise PlacementRejected("occlusion")
    reason = visibility_reason(vis, policy, rng)
    if reason:
        raise PlacementRejected(reason)
    return ~blocked[inverse], cells[~blocked], dict(context=context, overlap=occ, visibility=vis,
                                                   candidate_cells=n, blocked_cells=n_blocked)


def grid(points, device):
    # Same operator, FP32 precision and device as the subsequent whole-cloud projection.
    tensor = torch.from_numpy(points).to(device)
    _, cells, inverse = build_polar_point_features(tensor)
    rho = torch.sqrt(tensor[:, 0] ** 2 + tensor[:, 1] ** 2)
    return cells.cpu().numpy(), inverse.cpu().numpy(), ((rho > 0) & (rho < 51.2)).cpu().numpy()


def augment(sample, database, mapping, cp, device):
    stats = sample["stats"]
    epoch, index = sample["epoch"], sample["index"]
    rng = np.random.default_rng(stable_seed(epoch, index, "copy_paste"))
    if not cp["enabled"] or rng.random() >= cp["probability"]:
        return
    vis_rng = np.random.default_rng(stable_seed(epoch, index, "paper_high_visibility"))
    points, packed = sample["points"], sample["packed"]
    cells, inverse, valid = grid(points, device)
    target = deterministic_labels(cells, inverse, packed[valid], mapping).numpy().ravel()
    occupied = set(np.unique(packed >> 16).tolist())
    seen, additions, labels, placements = set(), [], [], []
    attempts, successes, rejects = Counter(), Counter(), Counter()
    exhausted = 0
    for _ in range(cp["max_objects_per_scan"]):
        row, patch = database.sample(rng)
        cid = row["learning_id"]
        if not np.all(map_semantic(patch[:, 4].astype(np.uint32), mapping) == cid):
            raise ValueError("donor_class_mismatch")
        if row["donor_id"] in seen:
            rejects["duplicate_donor"] += 1
            continue
        seen.add(row["donor_id"])
        attempts[str(cid)] += 1
        margin = float(np.linalg.norm(patch[:, :2], axis=1).max()) + .05
        lo, hi = 3. + margin, 51.2 - margin
        if lo >= hi:
            rejects["object_too_large"] += 1
            continue
        for trial in range(cp["max_candidate_trials"]):
            stats["candidate_trials"] += 1
            angle = rng.uniform(-np.pi, np.pi)
            center = np.sqrt(rng.uniform(lo ** 2, hi ** 2)) * np.array([np.cos(angle), np.sin(angle)])
            try:
                donor, transform = place_basic(points, patch, mapping, center, rng.uniform(-np.pi, np.pi))
                ids, inv, valid = grid(donor, device)
                if not valid.all():
                    raise PlacementRejected("outside_polar_radius")
                keep, visible_cells, scores = select_cells(ids[inv], target, cid, cp["paper"], vis_rng)
            except PlacementRejected as exc:
                rejects[str(exc)] += 1
                continue
            instance = next((i for i in range(1, 65536) if i not in occupied), None)
            if instance is None:
                rejects["instance_ids_exhausted"] += 1
                break
            occupied.add(instance)
            additions.append(donor[keep])
            labels.append(patch[keep, 4].astype(np.uint32) | np.uint32(instance << 16))
            target[visible_cells] = cid  # Reserve every retained cell before selecting the next object.
            successes[str(cid)] += 1
            placements.append(dict(transform, **scores, learning_id=cid, donor_id=row["donor_id"],
                                   synthetic_instance_id=instance, added_points=int(keep.sum()),
                                   candidate_points=len(donor), trial=trial + 1))
            break
        else:
            exhausted += 1
    n = len(placements)
    stats.update(attempted_scans=1, successful_scans=int(n > 0), fallback=int(n == 0),
                 class_attempts=dict(attempts), class_successes=dict(successes), rejection_reasons=dict(rejects),
                 exhausted_objects=exhausted, score_sums={k: sum(p[k] for p in placements)
                     for k in ("context", "overlap", "visibility")})
    sample["placement"] = placements
    if n:
        after = np.concatenate([points, *additions])
        new_labels = np.concatenate(labels)
        sample.update(points=after, packed=np.concatenate([packed, new_labels]), before_points=points,
                      paste_points=np.arange(len(after)) >= len(points))
        stats.update(added_points=len(new_labels), labels_after=(np.asarray(stats["labels_before"])
                     + np.bincount(map_semantic(new_labels, mapping), minlength=20)).tolist())


def tally(counts, stats):
    n = sum(stats["class_successes"].values())
    counts["scans"] += 1
    counts["attempted_scans"] += stats["attempted_scans"]
    counts["successful_scans"] += stats["successful_scans"]
    counts["successful_objects"] += n
    counts[f"scans_with_{n}"] += 1
    counts["exhausted_objects"] += stats.get("exhausted_objects", 0)
    counts["candidate_trials"] += stats["candidate_trials"]
    for name, value in stats["class_successes"].items():
        counts[f"class_{name}"] += value
    for name, value in stats["rejection_reasons"].items():
        counts[f"reject_{name}"] += value
    for name, value in stats.get("score_sums", {}).items():
        counts[f"sum_{name}"] += value


def epoch_stats(counts):
    result = dict(counts)
    for name in ("context", "occlusion", "minimum_visibility", "high_visibility_sampling"):
        result.setdefault("reject_" + name, 0)
    for i in range(4):
        result.setdefault(f"scans_with_{i}", 0)
    for i in range(2, 7):
        result.setdefault(f"class_{i}", 0)
    for name in ("context", "overlap", "visibility"):
        result["mean_" + name] = (counts["sum_" + name] / counts["successful_objects"]
                                  if counts["successful_objects"] else None)
    return result
