"""Train-only nine-class donors and context-guided proposals on the E.1d polar grid."""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lib.utils.pointnet_copy_paste_e0 import sha256
from lib.utils.raw_point_copy_paste_e0b import PlacementRejected, deterministic_labels, map_semantic, rigid_place
from lib.utils.raw_point_copy_paste_e0c import check_sequence, stable_seed
from lib.utils.raw_point_copy_paste_paper_e1d import grid, select_cells, tally, epoch_stats

CLASSES = (2, 3, 4, 5, 6, 7, 8, 18, 19)


class CombinedDonors:
    def __init__(self, legacy, root):
        self.legacy, self.root = legacy, Path(root).resolve()
        self.groups = dict(legacy.groups)
        provenance = json.loads((self.root/"provenance.json").read_text())
        if not provenance["complete"] or provenance["manifest_sha256"] != sha256(self.root/"manifest.json"):
            raise ValueError("incomplete E.1e manifest")
        for row in json.loads((self.root/"manifest.json").read_text()):
            cid = row["learning_id"]
            check_sequence(row["source_sequence"])
            if cid not in CLASSES[5:] or row["split"] != "train" or row["reasons"]:
                raise ValueError("unexpected E.1e donor")
            self.groups.setdefault(cid, []).append(row)
        self.rows = [r for c in CLASSES for r in self.groups[c]]
        if len({r["donor_id"] for r in self.rows}) != len(self.rows) or not all(self.groups.values()):
            raise ValueError("duplicate donor ID or missing class")

    def load(self, row):
        if row["learning_id"] in self.legacy.groups:
            return self.legacy.load(row)
        file = (self.root / row["file"]).resolve()
        if not file.is_relative_to(self.root) or sha256(file) != row["sha256"]:
            raise ValueError("E.1e donor changed")
        with np.load(file, allow_pickle=False) as donor:
            p, packed = donor["points"], donor["packed_labels"]
            if not np.all(map_semantic(packed, self.legacy.learning_map) == row["learning_id"]):
                raise ValueError("E.1e label mismatch")
            return np.column_stack((p[:,:3]-p[:,:3].mean(0,dtype=np.float64), p[:,3], packed & 65535)).astype(np.float32)


def pose(patch, center, yaw, floor, clearance):
    translation = np.r_[center, floor + clearance - float(patch[:,2].min())]
    moved, _ = rigid_place(patch[:,:3],yaw,translation)
    radius = np.hypot(moved[:,0],moved[:,1])
    elevation = np.degrees(np.arctan2(moved[:,2],radius))
    if not np.isfinite(moved).all() or (radius <= 0).any() or (radius >= 51.2).any():
        raise PlacementRejected("outside_polar_radius")
    if (elevation < -23).any() or (elevation > 3).any():
        raise PlacementRejected("outside_sensor_fov")
    return np.column_stack((moved,patch[:,3])), dict(yaw=float(yaw), translation=translation.tolist())


def unique_mask(ids, seen):
    # Sorted, unique cells from the actual projection. Never draw visibility twice for this mask.
    key = ids.tobytes()
    if key in seen:
        raise PlacementRejected("duplicate_mask")
    seen.add(key)


def augment(sample, db, mapping, cp, device):
    epoch, index = sample["epoch"], sample["index"]
    rng = np.random.default_rng(stable_seed(epoch,index,"copy_paste"))
    if not cp["enabled"] or rng.random() >= cp["probability"]:
        return
    check_sequence(sample["scan_id"].split("/")[0])
    start = time.monotonic()
    vis_rng = np.random.default_rng(stable_seed(epoch,index,"paper_high_visibility"))
    points, packed, stats = sample["points"], sample["packed"], sample["stats"]
    ids, inverse, valid = grid(points,device)
    target = deterministic_labels(ids,inverse,packed[valid],mapping).numpy().ravel()
    policy = cp["paper"]
    backgrounds = {c:tuple(policy["allowed_background"][c]) for c in CLASSES}
    anchors = {bg:np.flatnonzero(np.isin(target,bg)) for bg in set(backgrounds.values())}
    support = points[valid]
    tree = cKDTree(support[:,:2])
    seen_donors, seen_cells, seen_masks = set(), set(), set()
    occupied = set(np.unique(packed >> 16))
    additions, labels, placements = [], [], []
    selected, pasted, rejects, by_class = Counter(), Counter(), Counter(), Counter()
    budget = cp["search_budget"]
    trials = exhausted = 0
    while len(placements) < cp["max_objects_per_scan"] and trials < budget:
        if time.monotonic()-start > cp["search_timeout_seconds"]:
            # A slow host must not silently change which donors enter a seeded run.
            raise TimeoutError(f"E.1e proposal timeout: {sample['scan_id']}, trials={trials}")
        cid = int(rng.choice(CLASSES))
        row = db.groups[cid][int(rng.integers(len(db.groups[cid])))]
        selected[str(cid)] += 1
        if row["donor_id"] in seen_donors:
            trials += 1
            rejects["duplicate_donor"] += 1
            by_class[f"{cid}:duplicate_donor"] += 1
            continue
        seen_donors.add(row["donor_id"])
        patch = db.load(row)
        clearance = row["ground_clearance"] if cid == 19 else .03
        margin = float(np.linalg.norm(patch[:,:2], axis=1).max()) + .05
        # Propose ranges likely to fit the upper FOV; the full 3D test remains mandatory.
        lower = max(3., (float(np.ptp(patch[:,2]))+clearance-1.8)/np.tan(np.deg2rad(3)))
        pool = anchors[backgrounds[cid]]
        radii = (pool//512 + .5)*.1
        pool = pool[(radii >= lower+margin) & (radii < 51.2-margin)]
        accepted = False
        for _ in range(min(cp["trials_per_donor"],budget-trials)):
            trials += 1
            try:
                if not len(pool):
                    raise PlacementRejected("no_eligible_anchor")
                cell = int(rng.choice(pool))
                if cell in seen_cells or target[cell] not in backgrounds[cid]:
                    raise PlacementRejected("unavailable_anchor")
                seen_cells.add(cell)
                radius = (cell//512 + .5)*.1
                angle = (cell%512 + .5)*(2*np.pi/512)-np.pi
                center = radius*np.array([np.cos(angle),np.sin(angle)])
                near = tree.query_ball_point(center,2.)
                floor = float(np.quantile((support[near] if near else support)[:,2],.1))
                donor, transform = pose(patch,center,rng.uniform(-np.pi,np.pi),floor,clearance)
                cells, inv, in_range = grid(donor,device)
                if not in_range.all():
                    raise PlacementRejected("outside_polar_radius")
                unique_mask(cells,seen_masks)
                keep, visible, scores = select_cells(cells[inv],target,cid,policy,vis_rng)
            except PlacementRejected as exc:
                rejects[str(exc)] += 1
                by_class[f"{cid}:{exc}"] += 1
                if not len(pool):
                    break
                continue
            instance = next(i for i in range(1,65536) if i not in occupied)
            occupied.add(instance)
            target[visible] = cid
            additions.append(donor[keep])
            labels.append(patch[keep,4].astype(np.uint32) | np.uint32(instance<<16))
            placements.append(dict(transform,**scores,learning_id=cid,donor_id=row["donor_id"],anchor=cell,
                                   synthetic_instance_id=instance,added_points=int(keep.sum())))
            pasted[str(cid)] += 1
            accepted = True
            break
        exhausted += int(not accepted)
    n = len(placements)
    stats.update(attempted_scans=1,successful_scans=int(n>0),fallback=int(n==0),candidate_trials=trials,
                 class_attempts=dict(selected),class_successes=dict(pasted),rejection_reasons=dict(rejects),
                 class_rejections=dict(by_class),exhausted_objects=exhausted,search_exhausted=int(n<3),
                 score_sums={k:sum(p[k] for p in placements) for k in ("context","overlap","visibility")})
    sample["placement"] = placements
    if n:
        new_labels = np.concatenate(labels)
        after = np.concatenate([points,*additions])
        sample.update(before_points=points,points=after,packed=np.concatenate([packed,new_labels]),
                      paste_points=np.arange(len(after))>=len(points))
        stats.update(added_points=len(new_labels),labels_after=(np.asarray(stats["labels_before"])
                     +np.bincount(map_semantic(new_labels,mapping),minlength=20)).tolist())


def extra_tally(counts, stats):
    for key, prefix in (("class_attempts","selected_"),("class_rejections","failed_")):
        for name,v in stats.get(key,{}).items():
            counts[prefix+name] += v
    counts["search_exhausted"] += stats.get("search_exhausted",0)


def summary(counts):
    result = epoch_stats(counts)
    n = counts["attempted_scans"]
    for i in range(4):
        v = counts["scans_with_"+str(i)] if i else n-counts["successful_scans"]
        result[f"attempted_with_{i}"] = v
        result[f"rate_{i}"] = v/n if n else None
    for name,key in (("success_rate","successful_scans"),("objects_per_attempt","successful_objects")):
        result[name] = counts[key]/n if n else None
    for c in CLASSES:
        result.setdefault(f"class_{c}",0)
        result.setdefault(f"selected_{c}",0)
    return result
