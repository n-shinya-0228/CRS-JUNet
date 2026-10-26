"""Matched raw OFF/ON training; workers never own a CUDA encoder."""

from collections import Counter
from pathlib import Path
import time

import numpy as np
from torch.utils.data import Dataset

from lib.utils.raw_point_copy_paste_e0b import (PlacementRejected, PlacementRules, TargetScene,
                                              map_semantic, raw_copy_paste, read_train_scan)
from lib.utils.raw_point_copy_paste_e0c import DonorDatabase, TRAIN_SEQUENCES, check_sequence, stable_seed


def raw_collate(batch):
    return batch


class SemanticKitti(Dataset):
    def __init__(self, root, data, cp_config, enabled=False):
        self.root, self.data, self.cp = Path(root), data, cp_config
        sequences = tuple(f"{int(s):02d}" for s in data["split"]["train"])
        if set(sequences) != set(TRAIN_SEQUENCES):
            raise ValueError("E.1 requires the official complete train split")
        for seq in sequences:
            check_sequence(seq)
        self.files = [p for seq in sequences for p in sorted((self.root/"sequences"/seq/"velodyne").glob("*.bin"))]
        if len(self.files) != 19130:
            raise ValueError("Incomplete raw train split")
        self.enabled, self.epoch = bool(enabled), 0
        # Validate the identical DB even in the OFF arm, without sampling donors.
        self.database = DonorDatabase(cp_config["donor_database"], data["learning_map"])
        self.rules = PlacementRules()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        started = time.perf_counter()
        path = self.files[index]
        check_sequence(path.parent.parent.name)
        points, packed = read_train_scan(path, self.data["learning_map"], TRAIN_SEQUENCES)
        before_hist = np.bincount(map_semantic(packed, self.data["learning_map"]), minlength=20).tolist()
        rng = np.random.default_rng(stable_seed(self.epoch, index, "copy_paste"))
        stats = dict(scans=1, attempted_scans=0, successful_scans=0, fallback=0, candidate_trials=0,
                     class_attempts={}, class_successes={}, rejection_reasons={}, added_points=0, removed_points=0,
                     new_observed_cells=0, labels_before=before_hist, labels_after=before_hist)
        before_points, placement, paste_points = None, None, None
        if self.enabled and rng.random() < self.cp["probability"]:
            stats["attempted_scans"] = 1
            scene = TargetScene(points, packed, self.data["learning_map"])
            rejects, attempts = Counter(), Counter()
            for _ in range(self.cp["max_candidate_trials"]):
                row, patch = self.database.sample(rng)
                cid = row["learning_id"]
                attempts[str(cid)] += 1
                stats["candidate_trials"] += 1
                try:
                    center_xy = scene.sample_ground(cid, rng)
                    result = raw_copy_paste(scene, patch, self.data["learning_map"], center_xy,
                                            float(rng.uniform(-np.pi, np.pi)), self.rules)
                except PlacementRejected as exc:
                    rejects[str(exc)] += 1
                    continue
                before_points = points
                points, packed = result["points"], result["labels"]
                paste_points = result["paste_points"]
                # Synthetic pasted ID is distinct from every target ID; source ID stays in provenance.
                occupied_ids = set((scene.labels >> 16).tolist())
                new_id = next(i for i in range(1, 65536) if i not in occupied_ids)
                packed[paste_points] = (packed[paste_points] & 65535) | np.uint32(new_id << 16)
                stats.update(successful_scans=1, class_successes={str(cid): 1}, added_points=int(paste_points.sum()),
                             removed_points=int((~result["target_keep"]).sum()),
                             labels_after=np.bincount(map_semantic(packed, self.data["learning_map"]), minlength=20).tolist())
                placement = dict(result["metadata"], donor_id=row["donor_id"], donor_sha256=row["sha256"],
                                 source_sequence=row["source_sequence"], source_frame=row["source_frame"],
                                 source_instance_id=row["source_instance_id"], synthetic_instance_id=new_id,
                                 donor_instance_id=new_id, donor_source_instance_id=row["source_instance_id"])
                break
            stats.update(class_attempts=dict(attempts), rejection_reasons=dict(rejects),
                         fallback=int(not stats["successful_scans"]))
        stats["worker_seconds"] = time.perf_counter()-started
        return dict(points=points, packed=packed, paste_points=paste_points, before_points=before_points,
                    placement=placement, stats=stats, index=index, epoch=self.epoch,
                    scan_id=f"{path.parent.parent.name}/{path.stem}",
                    augmentation_seed=stable_seed(self.epoch, index, "regular_augmentation"))
