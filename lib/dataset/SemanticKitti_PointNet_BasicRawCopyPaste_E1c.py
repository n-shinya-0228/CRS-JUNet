"""Independent Basic mode; the legacy OFF/advanced implementation is untouched."""

import time
import numpy as np

from lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1 import SemanticKitti as LegacyDataset
from lib.utils.raw_point_copy_paste_e0b import map_semantic, read_train_scan
from lib.utils.raw_point_copy_paste_e0c import TRAIN_SEQUENCES, stable_seed
from lib.utils.raw_point_copy_paste_basic_e1c import basic_copy_paste


class SemanticKitti(LegacyDataset):
    def __init__(self, root, data, cp_config, enabled=False):
        super().__init__(root, data, cp_config, enabled)
        self.mode = cp_config.get("mode", "advanced")
        if self.mode not in ("basic", "advanced") or not 0 <= cp_config["probability"] <= 1:
            raise ValueError("Invalid Copy-Paste mode/probability")

    def __getitem__(self, index):
        if not self.enabled or self.mode == "advanced":
            return super().__getitem__(index)
        started = time.perf_counter()
        path = self.files[index]
        points, packed = read_train_scan(path, self.data["learning_map"], TRAIN_SEQUENCES)
        histogram = np.bincount(map_semantic(packed, self.data["learning_map"]), minlength=20).tolist()
        stats = dict(scans=1, attempted_scans=0, successful_scans=0, fallback=0, candidate_trials=0,
                     successful_objects=0, object_attempts=0, class_attempts={}, class_successes={},
                     rejection_reasons={}, added_points=0, removed_points=0, new_observed_cells=0,
                     labels_before=histogram, labels_after=histogram)
        rng = np.random.default_rng(stable_seed(self.epoch, index, "copy_paste"))
        before, paste, placements = None, None, []
        if rng.random() < self.cp["probability"]:
            stats["attempted_scans"] = 1
            result = basic_copy_paste(points, packed, self.database, self.data["learning_map"], rng, self.cp)
            if result["successful_objects"]:
                before = points
                points, packed, paste = result["points"], result["packed"], result["paste_points"]
                placements = result["placements"]
            for key in ("candidate_trials", "successful_objects", "object_attempts", "class_attempts",
                        "class_successes", "rejection_reasons", "added_points"):
                stats[key] = result[key]
            stats.update(successful_scans=int(bool(placements)), fallback=int(not placements),
                         labels_after=np.bincount(map_semantic(packed, self.data["learning_map"]), minlength=20).tolist())
        stats["worker_seconds"] = time.perf_counter() - started
        return dict(points=points, packed=packed, paste_points=paste, before_points=before,
                    placement=placements, stats=stats, index=index, epoch=self.epoch,
                    scan_id=f"{path.parent.parent.name}/{path.stem}",
                    augmentation_seed=stable_seed(self.epoch, index, "regular_augmentation"))
