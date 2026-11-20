"""E.1 training unchanged, with a Basic dataset and additional object-count logging."""

from collections import Counter
import json

from trainer_PointNet_RawCopyPaste_E1 import Trainer as LegacyTrainer, atomic_json
from lib.dataset.SemanticKitti_PointNet_BasicRawCopyPaste_E1c import SemanticKitti
from lib.utils.pointnet_copy_paste_e0 import sha256
from lib.utils.raw_point_copy_paste_basic_e1c import object_counts


SOURCE_FILES = (
    "lib/utils/raw_point_copy_paste_basic_e1c.py",
    "lib/dataset/SemanticKitti_PointNet_BasicRawCopyPaste_E1c.py",
    "lib/trainer/trainer_PointNet_BasicRawCopyPaste_E1c.py",
    "train_pointnet_basic_raw_copy_paste_e1c.py",
)


class Trainer(LegacyTrainer):
    def __init__(self, arch, data, root, output, logger, enabled=True, resume=None):
        super().__init__(arch, data, root, output, logger, enabled, resume=None)
        self.dataset = SemanticKitti(root, data, arch["copy_paste"], enabled)
        self.identity["code"].update({path: sha256(path) for path in SOURCE_FILES})
        self.identity["cp_mode"] = "basic"
        if resume:
            self.load(resume)
        self.count_epoch = self.next_epoch
        self.counts = Counter(object_counts(self.partial["meters"]["cp"] if self.partial else []))

    def update(self, batch):
        result = super().update(batch)
        epoch = batch[0]["epoch"]
        if epoch != self.count_epoch:
            self.count_epoch, self.counts = epoch, Counter(object_counts([]))
        self.counts.update(object_counts([sample["stats"] for sample in batch]))
        # Per-update journal; after an unclean restart, deduplicate by (epoch, scan_id).
        # The epoch histogram is checkpoint-aligned and excludes replayed updates.
        with (self.output / "cp_scan_counts.jsonl").open("a") as stream:
            for sample in batch:
                stats = sample["stats"]
                stream.write(json.dumps(dict(epoch=epoch, scan_id=sample["scan_id"],
                    successful_objects=stats.get("successful_objects", 0),
                    attempted=stats["attempted_scans"], added_points=stats["added_points"],
                    class_successes=stats["class_successes"])) + "\n")
        return result

    def save(self, epoch, next_batch=0, partial=None, best=False, elapsed_seconds=None):
        super().save(epoch, next_batch, partial, best, elapsed_seconds)
        histogram = object_counts(partial["meters"]["cp"]) if partial else dict(self.counts)
        record = dict(epoch=epoch, next_batch=next_batch, complete_epoch=partial is None,
                      success_count_histogram=histogram,
                      successful_objects=sum(int(k) * v for k, v in histogram.items()),
                      scans=sum(histogram.values()))
        atomic_json(self.output / "basic_cp_progress.json", record)
        if partial is None:
            path = self.output / "basic_cp_epoch_history.json"
            history = json.loads(path.read_text()) if path.exists() else {}
            history[str(epoch)] = record
            atomic_json(path, history)
            self.logger.info(f"Basic CP | epoch {epoch} | objects 0/1/2/3: {histogram} | total objects {record['successful_objects']}")
