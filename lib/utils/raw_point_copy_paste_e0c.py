"""Train-only, source-traceable donors. E.0b geometry stays read-only."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.spatial import cKDTree

from lib.utils.pointnet_copy_paste_e0 import sha256
from lib.utils.raw_point_copy_paste_e0b import inspect_patch, map_semantic

TRAIN_SEQUENCES = ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10")
CLASSES = (2, 3, 4, 5, 6)
SCHEMA = "e0c.instance.v1"
ENCODER_SHA256 = "98d3ebe4c120b1cc44b3e830cece8d2f4fd3f2401a40a1fb558e811bea161e97"


def check_sequence(sequence):
    if str(sequence) not in TRAIN_SEQUENCES:
        raise ValueError("donor_or_target_not_train_split: " + str(sequence))


def stable_seed(epoch, index, stream):
    return int.from_bytes(hashlib.sha256(f"0:{epoch}:{index}:{stream}".encode()).digest()[:8], "little")


def quality(points, packed, learning_map):
    """Reject whole observed instances; never trim points to pass shape checks."""
    raw = np.unique(packed & 65535)
    instances = np.unique(packed >> 16)
    center = points[:, :3].mean(0, dtype=np.float64)
    relative = (points[:, :3].astype(np.float64) - center).astype(np.float32)
    patch = np.column_stack((relative, points[:, 3], packed & 65535)).astype(np.float32)
    result = inspect_patch(patch, f"class_{int(raw[0])}_0.npy", learning_map)
    if len(instances) != 1 or instances[0] == 0:
        result["reasons"].append("invalid_or_mixed_instance")
    if len(np.unique(map_semantic(packed, learning_map))) != 1:
        result["reasons"].append("mixed_learning_classes")
    if len(points) >= 2 and np.isfinite(points).all():
        cls = learning_map[int(raw[0])]
        spacing = 1.2 if cls in (4, 5) else .6
        nearest = cKDTree(relative).query(relative, k=2)[0][:, 1]
        result["isolated_point_fraction"] = float(np.mean(nearest > spacing))
        # Coarse connected support tests fragmentation without constructing a dense N^2 graph.
        voxels = np.floor((relative-relative.min(0))/spacing).astype(int)
        shape = voxels.max(0)+1
        if int(np.prod(shape)) > 2_000_000:
            result["reasons"].append("extreme_extent")
        else:
            grid = np.zeros(tuple(shape), dtype=bool)
            grid[tuple(voxels.T)] = True
            components, _ = connected_components(grid, structure=np.ones((3, 3, 3)))
            counts = np.bincount(components[tuple(voxels.T)])
            result["largest_component_point_fraction"] = float(counts[1:].max()/len(points))
            if result["largest_component_point_fraction"] < .9:
                result["reasons"].append("fragmented_instance_support")
        if result["isolated_point_fraction"] > .05:
            result["reasons"].append("isolated_outliers_gt_5pct")
    return patch, center, result


class DonorDatabase:
    def __init__(self, root, learning_map):
        self.root = Path(root).resolve()
        self.learning_map = learning_map
        provenance = json.loads((self.root/"provenance.json").read_text())
        if provenance.get("schema") != SCHEMA or provenance.get("complete") is not True:
            raise ValueError("incomplete_or_wrong_donor_database")
        self.rows = json.loads((self.root/"manifest.json").read_text())
        self.groups = {cid: [] for cid in CLASSES}
        for row in self.rows:
            check_sequence(row["source_sequence"])
            if (row["split"] != "train" or not row["source_instance_id"] or
                    row["quality"]["reasons"] or row["learning_id"] not in CLASSES):
                raise ValueError("invalid_donor_manifest")
            self.groups[row["learning_id"]].append(row)
        if any(not group for group in self.groups.values()):
            raise ValueError("missing_primary_donor_class")

    def load(self, row):
        check_sequence(row["source_sequence"])
        path = (self.root/row["file"]).resolve()
        if not path.is_relative_to(self.root) or sha256(path) != row["sha256"]:
            raise ValueError("donor_path_or_hash_mismatch")
        with np.load(path, allow_pickle=False) as donor:
            packed = donor["packed_labels"]
            points = np.column_stack((donor["xyz_original"], donor["remission"]))
            patch, center, result = quality(points, packed, self.learning_map)
            if (result["reasons"] or len(points) != row["point_count"] or
                    not np.all(packed >> 16 == row["source_instance_id"]) or
                    not np.array_equal(packed & 65535, donor["raw_semantic"]) or
                    not np.array_equal(patch[:, :3], donor["xyz_relative"]) or
                    str(donor["schema_version"]) != SCHEMA or
                    str(donor["source_sequence"]) != row["source_sequence"] or
                    str(donor["source_frame"]) != row["source_frame"] or
                    not np.array_equal(center, donor["original_center"])):
                raise ValueError("donor_content_or_provenance_mismatch")
        return patch

    def sample(self, rng):
        cid = int(rng.choice(CLASSES))
        row = self.groups[cid][int(rng.integers(len(self.groups[cid])))]
        return row, self.load(row)
