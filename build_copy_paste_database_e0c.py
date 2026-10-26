"""Build a new instance-aware DB from raw training sequences; no legacy DB reads."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from lib.utils.raw_point_copy_paste_e0b import read_train_scan, map_semantic
from lib.utils.raw_point_copy_paste_e0c import CLASSES, SCHEMA, TRAIN_SEQUENCES, quality
from lib.utils.pointnet_copy_paste_e0 import sha256


def build(root, output, report, data_path):
    output, report = Path(output), Path(report)
    output.mkdir(exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    (output/"objects").mkdir()
    data = yaml.safe_load(Path(data_path).read_text())
    learning_map = data["learning_map"]
    sources = {}
    rows, candidates = [], []
    mixed, rider_pairs = Counter(), Counter()
    config = dict(schema=SCHEMA, sequences=TRAIN_SEQUENCES, classes=CLASSES,
                  instance_zero="exclude", mixed_raw="exclude_whole_instance", seed=0,
                  fragmentation_voxel_m=dict(small=.6, large=1.2), largest_component_min=.9,
                  isolated_outlier_max=.05)
    config_hash = __import__("hashlib").sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    code_hashes = {p: sha256(p) for p in ("build_copy_paste_database_e0c.py", "lib/utils/raw_point_copy_paste_e0c.py",
                                        "lib/utils/raw_point_copy_paste_e0b.py", data_path)}
    provenance = dict(schema=SCHEMA, complete=False, started=datetime.now(timezone.utc).isoformat(), seed=0,
                      git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      config=config, config_sha256=config_hash, code_hashes=code_hashes,
                      dataset_root=str(Path(root).resolve()))
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    paths = [p for seq in TRAIN_SEQUENCES for p in sorted((Path(root)/"sequences"/seq/"velodyne").glob("*.bin"))]
    expected = 19130
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} train scans, got {len(paths)}")
    with (output/"candidates.jsonl").open("w") as journal:
        for scan_index, path in enumerate(paths):
            points, packed = read_train_scan(path, learning_map, TRAIN_SEQUENCES)
            ids, mapped = packed >> 16, map_semantic(packed, learning_map)
            label_path = path.parent.parent/"labels"/(path.stem+".label")
            seq = path.parent.parent.name
            source_id = f"{seq}/{path.stem}"
            sources[source_id] = dict(bin=str(path.resolve()), label=str(label_path.resolve()),
                                      bin_sha256=sha256(path), label_sha256=sha256(label_path))
            for instance in np.unique(ids[np.isin(mapped, list(CLASSES)+( [7, 8]))]):
                selection = ids == instance
                semantics = np.unique(packed[selection] & 65535)
                classes = np.unique(mapped[selection])
                if instance == 0:
                    # Zero is not one object; only count its semantic-class exclusion.
                    for cid in classes[np.isin(classes, CLASSES)]:
                        candidate = dict(source_sequence=seq, source_frame=path.stem, source_instance_id=0,
                                         learning_id=int(cid), point_count=int(np.count_nonzero(selection & (mapped==cid))),
                                         reasons=["instance_zero_not_separable"])
                        candidates.append(candidate)
                        journal.write(json.dumps(candidate)+"\n")
                    continue
                if len(semantics) > 1:
                    mixed[str(semantics.tolist())] += 1
                if any(c in classes for c in (7, 8)):
                    rider_pairs[str(classes.tolist())] += 1
                if not any(c in CLASSES for c in classes):
                    continue
                obj, labels = points[selection], packed[selection]
                patch, center, check = quality(obj, labels, learning_map)
                cid = int(classes[0])
                candidate = dict(check, source_sequence=seq, source_frame=path.stem, source_instance_id=int(instance),
                                 learning_id=cid, raw_ids=semantics.tolist(), point_count=len(obj))
                candidates.append(candidate)
                journal.write(json.dumps(candidate)+"\n")
                if check["reasons"]:
                    continue
                donor_id = f"{seq}_{path.stem}_{int(instance):05d}"
                relative_file = f"objects/{donor_id}.npz"
                np.savez_compressed(output/relative_file, xyz_relative=patch[:, :3], xyz_original=obj[:, :3],
                                    remission=obj[:, 3], raw_semantic=labels & 65535, packed_labels=labels,
                                    source_point_indices=np.flatnonzero(selection), original_center=center,
                                    original_xyz_min=obj[:, :3].min(0), original_xyz_max=obj[:, :3].max(0),
                                    source_sequence=seq, source_frame=path.stem, source_instance_id=int(instance),
                                    learning_id=cid, point_count=len(obj), schema_version=SCHEMA)
                rows.append(dict(donor_id=donor_id, file=relative_file, sha256=sha256(output/relative_file),
                                 source_sequence=seq, source_frame=path.stem, source_instance_id=int(instance),
                                 raw_ids=semantics.tolist(), learning_id=cid, point_count=len(obj),
                                 original_center=center.tolist(), bbox_min=obj[:, :3].min(0).tolist(),
                                 bbox_max=obj[:, :3].max(0).tolist(), quality=check, split="train",
                                 source=sources[source_id], extraction_code_hashes=code_hashes, config_sha256=config_hash))
            if scan_index % 500 == 0:
                print(f"DB scans {scan_index}/{len(paths)} | approved={len(rows)}", flush=True)
    (output/"manifest.json").write_text(json.dumps(rows, indent=2))
    with (output/"manifest.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    statistics = {}
    for cid in CLASSES:
        c = [row for row in candidates if row["learning_id"] == cid]
        passed = [row for row in rows if row["learning_id"] == cid]
        statistics[cid] = dict(candidates=len(c), passed=len(passed), failed=len(c)-len(passed),
                              valid_instance_rate=sum(bool(r["source_instance_id"]) for r in c)/max(1,len(c)),
                              rejection_reasons=dict(Counter(reason for r in c for reason in r["reasons"])),
                              source_sequences=dict(Counter(r["source_sequence"] for r in passed)),
                              distributions={key: np.quantile([r["quality"][key] for r in passed], [0,.25,.5,.75,1]).tolist()
                                             for key in ("points", "length", "width", "height")} if passed else {})
    audit = dict(statistics=statistics, mixed_instance_semantics=dict(mixed), rider_instance_compositions=dict(rider_pairs),
                 riders_included=False, rider_reason="No authoritative rider/vehicle association; proximity is insufficient",
                 completeness="All observed points of one valid instance; unseen physical surfaces cannot be guaranteed",
                 source_scans=len(paths), donor_count=len(rows), config=config)
    for directory in (output, report):
        (directory/"database_quality_audit.json").write_text(json.dumps(audit, indent=2))
    provenance.update(complete=True, finished=datetime.now(timezone.utc).isoformat(), source_scans=len(paths),
                      manifest_sha256=sha256(output/"manifest.json"), donor_count=len(rows))
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    (report/"database_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--output", default="copy_paste_database_e0c_trainonly")
    parser.add_argument("--report", default="logs/SJNet3_PointNet_stepE0c_database_training_setup")
    parser.add_argument("--data-cfg", default="config/labels/semantic-kitti.yaml")
    args = parser.parse_args()
    build(args.dataset, args.output, args.report, args.data_cfg)
