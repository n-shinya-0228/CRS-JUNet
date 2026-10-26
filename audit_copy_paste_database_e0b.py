"""Audit every old donor .npy without changing it; emit a separate quality manifest."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from lib.utils.pointnet_copy_paste_e0 import sha256
from lib.utils.raw_point_copy_paste_e0b import inspect_patch, QUALITY_RULES


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["path", "reasons"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def audit_database(database, output, data, log=print):
    database, output = Path(database).resolve(), Path(output)
    files = sorted(database.glob("*.npy"))
    rows, seen = [], {}
    start = time.perf_counter()
    for number, path in enumerate(files):
        row = {"path": str(path), "file": path.name, "bytes": path.stat().st_size,
               "sha256": sha256(path), "provenance_verified": False, "source_sequence": "unknown",
               "source_scan": "unknown", "source_instance": "not stored"}
        try:
            array = np.load(path, allow_pickle=False)
            row.update(inspect_patch(array, path.name, data["learning_map"]))
            fingerprint = hashlib.sha256(array.tobytes()).hexdigest()
            if fingerprint in seen:
                row["reasons"].append("duplicate_patch_content")
                row["duplicate_of"] = seen[fingerprint]
            else:
                seen[fingerprint] = path.name
        except Exception as error:
            row["reasons"] = ["load_error:" + type(error).__name__]
            row["error"] = str(error)
        row["usable_for_audit"] = not row["reasons"]
        row["usable_for_formal_training"] = False
        rows.append(row)
        if (number + 1) % 5000 == 0:
            log(f"Database audit: {number + 1}/{len(files)} files")
    grouped = defaultdict(list)
    raw_grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("learning_id", -1)].append(row)
        raw_grouped[row.get("filename_raw_id", -1)].append(row)
    stats, raw_stats, distribution = [], [], {}
    for name, groups, destination in (("learning_id", grouped, stats), ("raw_id", raw_grouped, raw_stats)):
        for key, group in sorted(groups.items()):
            point_counts = [r.get("points", 0) for r in group]
            entry = {name: key, "files": len(group), "median_points": float(np.median(point_counts)),
                     "min_points": min(point_counts), "max_points": max(point_counts),
                     "mixed_label_files": sum(len(r.get("raw_ids", [])) > 1 for r in group),
                     "usable_for_audit": sum(r["usable_for_audit"] for r in group)}
            if name == "learning_id" and key in data["learning_map_inv"]:
                entry["class"] = data["labels"][data["learning_map_inv"][key]]
            destination.append(entry)
            for quantity in ("points", "length", "width", "height", "max_center_distance"):
                values = [r[quantity] for r in group if quantity in r]
                if values:
                    distribution[f"{name}_{key}_{quantity}"] = dict(zip(
                        ["min", "p10", "median", "p90", "p99", "max"], np.percentile(values, [0, 10, 50, 90, 99, 100]).tolist()))
    for cls in (2, 3, 4, 5, 6, 7, 8, 18, 19):
        if cls not in grouped:
            stats.append(dict(learning_id=cls, files=0, median_points=None, min_points=None,
                              max_points=None, mixed_label_files=0, usable_for_audit=0,
                              **{"class": data["labels"][data["learning_map_inv"][cls]]}))
    reasons = Counter(reason for row in rows for reason in row["reasons"])
    usable = sum(row["usable_for_audit"] for row in rows)
    summary = dict(total_files=len(files), usable_for_audit=usable, invalid_or_excluded=len(files)-usable,
                   usable_fraction=usable / max(len(files), 1), exclusion_reason_counts=reasons,
                   files_sha256_manifest="database_manifest.json", audit_seconds=time.perf_counter()-start,
                   provenance_verified=False, formal_training_usable=0,
                   mixed_label_files=sum(len(row.get("raw_ids", [])) > 1 for row in rows))
    write_json(output / "database_manifest.json", rows)
    write_csv(output / "database_manifest.csv", rows)
    write_csv(output / "invalid_patches.csv", (r for r in rows if not r["usable_for_audit"]))
    write_csv(output / "database_statistics.csv", sorted(stats, key=lambda r:r["learning_id"]))
    write_csv(output / "database_raw_id_statistics.csv", raw_stats)
    write_json(output / "database_distributions.json", distribution)
    write_json(output / "database_audit_summary.json", summary)
    write_json(output / "quality_thresholds.json", {"rules_by_learning_id": QUALITY_RULES,
        "relative_mean_tolerance_m": .001, "duplicate_point_fraction_max": .05,
        "basis": "Measured full DB quantiles in database_distributions.json; point counts and extent probe performed before gates were selected. Class-specific conservative geometry filters, not object-completeness certification.",
        "riders": "7/8 excluded without original vehicle alignment; cannot place rider fragments as grounded standalone objects"})
    write_json(output / "donor_provenance_audit.json", {
        "status": "source unverified", "japanese_status": "出典未確認", "verified_train_donors": 0,
        "database_files": len(files), "source_instances_stored": False,
        "source_sequence_and_scan_stored": False,
        "build_database_source_sha256": sha256("build_database.py"),
        "code_train_sequences": data["split"]["train"],
        "history": "build_database.py commits d4ab6ec8/d3ae0e3f/da792b79 inspected; code is train-only but no per-file run manifest proves which code generated current .npy files",
        "validation_leakage_excluded_by_file_format": False,
        "audit_only_use": True, "formal_training_allowed": False,
        "next_action": "recover trustworthy generation manifest or build a NEW train-only instance-aware DB with source sequence/frame, packed raw labels, transform and content hashes; never overwrite old DB"})
    table = "\n".join(f"| {r.get('class',r['learning_id'])} | {r['learning_id']} | {r['files']} | {r['median_points']} | {r['min_points']} | {r['max_points']} | {r['mixed_label_files']} | {r['usable_for_audit']} |" for r in sorted(stats,key=lambda r:r["learning_id"]))
    (output / "database_audit.md").write_text(f"""# E.0b Database Audit

All {len(files)} `.npy` files inspected; original bytes are not changed.
Quality-usable for **audit only**: {usable} ({usable/max(len(files),1):.2%});
excluded: {len(files)-usable}; mixed raw-label files: {summary['mixed_label_files']}.
Formal-training usable: **0 until donor provenance is verified**.

| Class | Learning ID | Files | Median points | Min | Max | Mixed | Audit usable |
| --- | --- | --- | --- | --- | --- | --- | --- |
{table}

## Threshold Rationale

The initial full-DB distribution probe found median points53 (raw11),118 (15),
54 (18),75 (20),56 (30),48 (254), with p10 mostly12-15; many are partial surfaces.
Ten points cannot certify shape. We require bicycle30, motorcycle40, truck150,
other-vehicle100, person30 points, plus per-class PCA length/width/height ranges
in `quality_thresholds.json`. Bounds were chosen conservatively after inspecting
measured point/extent distributions, not fitted to validation or success counts.
They reject many incomplete large-vehicle fragments. PCA dimensions remove global
yaw dependence. Broad other-vehicle limits account for bus-like raw257 without
pretending raw257 is a truck. Size caps flag potentially merged same-class bodies,
but single raw class does not prove a single instance. Class7/8 riders have no
vehicle-relative alignment and are excluded rather than planted standing on road.

Float32,Nx5,finite/integer known raw labels, matching filename, meanXYZ within1mm,
remission[0,1], duplicate contents/points, and dimensions are checked. Content and
file hashes are recorded. Mixed labels are excluded even if they share a learning
ID. Reasons overlap; their sum is not the number of excluded files.

## Provenance And Labels

Original `build_database.py` clusters all target semantic classes together, records
the first point's label in the filename, subtracts the meanXYZ, then saves five
float32 columns. DBSCAN clusters may include border groups with fewer than10 points;
the measured minimum is reported, not assumed to equal min_samples.
Original instance ID, source frame/sequence and original center are lost.
No source is inferred from the global numeric filename suffix or timestamps.
The source code and Git history use training sequences only, but that does not
prove current files' provenance. Status: **出典未確認**. All successful insertions
in this audit explicitly opt into unverified-provenance diagnostic use only.

Raw ID11 is bicycle -> learning2; raw18 is truck ->4, NOT learning18(pole).
Actual mapping always comes from `semantic-kitti.yaml`; raw257 maps to5.
There are no pole80/sign81 filename donors. Raw31/32 may occur inside excluded
mixed patches; moving253/255 map to riders7/8 but are not standalone-safe here.

Files: complete CSV/JSON manifest, invalid CSV, per-learning/per-raw statistics,
full quantiles, thresholds, provenance audit and summary JSON. No DB was rebuilt.
""")
    log(f"Database audit finished: {usable}/{len(files)} quality-usable (provenance still unverified)")
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="copy_paste_database")
    parser.add_argument("--output", default="logs/SJNet3_PointNet_stepE0b_database_audit")
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve().parent != Path.cwd() / "logs" or "stepE0b" not in output.name:
        raise ValueError("Use a new E.0b logs directory")
    output.mkdir(exist_ok=False)
    def log(message):
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with (output / "execution_log.txt").open("a") as stream:
            stream.write(line + "\n")
    log("E.0b full database audit only; no formal training.")
    audit_database(args.database, output, yaml.safe_load(Path("config/labels/semantic-kitti.yaml").read_text()), log)
