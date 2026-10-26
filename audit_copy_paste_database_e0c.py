"""Verify every donor against its complete source instance and source hashes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from lib.utils.pointnet_copy_paste_e0 import sha256
from lib.utils.raw_point_copy_paste_e0c import DonorDatabase, TRAIN_SEQUENCES, check_sequence


def audit(root, data, output):
    database = DonorDatabase(root,data["learning_map"])
    sources, points, labels = {}, None, None
    last_source = None
    for index,row in enumerate(database.rows):
        check_sequence(row["source_sequence"])
        source = row["source"]
        key = row["source_sequence"]+"/"+row["source_frame"]
        if key != last_source:
            bin_path, label_path = Path(source["bin"]), Path(source["label"])
            if (bin_path.parent.parent.name != row["source_sequence"] or
                    label_path.parent.parent.name != row["source_sequence"] or
                    bin_path.stem != row["source_frame"] or label_path.stem != row["source_frame"] or
                    sha256(bin_path) != source["bin_sha256"] or sha256(label_path) != source["label_sha256"]):
                raise ValueError("source_hash_or_path_mismatch")
            points = np.fromfile(bin_path,dtype=np.float32).reshape(-1,4)
            labels = np.fromfile(label_path,dtype=np.uint32)
            if len(points) != len(labels):
                raise ValueError("source_point_label_count_mismatch")
            last_source = key
            sources[key] = source
        database.load(row)
        selection = np.flatnonzero(labels >> 16 == row["source_instance_id"])
        with np.load(database.root/row["file"],allow_pickle=False) as donor:
            if (not np.array_equal(selection,donor["source_point_indices"]) or
                    not np.array_equal(points[selection,:3],donor["xyz_original"]) or
                    not np.array_equal(points[selection,3],donor["remission"]) or
                    not np.array_equal(labels[selection],donor["packed_labels"]) or
                    not all(data["learning_map"][int(raw)] == row["learning_id"] for raw in row["raw_ids"])):
                raise ValueError("incomplete_instance_or_point_label_correspondence")
        if index % 1000 == 0:
            print(f"Provenance audit {index}/{len(database.rows)}",flush=True)
    provenance = json.loads((database.root/"provenance.json").read_text())
    if sha256(database.root/"manifest.json") != provenance["manifest_sha256"]:
        raise ValueError("manifest_hash_mismatch")
    result = dict(status="PASS", donors=len(database.rows), source_frames=len(sources),
                  class_counts={str(k):len(v) for k,v in database.groups.items()},
                  all_source_instances_complete=True, all_source_point_indices_and_packed_labels_exact=True,
                  validation_donors=0, train_sequences=list(TRAIN_SEQUENCES),
                  manifest_sha256=provenance["manifest_sha256"])
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"database_provenance_audit.json").write_text(json.dumps(result,indent=2))
    # Reports are copies in the new output tree, never mutations of legacy artifacts.
    (output/"database_manifest.csv").write_bytes((database.root/"manifest.csv").read_bytes())
    statistics = json.loads((database.root/"database_quality_audit.json").read_text())["statistics"]
    with (output/"database_statistics.csv").open("w") as file:
        writer = csv.DictWriter(file,fieldnames=["learning_id",*next(iter(statistics.values()))])
        writer.writeheader()
        writer.writerows(dict(learning_id=cid,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}) for cid,row in statistics.items())
    (output/"database_build_report.md").write_text("# E.0c Instance Database\n\n"+json.dumps(result,indent=2)+
        "\n\nAll observed source-instance points retained. Hidden physical surfaces are not inferred. "
        "Whole mixed-semantic/zero-ID instances rejected. No validation-based quality tuning. "
        "Rider/vehicle pairs excluded: no authoritative associations.\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database",default="copy_paste_database_e0c_trainonly")
    parser.add_argument("--output",default="logs/SJNet3_PointNet_stepE0c_database_training_setup")
    args = parser.parse_args()
    data = yaml.safe_load(Path("config/labels/semantic-kitti.yaml").read_text())
    print(json.dumps(audit(args.database,data,args.output),indent=2))
