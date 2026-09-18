import csv
import json
from pathlib import Path


ROOT = Path("point_encoder_balanced_sampling")
STAGE1_RESULTS = Path(
    "point_encoder_tuning/stage1_20260902_seed31415/stage1_results.csv"
)
EXPERIMENTS = [
    ("control", None),
    ("cap4000", 4000),
    ("cap2000", 2000),
    ("cap1000", 1000),
    ("cap500", 500),
]
REPORT_CLASSES = [
    "bicycle",
    "motorcycle",
    "person",
    "bicyclist",
    "motorcyclist",
    "pole",
    "traffic-sign",
]


def percent(value):
    return round(100.0 * value, 4) if value is not None else None


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    result_rows = []
    result_records = []
    sampling_rows = []
    sampling_payloads = {}

    for experiment, cap in EXPERIMENTS:
        run_dir = ROOT / experiment
        summary = json.loads((run_dir / "summary.json").read_text())
        record = summary["best_stage2a"]

        result = {
            "experiment": experiment,
            "majority_cell_cap": cap,
            "best_epoch": record["epoch"],
            "val_mIoU_percent": percent(record["val_miou"]),
            "val_Acc_percent": percent(record["val_accuracy"]),
            "small_mIoU_percent": percent(record["small_miou"]),
            "core_small_mIoU_percent": percent(record["core_small_miou"]),
            "core_nonzero_over_5": (
                f"{record['core_small_positive_classes']}/5"
            ),
            "core_above_1_percent_over_5": (
                f"{record['core_small_above_one_percent_classes']}/5"
            ),
        }
        for class_name in REPORT_CLASSES:
            result[f"{class_name}_IoU_percent"] = percent(
                record["class_iou"][class_name]
            )

        result_rows.append(result)
        result_records.append({
            **result,
            "selection_source": str(run_dir / "summary.json"),
        })

        sampling = json.loads(
            (run_dir / "cell_sampling_stats.json").read_text()
        )
        sampling_payloads[experiment] = sampling
        for class_record in sampling["classes"]:
            sampling_rows.append({
                "experiment": experiment,
                "majority_cell_cap": cap,
                **class_record,
            })

    result_fieldnames = list(result_rows[0].keys())
    write_csv(ROOT / "stage2a_results.csv", result_rows, result_fieldnames)

    with open(STAGE1_RESULTS, "r", newline="") as f:
        stage1_rows = list(csv.DictReader(f))
    stage1_best = next(
        row for row in stage1_rows
        if row["experiment"] == "exp_006_eps1p05"
    )

    results_payload = {
        "selection_priority": [
            "core_nonzero_classes",
            "core_small_mIoU",
            "core_small_class_IoUs",
            "small_mIoU",
            "overall_mIoU",
            "accuracy",
        ],
        "stage1_best": stage1_best,
        "experiments": result_records,
    }
    with open(ROOT / "stage2a_results.json", "w") as f:
        json.dump(results_payload, f, indent=2, sort_keys=True)

    sampling_fieldnames = [
        "experiment",
        "majority_cell_cap",
        "class_id",
        "class_name",
        "before_cells",
        "after_cells",
        "retained_ratio",
        "retained_percent",
    ]
    write_csv(
        ROOT / "stage2a_sampling_stats.csv",
        sampling_rows,
        sampling_fieldnames,
    )
    with open(ROOT / "stage2a_sampling_stats.json", "w") as f:
        json.dump(sampling_payloads, f, indent=2, sort_keys=True)

    print(f"Wrote {ROOT / 'stage2a_results.csv'}")
    print(f"Wrote {ROOT / 'stage2a_results.json'}")
    print(f"Wrote {ROOT / 'stage2a_sampling_stats.csv'}")
    print(f"Wrote {ROOT / 'stage2a_sampling_stats.json'}")


if __name__ == "__main__":
    main()
