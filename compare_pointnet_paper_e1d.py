"""Read best-epoch metrics and measured object counts; write only into E.1d."""
import csv
import json
from pathlib import Path

RUN = Path("logs/SJNet3_PointNet_stepE1d_raw_cp_paper3")
NAMES = ("SJNet3_PointNet_stepE1a_raw_matched_cp0", "SJNet3_PointNet_stepE1b_raw_cp_on",
         "SJNet3_PointNet_stepE1c_raw_cp_basic3", RUN.name)
FOCUS = ("bicycle", "motorcycle", "truck", "other-vehicle", "person")


def locate(name):
    for parent in ("logs", "log_pointnet", "log_old"):
        path = Path(parent) / name
        if (path / "training_history.json").is_file():
            return path
    raise FileNotFoundError(name)


def report():
    summary = {}
    for arm, name in zip(("E1a", "E1b", "E1c", "E1d"), NAMES):
        folder = locate(name)
        history = json.loads((folder / "training_history.json").read_text())
        if not history:
            summary[arm] = {"status": "pending validation"}
            continue
        row = max(history, key=lambda v: v["miou"])
        classes = row["class_iou"]
        stats_file = folder / "cp_stats.csv"
        with stats_file.open() as stream:
            rows = {int(r["epoch"]): r for r in csv.DictReader(stream)}
        n = sum(sum(json.loads(r["class_successes"]).values()) for r in rows.values())
        attempted = sum(int(r["attempted_scans"]) for r in rows.values())
        successes = sum(int(r["successful_scans"]) for r in rows.values())
        scans = sum(int(r["scans"]) for r in rows.values())
        summary[arm] = dict(log=str(folder), epochs=len(history), best_epoch=row["epoch"], best_miou=row["miou"],
                            class_iou=classes, focus_mean=sum(classes[c] for c in FOCUS)/len(FOCUS),
                            pasted_objects=n, objects_per_scan=n/scans, attempted_scans=attempted,
                            successful_scans=successes, success_per_attempt=successes/attempted if attempted else None,
                            training_hours=sum(v["epoch_seconds"] for v in history)/3600)
    comparison = {}
    if "best_epoch" in summary["E1d"]:
        d = summary["E1d"]
        for arm in ("E1a", "E1b", "E1c"):
            ref = summary[arm]
            comparison[arm] = dict(mIoU=d["best_miou"]-ref["best_miou"], focus_mean=d["focus_mean"]-ref["focus_mean"],
                                  objects_per_scan=d["objects_per_scan"]-ref["objects_per_scan"],
                                  class_iou={k: v-ref["class_iou"][k] for k, v in d["class_iou"].items()})
    note = ("Best EMA validation epochs, not test-set metrics. Selection changes realized exposure; "
            "the E1d-E1c difference is not a pure causal effect of the gates.")
    tmp = RUN / "comparison.json.tmp"
    tmp.write_text(json.dumps(dict(runs=summary, E1d_minus=comparison, focus_classes=FOCUS, note=note), indent=2))
    tmp.replace(RUN / "comparison.json")
    with (RUN / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", *summary])
        for k in ("epochs", "best_epoch", "best_miou", "focus_mean", "pasted_objects", "objects_per_scan",
                  "attempted_scans", "successful_scans", "success_per_attempt", "training_hours"):
            writer.writerow([k, *(r.get(k) for r in summary.values())])
        for cls in summary["E1c"]["class_iou"]:
            writer.writerow([cls, *(r.get("class_iou", {}).get(cls) for r in summary.values())])


if __name__ == "__main__":
    report()
