#!/usr/bin/env python3
"""Independent E.1c Basic CP experiment, gated by CPU and real CUDA checks."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib/trainer"))

import yaml
from loguru import logger
from trainer_PointNet_BasicRawCopyPaste_E1c import Trainer, SOURCE_FILES
from trainer_PointNet_RawCopyPaste_E1 import atomic_json
from lib.utils.pointnet_copy_paste_e0 import sha256

CONFIG = "config/arch/sjnet_pointnet_local3_raw_cp_e1c_basic3.yaml"
OUTPUT = "logs/SJNet3_PointNet_stepE1c_raw_cp_basic3"
PREFLIGHT = OUTPUT + "_preflight"
REFERENCES = {"E1a_OFF": "logs/SJNet3_PointNet_stepE1a_raw_matched_cp0",
              "E1b_advanced_max1": "logs/SJNet3_PointNet_stepE1b_raw_cp_on"}
CONDITIONS = {"cpu_tests", "real_scan_counts_0_to_3", "whole_cloud_regeneration",
              "batch3_forward_backward_frozen_encoder", "resume", "no_advanced_placement",
              "off_regression", "protected_inputs_unchanged"}


def check_config(arch):
    reference = yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1b_cp_on.yaml").read_text())
    matched = copy.deepcopy(arch)
    if matched["copy_paste"].pop("mode", None) != "basic" or matched["copy_paste"]["max_objects_per_scan"] != 3:
        raise ValueError("Require Basic mode with at most 3 objects")
    matched["copy_paste"]["max_objects_per_scan"] = 1
    if matched != reference:
        raise ValueError("E.1c may only change CP mode and maximum objects from E.1b")


def check_gate(path, config, data):
    gate = json.loads(Path(path).read_text())
    if gate.get("verdict") != "PASS" or not all(gate.get("conditions", {}).get(k) is True for k in CONDITIONS):
        raise ValueError("E.1c preflight has not passed")
    for name, checksum in gate["protected_inputs"].items():
        if sha256(name) != checksum:
            raise ValueError("Preflight input changed: " + name)
    if sha256(config) != gate["config_sha256"] or sha256(data) != gate["data_sha256"]:
        raise ValueError("Configuration changed after preflight")
    return gate


def comparison(output):
    output = Path(output)
    runs = dict(REFERENCES, E1c_basic_max3=str(output))
    summaries = {}
    for name, folder in runs.items():
        folder = Path(folder)
        result = json.loads((folder / "results.json").read_text())
        history = json.loads((folder / "training_history.json").read_text())
        best = next((row for row in history if row["epoch"] == result.get("best_epoch")), None)
        summaries[name] = dict(status=result["status"], best_epoch=result.get("best_epoch"),
                               best_miou=best["miou"] if best else None,
                               class_iou=best["class_iou"] if best else None)
    deltas = {}
    new = summaries["E1c_basic_max3"]
    for name in REFERENCES:
        old = summaries[name]
        deltas[name] = None if new["class_iou"] is None else dict(
            miou=new["best_miou"] - old["best_miou"],
            class_iou={cls: value - old["class_iou"][cls] for cls, value in new["class_iou"].items()})
    note = ("E1b changes both placement method and maximum object count (1 vs 3); "
            "differences cannot be attributed solely to the method. BEV-cell validation, not official point-wise test mIoU.")
    atomic_json(output / "comparison.json", dict(runs=summaries, delta_e1c_minus= deltas,
                focus_classes=["bicycle", "motorcycle", "truck", "other-vehicle", "person"], note=note))
    lines = ["# E.1c Basic CP Comparison", "", note, "", "| Experiment | Best epoch (0-based) | Best mIoU |",
             "|---|---:|---:|"]
    for name, row in summaries.items():
        score = "pending" if row["best_miou"] is None else f"{row['best_miou']:.9f}"
        lines.append(f"| {name} | {row['best_epoch']} | {score} |")
    lines += ["", "| Class | E1a OFF | E1b advanced/max1 | E1c Basic/max3 | Basic - OFF |", "|---|---:|---:|---:|---:|"]
    for cls, value in summaries["E1a_OFF"]["class_iou"].items():
        basic = "pending" if new["class_iou"] is None else f"{new['class_iou'][cls]:.6f}"
        delta = "pending" if new["class_iou"] is None else f"{new['class_iou'][cls] - value:+.6f}"
        lines.append(f"| {cls} | {value:.6f} | {summaries['E1b_advanced_max1']['class_iou'][cls]:.6f} | {basic} | {delta} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--arch_cfg", default=CONFIG)
    parser.add_argument("--data_cfg", default="config/labels/semantic-kitti.yaml")
    parser.add_argument("--log", default=OUTPUT)
    parser.add_argument("--gate", default=PREFLIGHT + "/verification.json")
    parser.add_argument("--resume")
    args = parser.parse_args()
    arch = yaml.safe_load(Path(args.arch_cfg).read_text())
    data = yaml.safe_load(Path(args.data_cfg).read_text())
    check_config(arch)
    gate = check_gate(args.gate, args.arch_cfg, args.data_cfg)
    output = Path(args.log)
    if output.exists() and not args.resume:
        parser.error("Refusing to overwrite existing experiment; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    logger.add(output / "train_log.txt", mode="a")
    for name, value in (("training_config.yaml", arch), ("dataset_config.yaml", data)):
        path = output / name
        if path.exists() and yaml.safe_load(path.read_text()) != value:
            raise ValueError("Resume configuration mismatch")
        if not path.exists():
            path.write_text(yaml.safe_dump(value, sort_keys=False))
    trainer = Trainer(arch, data, args.dataset, output, logger, True, args.resume)
    if trainer.initial_hash != gate["initial_state_sha256"] or trainer.validation_manifest_sha256 != gate["validation_manifest_sha256"]:
        raise ValueError("Initialization/validation differs from preflight")
    atomic_json(output / "process.json", dict(pid=os.getpid(), command=sys.argv, resume=args.resume))
    if not args.resume:
        atomic_json(output / "metadata.json", dict(identity=trainer.identity, cp_mode="basic", max_objects=3,
                    initial_state_sha256=trainer.initial_hash, scratch_segmentation=True,
                    pointnet_encoder_frozen=True, preflight=str(Path(args.gate).resolve()),
                    height="local 10th-percentile z (2 m); scan percentile fallback",
                    placement="area-uniform polar center; yaw uniform [-pi, pi]; full donor in sensor FOV/radius",
                    collision=False, visibility=False, ground_plane_fit=False,
                    scan_count_journal="Deduplicate replayed journal rows by (epoch, scan_id); epoch histograms are checkpoint-aligned"))
        atomic_json(output / "training_history.json", [])
        atomic_json(output / "results.json", dict(status="running", best_epoch=None, best_miou=None))
    with (output / "used_files.txt").open("a") as stream:
        stream.write("\nCommand: " + " ".join(sys.argv) + "\n" + json.dumps(trainer.identity, indent=2) + "\n")
    logger.info("Step E.1c | Basic CP p=0.5, maximum 3 objects | batch3 | frozen PointNet | seed0")
    comparison(output)
    trainer.train()
    comparison(output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("E.1c stopped with an error; existing checkpoints are preserved")
        raise
