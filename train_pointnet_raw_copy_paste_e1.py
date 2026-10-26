#!/usr/bin/env python3
"""Guarded entry point for matched single-pass C.2 raw OFF/ON scratch runs."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import yaml
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent/"lib/trainer"))
from trainer_PointNet_RawCopyPaste_E1 import Trainer, atomic_json
from lib.utils.pointnet_copy_paste_e0 import sha256


def check_gate(path):
    gate = json.loads(Path(path).read_text())
    required = {"cpu_tests", "all_donors_train_instance_source_verified", "c2_major_recipe_verified",
                "raw_7_16_label_mask_integrity_and_off_regression", "gpu_off_on_three_updates_resume_identical_init",
                "full_validation_regression_and_inputs_unchanged"}
    conditions = gate.get("mandatory_conditions", {})
    if gate.get("verdict") != "PASS" or not required.issubset(conditions) or not all(conditions.values()):
        raise ValueError("Formal E.1 requires all E.0c mandatory conditions to PASS")
    for filename, checksum in gate["protected_inputs"].items():
        if sha256(filename) != checksum:
            raise ValueError("Gate input changed: "+filename)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="SemanticKitti")
    parser.add_argument("--arch_cfg",default="config/arch/sjnet_pointnet_local3_raw_cp_e1.yaml")
    parser.add_argument("--data_cfg",default="config/labels/semantic-kitti.yaml")
    parser.add_argument("--copy-paste",choices=("off","on"),help="Optional override of config copy_paste.enabled")
    parser.add_argument("--log",required=True)
    parser.add_argument("--resume")
    parser.add_argument("--gate",default="logs/SJNet3_PointNet_stepE0c_database_training_setup/verification.json")
    args = parser.parse_args()
    gate = check_gate(args.gate)
    arch = yaml.safe_load(Path(args.arch_cfg).read_text())
    verified = yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1.yaml").read_text())
    matched=copy.deepcopy(arch)
    if not isinstance(arch["copy_paste"]["enabled"],bool):
        raise ValueError("copy_paste.enabled must be boolean")
    matched["copy_paste"]["enabled"]=verified["copy_paste"]["enabled"]
    if (matched != verified or sha256(args.data_cfg) != gate["protected_inputs"]["config/labels/semantic-kitti.yaml"]):
        raise ValueError("Formal config/label files differ from the verified recipe")
    if args.copy_paste is None:
        args.copy_paste="on" if arch["copy_paste"]["enabled"] else "off"
    output = Path(args.log)
    if output.exists() and not args.resume:
        parser.error("Refusing to overwrite an existing run; use --resume")
    output.mkdir(parents=True,exist_ok=True)
    logger.add(output/"train_log.txt",mode="a",enqueue=False)
    data = yaml.safe_load(Path(args.data_cfg).read_text())
    arch["copy_paste"]["enabled"] = args.copy_paste == "on"
    for name,value in (("training_config.yaml",arch),("dataset_config.yaml",data)):
        path = output/name
        if path.exists() and yaml.safe_load(path.read_text()) != value:
            raise ValueError("Resume config mismatch: "+name)
        if not path.exists():
            path.write_text(yaml.safe_dump(value,sort_keys=False))
    logger.info(f"Step E.1 | CP={args.copy_paste} | resume={args.resume} | scratch segmentation / frozen pretrained PointNet")
    trainer = Trainer(arch,data,args.dataset,output,logger,args.copy_paste=="on",args.resume)
    if trainer.initial_hash != gate["initial_state_sha256"]:
        raise ValueError("Formal random initialization differs from the verified seed-0 state")
    if trainer.validation_manifest_sha256 != gate["validation_manifest_sha256"]:
        raise ValueError("Formal validation cache files differ from the regression gate")
    metadata = dict(identity=trainer.identity, initial_state_sha256=trainer.initial_hash,
                    scratch_segmentation=True, pointnet_encoder_frozen=True, effective_batch_size=3,
                    copy_paste_effective_probability=arch["copy_paste"]["probability"] if trainer.enabled else 0.,
                    micro_batch_size=3, tie_break="priority then highest raw semantic ID", dropout_policy="legacy_denoising",
                    ordinary_augmentation="C.2 post-regeneration azimuth roll/flip, 50% chance of 10% cell dropout; labels retained",
                    validation="original C.2 caches; 4071 seq08; EMA FP32; ignore0; height correction; KNN OFF",
                    visibility="approximate 2048x64 angular depth buffer, not calibrated 64-beam simulation",
                    gate_sha256=sha256(args.gate), torch=str(torch.__version__), gpu=torch.cuda.get_device_name())
    atomic_json(output/"metadata.json",metadata)
    if not args.resume:
        atomic_json(output/"training_history.json",[])
        atomic_json(output/"results.json",dict(status="running",completed_epoch=-1,best_epoch=None,best_miou=None))
        (output/"comparison.md").write_text("# E.1 Comparison\n\nTraining in progress. OFF/ON and fixed-alpha comparison pending both completed best EMA checkpoints.\n")
    with (output/"used_files.txt").open("a") as file:
        file.write("\nCommand: "+" ".join(sys.argv)+"\n"+json.dumps(metadata,indent=2)+"\n")
        file.write("Estimated epoch/70-epoch time from >=3-update smoke (approximate, validation/IO variance):\n"+
                   json.dumps(gate.get("performance_estimates",{}),indent=2)+"\n")
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("E.1 stopped with an error; existing checkpoints are preserved")
        raise
