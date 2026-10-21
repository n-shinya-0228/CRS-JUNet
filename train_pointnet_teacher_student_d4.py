#!/usr/bin/env python3
"""Run a separate D.4 experiment, gated by GPU smoke and full D.2b reproduction."""

import argparse
import copy
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stepd4_mpl")

import torch
import yaml

from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import ALPHA_TABLE, CLASS_NAMES

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/"logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4"
DEFAULT_ARCH = ROOT/"config/arch/sjnet_pointnet_teacher_student_d4_cp0.yaml"
TEACHER = ROOT/"logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arch_cfg", type=Path)
    parser.add_argument("--data_cfg", type=Path)
    parser.add_argument("--log", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=TEACHER)
    parser.add_argument("--resume", type=Path, help="Native D.4 latest.path; restores the complete epoch boundary")
    parser.add_argument("--prepare-only", action="store_true", help="GPU smoke + full epoch0 gate, without formal updates")
    parser.add_argument("--evaluate", action="store_true", help="Re-evaluate the existing D.4 best checkpoint")
    return parser.parse_args(argv)


def check_output(args):
    output = args.log.resolve()
    if output.parent != (ROOT/"logs").resolve() or not output.name.startswith("SJNet3_PointNet_stepD4_"):
        raise ValueError("Use a new logs/SJNet3_PointNet_stepD4_* directory; older experiments are read-only")
    if args.resume or args.evaluate:
        checkpoint = args.resume or output/"latest.path"
        if checkpoint.resolve().parent != output or not checkpoint.is_file():
            raise ValueError("Resume/evaluation requires an existing checkpoint in the same D.4 output directory")
        if args.prepare_only:
            raise ValueError("--prepare-only is for a new experiment, not resume/evaluation")
    elif output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use --resume with its latest.path")
    return output


def load_trainer():
    path = ROOT/"lib/trainer/trainer_PointNet_TeacherStudent_D4.py"
    spec = importlib.util.spec_from_file_location("pointnet_d4_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_attempt(module, arch, data, args, output, logger):
    disposable = None
    try:
        disposable = module.Trainer(arch, data, args.dataset, output, logger, args.teacher_checkpoint)
        return disposable.smoke()
    finally:
        del disposable
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    args = arguments(argv)
    output = check_output(args)
    continuing = bool(args.resume or args.evaluate)
    arch_path = args.arch_cfg or (output/"arch_cfg.yaml" if continuing else DEFAULT_ARCH)
    data_path = args.data_cfg or (output/"data_cfg.yaml" if continuing else ROOT/"config/labels/semantic-kitti.yaml")
    arch = yaml.safe_load(arch_path.read_text())
    data = yaml.safe_load(data_path.read_text())
    module = load_trainer()
    module.validate_config(arch)
    if not continuing and arch["train"]["micro_batch_size"] != 3:
        raise ValueError("A new experiment must first try FP32 batch 3; fallback is allowed only after real OOM")
    output.mkdir(parents=True, exist_ok=continuing)
    logger = logging.getLogger("pointnet_d4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output/"train_log.txt")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    status = module.d1.atomic_json
    try:
        if not continuing:
            status(output/"status.json", dict(status="gpu_preflight", formal_student_updates=0))
            attempts = []
            for micro in (3, 2, 1):
                candidate = copy.deepcopy(arch)
                candidate["train"]["micro_batch_size"] = micro
                logger.info(f"Disposable FP32 smoke: microbatch={micro}, effective batch=3; weights will be discarded")
                try:
                    result = smoke_attempt(module, candidate, data, args, output/"preflight"/f"bs{micro}", logger)
                except torch.cuda.OutOfMemoryError as error:
                    attempts.append(dict(status="CUDA_OOM", micro_batch_size=micro, error=str(error)))
                    status(output/"smoke_report.json", dict(attempts=attempts))
                    logger.warning(f"FP32 microbatch {micro} OOM; trying the next smaller microbatch only")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                attempts.append(result)
                arch = candidate
                status(output/"smoke_report.json", dict(status="PASS", attempts=attempts, disposable_weights_discarded=True))
                break
            else:
                raise RuntimeError("FP32 microbatch 1 also OOM; no formal training was started")
        trainer = module.Trainer(arch, data, args.dataset, output, logger, args.teacher_checkpoint,
                                 resume=args.resume or (output/"latest.path" if args.evaluate else None))
        if not continuing:
            (output/"arch_cfg.yaml").write_text(yaml.safe_dump(arch, sort_keys=False))
            (output/"data_cfg.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            status(output/"metadata.json", dict(metadata_id=trainer.metadata_id, **trainer.metadata))
            status(output/"class_alpha_table.json", dict(temperature=1., unobserved_alpha=1.2,
                classes=[dict(learning_id=i, name=CLASS_NAMES[i], alpha=a) for i, a in enumerate(ALPHA_TABLE)]))
            status(output/"validation_manifest.json", dict(entries=trainer.validation_entries, scan_ids=trainer.scan_ids))
            status(output/"status.json", dict(status="epoch0_reproduction", formal_student_updates=0))
            trainer.prepare()
            status(output/"status.json", dict(status="prepared", epoch=0, formal_student_updates=0,
                reproduction_gate_passed=True, epoch0_miou=trainer.best_val_iou))
        if args.prepare_only:
            logger.info("Preparation complete. Formal Student has zero optimizer updates. Resume latest.path to train.")
            return
        if not args.evaluate:
            logger.info(f"Formal D.4 training: next_epoch={trainer.next_epoch}, max=20, FP32, CP=0, Teacher frozen")
            status(output/"status.json", dict(status="training", next_epoch=trainer.next_epoch,
                best_epoch=trainer.best_val_epoch, best_miou=trainer.best_val_iou))
            trainer.train()
        status(output/"status.json", dict(status="final_diagnostics", best_epoch=trainer.best_val_epoch,
            best_miou=trainer.best_val_iou))
        result = trainer.finalize()
        logger.info(f"Complete: best_epoch={result['best_epoch']} mIoU={result['best_miou']:.12f}")
    except BaseException as error:
        logger.exception("D.4 stopped; no successful completion is claimed")
        status(output/"error.json", dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        status(output/"status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            error=str(error), resumable_checkpoint=str(output/"latest.path") if (output/"latest.path").is_file() else None))
        raise


if __name__ == "__main__":
    main()
