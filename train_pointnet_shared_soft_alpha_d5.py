#!/usr/bin/env python3
"""D.5 scratch shared-weight two-pass training; no pretrained formal initialization."""

import argparse
import copy
from datetime import datetime
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stepd5_mpl")

import torch
import yaml

from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    ALPHA_TABLE, CLASS_NAMES, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps,
)
from lib.models.SJNet3_PointNet_Gated_Local3_SharedSoftAlpha import SJNet3_PointNet_Gated_Local3_SharedSoftAlpha

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/"logs/SJNet3_PointNet_stepD5_shared_soft_alpha_scratch_cp0"
DEFAULT_ARCH = ROOT/"config/arch/sjnet_pointnet_shared_soft_alpha_d5_cp0.yaml"
C2_CHECKPOINT = ROOT/"logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arch_cfg", type=Path)
    parser.add_argument("--data_cfg", type=Path)
    parser.add_argument("--log", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args(argv)


def check_output(args):
    output = args.log.resolve()
    if output.parent != (ROOT/"logs").resolve() or not output.name.startswith("SJNet3_PointNet_stepD5_"):
        raise ValueError("Use a new logs/SJNet3_PointNet_stepD5_* directory; old experiments are read-only")
    if args.resume or args.evaluate:
        checkpoint = args.resume or output/"latest.path"
        if checkpoint.resolve().parent != output or not checkpoint.is_file() or args.prepare_only:
            raise ValueError("Resume/evaluation requires a native checkpoint in this D.5 directory")
    elif output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; resume its latest.path instead")
    return output


def load_trainer():
    path = ROOT/"lib/trainer/trainer_PointNet_SharedSoftAlpha_D5.py"
    spec = importlib.util.spec_from_file_location("pointnet_shared_soft_alpha_d5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_logging(output):
    logger = logging.getLogger("pointnet_d5")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    formatter.converter = lambda stamp:datetime.fromtimestamp(stamp, ZoneInfo("Asia/Tokyo")).timetuple()
    for handler in (logging.StreamHandler(), logging.FileHandler(output/"train_log.txt", mode="a")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logging.captureWarnings(True)
    warnings = logging.getLogger("py.warnings")
    warnings.handlers = logger.handlers[:]
    warnings.propagate = False
    warnings.setLevel(logging.WARNING)
    return logger


@torch.no_grad()
def c2_compatibility_regression(module, arch, data, dataset, logger):
    """The only non-native weight loading path: disposable regression models."""
    device = module.configure_cuda()
    parser = module.Parser(root=dataset, data_cfg=data, arch_cfg=arch, gt=True, shuffle_train=False)
    batch = next(iter(parser.validloader))
    x = torch.cat((batch[0], batch[1].float()), dim=1).to(device)
    saved = torch.load(C2_CHECKPOINT, map_location="cpu", weights_only=False)
    state, source = module.d1.select_evaluation_state(saved, saved["arch_cfg"], data)
    reference_info = json.loads((ROOT/"logs/SJNet3_PointNet_stepD2_class_fixed_alpha/results.json").read_text())["metadata"]
    if (module.d1.file_sha256(C2_CHECKPOINT) != reference_info["checkpoint_sha256"]
            or module.d1.state_digest(state) != reference_info["model_state_sha256"]):
        raise ValueError("Regression requires the unchanged C.2 best EMA")
    options = dict(in_channels=24, nclasses=20, drop=arch["model"]["dropout"])
    reference = SJNet3_PointNet_Gated_Local3_ClassAlpha(**options).to(device).eval()
    candidate = SJNet3_PointNet_Gated_Local3_SharedSoftAlpha(**options).to(device).eval()
    reference.load_state_dict(state, strict=True)
    candidate.load_state_dict(state, strict=True)
    differences = {}
    first = reference(x)
    actual = candidate.forward_with_alpha(x)
    differences["global12"] = {}
    for name in first:
        torch.testing.assert_close(actual[name], first[name], atol=0, rtol=0)
        differences["global12"][name] = (actual[name]-first[name]).abs().max().item()
    _, soft = build_alpha_maps(first["logits"], x[:, 23:24])
    del first, actual
    expected = reference(x, alpha_map=soft)
    actual = candidate(x)
    differences["D2_soft"] = {}
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        differences["D2_soft"][name] = (actual[name]-expected[name]).abs().max().item()
    logger.info("C.2 global alpha=1.2 / D.2b shared Soft regression PASS: all five outputs exactly match; pretrained regression weights will be discarded")
    return dict(status="PASS", five_output_max_abs_difference=differences, source=source,
                checkpoint_sha256=reference_info["checkpoint_sha256"], pretrained_state_sha256=reference_info["model_state_sha256"],
                regression_weights_discarded=True, formal_pretrained_initialization=False)


def smoke_attempt(module, arch, data, args, output, logger):
    trainer = None
    try:
        trainer = module.Trainer(arch, data, args.dataset, output, logger, disposable=True)
        result = trainer.smoke()
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    resumed = None
    try:
        resumed = module.Trainer(arch, data, args.dataset, output, logger,
                                 resume=result["checkpoint"], disposable=True)
        result["resume"] = resumed.resume_smoke(result["checkpoint"])
        return result
    finally:
        del resumed
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    args = arguments(argv)
    output = check_output(args)
    continuing = bool(args.resume or args.evaluate)
    arch = yaml.safe_load((args.arch_cfg or (output/"arch_cfg.yaml" if continuing else DEFAULT_ARCH)).read_text())
    data = yaml.safe_load((args.data_cfg or (output/"data_cfg.yaml" if continuing else ROOT/"config/labels/semantic-kitti.yaml")).read_text())
    module = load_trainer()
    module.validate_config(arch)
    if not continuing and arch["train"]["micro_batch_size"] != 3:
        raise ValueError("A new D.5 run must first try native batch 3")
    output.mkdir(parents=True, exist_ok=continuing)
    logger = configure_logging(output)
    status = module.d1.atomic_json
    try:
        if not continuing:
            logger.info("Step D.5: scratch shared-weight two-pass; no Teacher, CP=0; C.2 learning conditions retained")
            logger.info("Stage 1/alpha FP32 eval/no_grad; Stage 2 training BF16 + GradScaler (C.2); validation both FP32")
            status(output/"status.json", dict(status="preflight", formal_optimizer_updates=0))
            regression = c2_compatibility_regression(module, arch, data, args.dataset, logger)
            status(output/"compatibility_regression.json", regression)
            gc.collect()
            torch.cuda.empty_cache()
            attempts = []
            for micro in (3, 2, 1):
                candidate = copy.deepcopy(arch)
                candidate["train"]["micro_batch_size"] = micro
                logger.info(f"Disposable scratch GPU smoke: microbatch={micro}, effective batch=3; no formal training updates")
                try:
                    result = smoke_attempt(module, candidate, data, args, output/"preflight"/f"bs{micro}", logger)
                except torch.cuda.OutOfMemoryError as error:
                    attempts.append(dict(status="CUDA_OOM", micro_batch_size=micro, error=str(error)))
                    status(output/"smoke_report.json", dict(attempts=attempts))
                    logger.warning(f"CUDA OOM for microbatch {micro}; using sample-weighted accumulation with the next smaller microbatch")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                attempts.append(result)
                arch = candidate
                status(output/"smoke_report.json", dict(status="PASS", attempts=attempts, all_preflight_weights_discarded=True))
                logger.info(f"GPU smoke/resume PASS | updates=3+1 disposable | peak allocated/reserved GiB: {result['peak_allocated_GiB']:.3f}/{result['peak_reserved_GiB']:.3f}")
                break
            else:
                raise RuntimeError("All microbatch smoke tests OOM; formal training must not start")
        trainer = module.Trainer(arch, data, args.dataset, output, logger,
                                 resume=args.resume or (output/"latest.path" if args.evaluate else None))
        if not continuing:
            (output/"arch_cfg.yaml").write_text(yaml.safe_dump(arch, sort_keys=False))
            (output/"data_cfg.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            status(output/"class_alpha_table.json", dict(temperature=1., unobserved=1.2,
                classes=[dict(learning_id=i, name=CLASS_NAMES[i], alpha=a) for i, a in enumerate(ALPHA_TABLE)]))
            status(output/"metadata.json", dict(metadata_id=trainer.metadata_id, **trainer.metadata))
            status(output/"validation_manifest.json", dict(entries=trainer.validation_entries, scan_ids=trainer.scan_ids))
            trainer.prepare()
            status(output/"status.json", dict(status="prepared", next_epoch=0, formal_optimizer_updates=0,
                initialization_sha256=trainer.initial_hash))
        if args.prepare_only:
            logger.info("Preparation complete; formal model is random and has zero optimizer updates")
            return
        if not args.evaluate:
            logger.info(f"Formal scratch training starts/resumes at epoch {trainer.next_epoch}; effective batches/epoch={len(trainer.parser.train_dataset)//3}; microbatch={arch['train']['micro_batch_size']}; CP=0")
            logger.info(trainer.metadata["accumulation_caveat"])
            status(output/"status.json", dict(status="training", next_epoch=trainer.next_epoch,
                best_epoch=trainer.best_val_epoch, best_miou=trainer.best_val_iou))
            trainer.train()
        status(output/"status.json", dict(status="final_evaluation", best_epoch=trainer.best_val_epoch, best_miou=trainer.best_val_iou))
        trainer.finalize()
    except BaseException as error:
        logger.exception("D.5 stopped; no successful completion is claimed")
        status(output/"error.json", dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        status(output/"status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            error=str(error), latest_checkpoint_exists=(output/"latest.path").is_file()))
        raise


if __name__ == "__main__":
    main()
