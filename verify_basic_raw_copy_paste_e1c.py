#!/usr/bin/env python3
"""CPU regressions and real GPU checks, isolated from the formal E.1c run."""

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from train_pointnet_basic_raw_copy_paste_e1c import (
    CONFIG, PREFLIGHT, REFERENCES, CONDITIONS, SOURCE_FILES, Trainer, atomic_json, check_config)
import numpy as np
import torch
import yaml
from loguru import logger
from trainer_PointNet_RawCopyPaste_E1 import deterministic_encoder, state_digest
from lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1 import SemanticKitti as LegacyDataset
from lib.utils.raw_point_copy_paste_e0b import read_train_scan, regenerate_hybrid, map_semantic
from lib.utils.raw_point_copy_paste_e0c import TRAIN_SEQUENCES
from lib.utils.pointnet_copy_paste_e0 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SemanticKitti")
    parser.add_argument("--output", default=PREFLIGHT)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "verification.json").exists():
        raise ValueError("Use a new preflight folder; passed reports are immutable")
    logger.add(output / "verification.log", mode="a")
    historical = json.loads(Path("logs/SJNet3_PointNet_stepE0c_database_training_setup/verification.json").read_text())
    protected, missing = {}, []
    for name, expected in historical["protected_inputs"].items():
        if not Path(name).is_file():
            missing.append(name)
            continue
        if sha256(name) != expected:
            raise ValueError("Historical protected file changed: " + name)
        protected[name] = expected
    # Historical C.2 weights are not used by this scratch run; E.1 results and code are present.
    allowed_missing = {"logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path"}
    if set(missing) - allowed_missing:
        raise ValueError("Required E.1 input missing: " + str(missing))
    for folder in REFERENCES.values():
        for path in Path(folder).rglob("*"):
            if path.is_file():
                protected[str(path)] = sha256(path)
    for name in (*SOURCE_FILES, CONFIG, __file__, "tests/test_basic_raw_copy_paste_e1c.py"):
        protected[str(Path(name))] = sha256(name)
    conditions = {name: False for name in CONDITIONS}
    for pattern in ("test_basic_raw_copy_paste_e1c.py", "test_pointnet_training_e1.py",
                    "test_raw_point_copy_paste_e0b.py", "test_raw_point_copy_paste_e0c.py"):
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"],
                                capture_output=True, text=True)
        (output / (pattern + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("CPU tests failed: " + pattern + "\n" + result.stderr)
        logger.info(f"CPU regression PASS: {pattern}")
    conditions["cpu_tests"] = True
    arch = yaml.safe_load(Path(CONFIG).read_text())
    data_path = "config/labels/semantic-kitti.yaml"
    data = yaml.safe_load(Path(data_path).read_text())
    check_config(arch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required before formal training")
    smoke = output / "smoke"
    smoke.mkdir(exist_ok=True)
    trainer = Trainer(arch, data, args.dataset, smoke, logger)
    assert trainer.initial_hash == historical["initial_state_sha256"]
    initial_hash, validation_hash = trainer.initial_hash, trainer.validation_manifest_sha256
    assert validation_hash == historical["validation_manifest_sha256"], "Validation cache manifest changed"
    legacy_off = LegacyDataset(args.dataset, data, arch["copy_paste"], False)
    basic_off = copy.copy(trainer.dataset); basic_off.enabled = False
    for index in (0, 173, 346):
        a, b = legacy_off[index], basic_off[index]
        for key in a:
            if key == "stats":
                assert {k: v for k, v in a[key].items() if k != "worker_seconds"} == {k: v for k, v in b[key].items() if k != "worker_seconds"}
            elif isinstance(a[key], np.ndarray):
                np.testing.assert_array_equal(a[key], b[key])
            else:
                assert a[key] == b[key]
    conditions["off_regression"] = True
    forced = copy.copy(trainer.dataset); forced.cp = dict(arch["copy_paste"], probability=1.)
    counts, classes, scans, examples = Counter(), Counter(), [], []
    forbidden = ("lib.utils.raw_point_copy_paste_e0b.raw_copy_paste",
                 "lib.utils.raw_point_copy_paste_e0b.TargetScene",
                 "lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1.TargetScene",
                 "lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1.raw_copy_paste",
                 "lib.utils.raw_point_copy_paste_e0b.visibility_filter", "numpy.linalg.lstsq")
    with ExitStack() as stack:
        for name in forbidden:
            stack.enter_context(patch(name, side_effect=AssertionError("Advanced placement invoked: " + name)))
        for i in range(64):
            dataset = trainer.dataset if i < 32 else forced
            index = i * 173 % len(dataset)
            sample = dataset[index]
            original, packed = read_train_scan(dataset.files[index], data["learning_map"], TRAIN_SEQUENCES)
            count = sample["stats"]["successful_objects"]
            assert 0 <= count <= 3 and count == len(sample["placement"])
            assert len(sample["points"]) == len(sample["packed"])
            np.testing.assert_array_equal(sample["points"][:len(original)], original)
            np.testing.assert_array_equal(sample["packed"][:len(packed)], packed)
            offset = len(original)
            for placement in sample["placement"]:
                size = placement["added_points"]
                part = sample["packed"][offset:offset + size]
                assert np.all(map_semantic(part, data["learning_map"]) == placement["learning_id"])
                assert np.all(part >> 16 == placement["synthetic_instance_id"])
                offset += size
            assert offset == len(sample["points"])
            counts[str(count)] += 1
            classes.update(sample["stats"]["class_successes"])
            scans.append(dict(scan_id=sample["scan_id"], stats=sample["stats"], placements=sample["placement"]))
            if count == 3 and len(examples) < 3:
                examples.append(sample)
    assert counts["0"] and counts["3"] and all(classes[str(cid)] for cid in (2, 3, 4, 5, 6))
    conditions["no_advanced_placement"] = conditions["real_scan_counts_0_to_3"] = True
    atomic_json(output / "real_scans.json", dict(histogram=dict(counts), class_successes=dict(classes), scans=scans))
    logger.info(f"Real scans PASS | success histogram={dict(counts)} | classes={dict(classes)}")
    from verify_raw_copy_paste_e0c import canonical_reference, compare
    regenerated = []
    encoder_hash = state_digest(trainer.encoder.state_dict())
    for sample in examples:
        with deterministic_encoder():
            result = regenerate_hybrid(sample["points"], sample["packed"], trainer.encoder,
                                       data["learning_map"], sample["paste_points"])
            independent = canonical_reference(sample["points"], sample["packed"], data["learning_map"], trainer.encoder, trainer.device)
        assert result["point16"].shape == (16, 512, 512)
        assert result["hybrid24"].shape == (24, 512, 512)
        assert torch.isfinite(result["hybrid24"]).all() and result["point16"].abs().sum() > 0
        torch.testing.assert_close(result["hybrid24"][7:23], result["point16"].half().float(), rtol=0, atol=0)
        torch.testing.assert_close(result["hybrid24"][23:24], result["mask"], rtol=0, atol=0)
        regenerated.append(dict(scan_id=sample["scan_id"], differences=compare(result, independent)))
    conditions["whole_cloud_regeneration"] = True
    atomic_json(output / "regeneration.json", regenerated)
    del result, independent
    durations = []
    from trainer_PointNet_SharedSoftAlpha_D5 import nested_digest
    for step in range(2):
        start = time.perf_counter()
        loss, _, _, diagnostics = trainer.update(examples)
        durations.append(time.perf_counter() - start)
        assert diagnostics["input_shape"] == [3, 24, 512, 512]
        assert state_digest(trainer.encoder.state_dict()) == encoder_hash
        logger.info(f"GPU update {step + 1} PASS | loss={loss:.6f} | shape={diagnostics['input_shape']}")
        if step == 0:
            trainer.save(0, 1, dict(meters=dict(cp=[s["stats"] for s in examples])))
            roundtrip = dict(model=state_digest(trainer.model.state_dict()),
                             ema=state_digest(trainer.ema_model.state_dict()),
                             optimizer=nested_digest(trainer.optimizer.state_dict()),
                             scheduler=copy.deepcopy(trainer.scheduler.state_dict()),
                             scaler=copy.deepcopy(trainer.scaler.state_dict()),
                             cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    expected_loss, expected_state = loss, state_digest(trainer.model.state_dict())
    expected_tensors = {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}
    assert expected_state != initial_hash
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    trainer = Trainer(arch, data, args.dataset, smoke, logger, resume=str(smoke / "latest.path"))
    assert trainer.next_epoch == 0 and trainer.next_batch == 1 and trainer.counts["3"] == 3
    assert state_digest(trainer.model.state_dict()) == roundtrip["model"]
    assert state_digest(trainer.ema_model.state_dict()) == roundtrip["ema"]
    assert nested_digest(trainer.optimizer.state_dict()) == roundtrip["optimizer"]
    assert trainer.scheduler.state_dict() == roundtrip["scheduler"]
    assert trainer.scaler.state_dict() == roundtrip["scaler"]
    assert torch.equal(torch.get_rng_state(), roundtrip["cpu_rng"])
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), roundtrip["cuda_rng"]))
    resumed_loss, _, _, _ = trainer.update(examples)
    assert abs(resumed_loss - expected_loss) < 1e-6
    post_step_differences = {key: float((value.detach().cpu().float() - expected_tensors[key].float()).abs().max())
                             for key, value in trainer.model.state_dict().items()}
    # Preserve E.1's existing CUDA training recipe; only encoder projection is explicitly deterministic.
    resume_report = dict(checkpoint_model_ema_optimizer_scheduler_scaler_rng_exact=True,
                         next_update_loss_difference=abs(resumed_loss - expected_loss),
                         next_update_bitwise_equal=state_digest(trainer.model.state_dict()) == expected_state,
                         next_update_max_absolute_parameter_difference=max(post_step_differences.values()),
                         next_update_parameter_differences={k: v for k, v in post_step_differences.items() if v},
                         note="Checkpoint restoration is exact; training CUDA operations retain the original E.1 determinism settings")
    atomic_json(output / "resume_diagnostics.json", resume_report)
    logger.info(f"Resume exact-state PASS | next-update max parameter difference={max(post_step_differences.values()):.3e}")
    assert state_digest(trainer.encoder.state_dict()) == encoder_hash
    conditions["batch3_forward_backward_frozen_encoder"] = conditions["resume"] = True
    for name, checksum in protected.items():
        assert sha256(name) == checksum, "Protected file changed: " + name
    conditions["protected_inputs_unchanged"] = True
    atomic_json(output / "verification.json", dict(verdict="PASS", conditions=conditions,
                protected_inputs=protected, config_sha256=sha256(CONFIG), data_sha256=sha256(data_path),
                initial_state_sha256=initial_hash, validation_manifest_sha256=validation_hash,
                historical_checkpoint_not_available=missing,
                note="Historical C.2 checkpoint not required for scratch training; all available historical hashes and E.1a/E.1b files verified",
                real_success_histogram=dict(counts), class_successes=dict(classes),
                update_seconds=durations, gpu=torch.cuda.get_device_name(),
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
    logger.info(f"E.1c preflight PASS: {output / 'verification.json'}")


if __name__ == "__main__":
    main()
