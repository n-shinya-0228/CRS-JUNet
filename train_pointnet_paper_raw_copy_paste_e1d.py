"""Gated scratch training: E.1c geometry and E.1 segmentation, paper BEV selection."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib/trainer"))

import torch
import yaml
from loguru import logger
from trainer_PointNet_RawCopyPaste_E1 import Trainer as E1, atomic_json, deterministic_encoder
from lib.utils.raw_point_copy_paste_paper_e1d import augment, tally, epoch_stats
from lib.utils.pointnet_copy_paste_e0 import sha256
from compare_pointnet_paper_e1d import RUN, NAMES, locate, report

CONFIG = Path("config/arch/sjnet_pointnet_local3_raw_cp_e1d_paper3.yaml")
LABELS = Path("config/labels/semantic-kitti.yaml")
FILES = [str(CONFIG), str(LABELS), "lib/utils/raw_point_copy_paste_paper_e1d.py",
         "train_pointnet_paper_raw_copy_paste_e1d.py", "compare_pointnet_paper_e1d.py",
         "verify_pointnet_paper_e1d.py", "tests/test_pointnet_paper_e1d.py"]
PREFLIGHT = RUN.with_name(RUN.name + "_preflight")


def read_config():
    arch = yaml.safe_load(CONFIG.read_text())
    ref = yaml.safe_load((locate(NAMES[2]) / "training_config.yaml").read_text())
    expected = dict(ref, copy_paste=dict(ref["copy_paste"], mode="paper", paper=arch["copy_paste"]["paper"]))
    if arch != expected:
        raise ValueError("Non-selection settings differ from E.1c")
    return arch, yaml.safe_load(LABELS.read_text())


class Trainer(E1):
    def __init__(self, arch, data, output, resume=None):
        super().__init__(arch, data, "SemanticKitti", output, logger, arch["copy_paste"]["enabled"])
        # Workers only read the OFF path. The main process scores candidates on the
        # authoritative CUDA grid; neither a CUDA worker nor a cached patch is used.
        self.dataset.enabled = False
        ref = json.loads((locate(NAMES[2]) / "metadata.json").read_text())["identity"]
        for key in ("data", "raw_train_manifest", "validation_manifest_sha256", "encoder",
                    "db_manifest", "initial_state_sha256"):
            actual = json.loads(json.dumps(self.identity[key])) if key == "data" else self.identity[key]
            if actual != ref[key]:
                raise ValueError("E.1c identity mismatch: " + key)
        for path, value in ref["code"].items():
            if sha256(path) != value:
                raise ValueError("E.1c code mismatch: " + path)
        self.identity["code"].update({p:sha256(p) for p in FILES})
        self.identity["mode"] = "paper"
        if resume:
            self.load(resume)
        self.stats_epoch, self.counts = self.next_epoch, Counter()
        if self.partial:
            for stats in self.partial["meters"]["cp"]:
                tally(self.counts, stats)

    def prepare(self, sample):
        if self.enabled:
            with deterministic_encoder():
                augment(sample, self.dataset.database, self.data["learning_map"], self.arch["copy_paste"], self.device)

    def batch_input(self, batch):
        for sample in batch:
            self.prepare(sample)
        return super().batch_input(batch)

    def update(self, batch):
        out = super().update(batch)
        epoch = batch[0]["epoch"]
        if self.stats_epoch != epoch:
            self.stats_epoch, self.counts = epoch, Counter()
        for sample in batch:
            tally(self.counts, sample["stats"])
        return out

    def save(self, epoch, next_batch=0, partial=None, best=False, elapsed_seconds=None):
        record = dict(epoch=epoch, next_batch=next_batch, **epoch_stats(self.counts))
        atomic_json(self.output / "paper_cp_progress.json", record)
        if partial is None and not best:
            path = self.output / "paper_cp_epoch_history.json"
            history = json.loads(path.read_text()) if path.exists() else {}
            history[str(epoch)] = record
            atomic_json(path, history)
            logger.info(f"Paper CP epoch {epoch}: {record}")
        super().save(epoch, next_batch, partial, best, elapsed_seconds)
        if partial is None and not best and self.output == RUN:
            report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume only this run's latest checkpoint")
    args = parser.parse_args()
    arch, data = read_config()
    gate = json.loads((PREFLIGHT / "verification.json").read_text())
    if gate["status"] != "PASS" or gate["inputs"] != {p:sha256(p) for p in FILES}:
        raise ValueError("Run E.1d preflight before training")
    if args.resume and not (RUN / "latest.path").is_file():
        parser.error("No E.1d checkpoint")
    RUN.mkdir(exist_ok=args.resume, parents=True)
    logger.add(RUN / "train_log.txt", mode="a")
    trainer = Trainer(arch, data, RUN, RUN / "latest.path" if args.resume else None)
    if not args.resume:
        snapshots = RUN / "source"
        for name in FILES:
            dest = snapshots / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
        shutil.copyfile(CONFIG, RUN / "training_config.yaml")
        shutil.copyfile(LABELS, RUN / "dataset_config.yaml")
        atomic_json(RUN / "metadata.json", dict(identity=trainer.identity, scratch=True, frozen_encoder=True,
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), preflight=str(PREFLIGHT),
                    git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
                    placement="Unchanged E1c proposal and height; gate on CUDA BEV, before raw-point append. "
                              "Strict context >0.1; no OBB, plane, depth buffer or minimum cell-count gate."))
        atomic_json(RUN / "results.json", dict(status="running", completed_epoch=-1))
        atomic_json(RUN / "training_history.json", [])
        report()
    logger.info(f"E.1d | PID={os.getpid()} | initialized={trainer.initial_hash} | resume={args.resume}")
    trainer.train()
    report()


if __name__ == "__main__":
    main()
