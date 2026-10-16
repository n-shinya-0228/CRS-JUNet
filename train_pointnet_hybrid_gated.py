#!/usr/bin/env python3
"""Isolated Step C.1 entry point. Existing Step C logs are never reused."""

import argparse
import os
import os.path as osp
from shutil import copyfile
import sys

import torch
import yaml
from loguru import logger

TRAINER_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "lib", "trainer")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)
from trainer_PointNet_Gated import Trainer, validate_config, validate_resume_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--arch_cfg", "-ac", required=True)
    parser.add_argument("--data_cfg", "-dc", default="config/labels/semantic-kitti.yaml")
    parser.add_argument("--log", "-l", required=True)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--pretrained", "-p", help="Compatible gated model weights only")
    weights.add_argument("--resume", "-r", help="Step C.1 latest.path or best_val.path")
    args = parser.parse_args()

    # Check the experiment identity before writing any logs or config snapshots.
    try:
        with open(args.arch_cfg) as f:
            arch = yaml.safe_load(f)
        with open(args.data_cfg) as f:
            data = yaml.safe_load(f)
        validate_config(arch)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume_checkpoint(checkpoint, arch, data)
            del checkpoint
        if args.pretrained and not osp.isfile(args.pretrained):
            raise ValueError(f"Pretrained checkpoint not found: {args.pretrained}")
        if osp.exists(args.log):
            if not args.resume:
                raise ValueError(f"Log directory already exists: {args.log}. Use a new directory.")
            for filename, expected in (("arch_cfg.yaml", arch), ("data_cfg.yaml", data)):
                with open(osp.join(args.log, filename)) as f:
                    if yaml.safe_load(f) != expected:
                        raise ValueError(f"Refusing to reuse a different experiment: {args.log}")
        else:
            os.makedirs(args.log)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    logger.add(osp.join(args.log, "log.txt"))
    logger.info(f"Step C.1 | dataset={args.dataset} | log={args.log}")
    logger.info(f"arch_cfg={args.arch_cfg} | resume={args.resume} | pretrained={args.pretrained}")
    for source, name in ((args.arch_cfg, "arch_cfg.yaml"), (args.data_cfg, "data_cfg.yaml")):
        destination = osp.join(args.log, name)
        if not osp.exists(destination):
            copyfile(source, destination)

    trainer = Trainer(
        arch, data, args.dataset, args.log, logger,
        pretrained=args.pretrained, use_mps=arch["model"]["use_mps"], resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
