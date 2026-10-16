#!/usr/bin/env python3
"""Step C.4: Local3 + a global and spatial residual scale, trained from scratch."""

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
from trainer_PointNet_Gated_StableLR import Trainer, validate_config, validate_resume_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--arch_cfg", "-ac", required=True)
    parser.add_argument("--data_cfg", "-dc", default="config/labels/semantic-kitti.yaml")
    parser.add_argument("--log", "-l", required=True)
    parser.add_argument("--resume", "-r", help="Resume only a matching Step C.4 checkpoint")
    args = parser.parse_args()

    try:
        with open(args.arch_cfg) as f:
            arch = yaml.safe_load(f)
        with open(args.data_cfg) as f:
            data = yaml.safe_load(f)
        validate_config(arch)
        if arch["model"]["name"] != "SJNet3_PointNet_Gated_Local3_SpatialAlpha":
            raise ValueError("This entry point requires the Step C.4 Local3 SpatialAlpha model")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_resume_checkpoint(checkpoint, arch, data)
            del checkpoint
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
    logger.info(f"Step C.4 | dataset={args.dataset} | log={args.log}")
    logger.info(f"arch_cfg={args.arch_cfg} | resume={args.resume} | pretrained=None")
    for source, name in ((args.arch_cfg, "arch_cfg.yaml"), (args.data_cfg, "data_cfg.yaml")):
        destination = osp.join(args.log, name)
        if not osp.exists(destination):
            copyfile(source, destination)
    trainer = Trainer(
        arch, data, args.dataset, args.log, logger,
        pretrained=None, use_mps=arch["model"]["use_mps"], resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
