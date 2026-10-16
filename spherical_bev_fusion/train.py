"""Run with python -m spherical_bev_fusion.train."""
import argparse
import logging
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SemanticKitti')
    parser.add_argument('--arch_cfg', default=str(here / 'config.yaml'))
    parser.add_argument('--data_cfg', default=str(here / 'labels.yaml'))
    parser.add_argument('--log', default=str(here / 'runs' / 'fusion128'))
    parser.add_argument('--resume')
    parser.add_argument('--pretrained')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    # Check the dataset before expensive imports or creating a run directory.
    if not (Path(args.dataset) / 'sequences').is_dir():
        parser.error(f'SemanticKITTI dataset missing: {Path(args.dataset).resolve() / "sequences"}')
    import json
    import yaml
    from .trainer import Trainer
    with open(args.arch_cfg, encoding='utf-8') as stream:
        arch = yaml.safe_load(stream)
    with open(args.data_cfg, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    log = Path(args.log)
    if log.exists() and any(log.iterdir()) and not (args.resume or args.evaluate):
        parser.error(f'Run directory is not empty: {log}; choose another --log or --resume')
    if args.evaluate and not (args.pretrained or args.resume):
        parser.error('--evaluate requires --pretrained or --resume')
    log.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log / 'log.txt', level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger('fusion128')
    for src, name in ((args.arch_cfg, 'arch_cfg.yaml'), (args.data_cfg, 'data_cfg.yaml')):
        if Path(src).resolve() != (log / name).resolve():
            shutil.copyfile(src, log / name)
    trainer = Trainer(arch, data, args.dataset, str(log), logger,
                      pretrained=args.pretrained, resume=args.resume)
    if args.evaluate:
        try:
            metrics = trainer._run_epoch(trainer.parser.validloader, training=False)
            with open(log / 'evaluation.json', 'w', encoding='utf-8') as stream:
                json.dump(metrics, stream, indent=2)
            print(json.dumps(metrics, ensure_ascii=False))
        finally:
            trainer.writer.close()
    else:
        trainer.train()
        print((log / 'results.json').read_text())


if __name__ == '__main__':
    main()
