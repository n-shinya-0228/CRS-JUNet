"""One real training batch only. Never starts an epoch or uses a checkpoint."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from unittest.mock import patch


def forbid_copy_paste(*args, **kwargs):
    raise AssertionError('Copy-Paste was called while disabled')


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--arch_cfg', default=str(here / 'config_nocopypaste.yaml'))
    parser.add_argument('--data_cfg', default=str(here / 'labels.yaml'))
    parser.add_argument('--log', default=str(here / 'runs' / 'SphericalBEVFusion_NoCopyPaste_bs2'))
    args = parser.parse_args()
    destination = Path(args.log).resolve()
    if here not in destination.parents:
        parser.error('--log must be inside spherical_bev_fusion')
    if destination.exists():
        parser.error('Experiment directory already exists; choose a new unused path')
    if not (Path(args.dataset) / 'sequences').is_dir():
        parser.error('SemanticKITTI sequences directory missing; run on the data host')

    import numpy as np
    import torch
    import yaml
    from .dataset import DualProjectionDataset
    from .trainer import Trainer
    arch = yaml.safe_load(Path(args.arch_cfg).read_text())
    data = yaml.safe_load(Path(args.data_cfg).read_text())
    baseline = yaml.safe_load((here / 'config.yaml').read_text())
    expected = dict(baseline['dataset'], copy_paste=False)
    assert arch == dict(baseline, dataset=expected), 'Unexpected changes beyond Copy-Paste'
    assert arch['train']['batch_size'] == 2 and arch['train']['max_epochs'] == 100
    assert arch['train']['lr'] == 0.001 and arch['train']['workers'] == 4
    assert torch.get_default_dtype() == torch.float32

    # Diagnostic artifacts have their own unique directory, never the training run.
    checks = here / 'preflight_runs'
    checks.mkdir(exist_ok=True)
    diagnostic = Path(tempfile.mkdtemp(prefix='nocopypaste_', dir=checks))
    trainer = Trainer(arch, data, args.dataset, str(diagnostic), logging.getLogger('preflight'))
    try:
        assert trainer.start_epoch == 0 and trainer.resume is None and trainer.pretrained is None
        assert trainer.parser.train_dataset.copy_paste is False
        assert trainer.parser.trainloader.num_workers == 4
        # Force the original 50% gate open: the disabled path must still not call Copy-Paste.
        with patch.object(DualProjectionDataset, 'apply_copy_paste', forbid_copy_paste):
            with patch('torch.rand', return_value=torch.tensor(1.0)):
                trainer.parser.train_dataset[0]
            batch = trainer._move_batch(next(iter(trainer.parser.trainloader)))
        assert tuple(batch['bev'].shape) == (2, 7, 512, 512)
        assert tuple(batch['sphere'].shape) == (2, 6, 64, 512)
        assert batch['bev'].dtype == batch['sphere'].dtype == torch.float32
        assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        outputs, loss, _, _ = trainer._forward_loss(batch)
        assert outputs['bev_logits'].shape == (2, trainer.nclasses, 512, 512)
        assert torch.isfinite(loss)
        loss.backward()
        for branch in (trainer.model.bev_encoder, trainer.model.sphere_encoder):
            grads = [p.grad for p in branch.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert not destination.exists(), 'Preflight must not create the training run'
        result = dict(status='passed', dataset=str(Path(args.dataset).resolve()),
                      device=str(trainer.device), copy_paste_called=False,
                      bev_shape=list(batch['bev'].shape), sphere_shape=list(batch['sphere'].shape),
                      precision='float32', forward_loss_backward='passed',
                      full_training_started=False, checkpoint_saved=False,
                      planned_training_output=str(destination),
                      model_sha256=hashlib.sha256((here / 'model.py').read_bytes()).hexdigest())
        (diagnostic / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        print(f'Preflight result: {diagnostic / "result.json"}')
    finally:
        trainer.writer.close()


if __name__ == '__main__':
    main()
