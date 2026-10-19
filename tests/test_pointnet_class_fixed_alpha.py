import copy
import csv
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import evaluate_pointnet_class_fixed_alpha as evaluation
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    ALPHA_TABLE, CLASS_NAMES, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps,
)
from lib.models.SJNet3_PointNet_Gated_Local3_FixedAlpha import SJNet3_PointNet_Gated_Local3_FixedAlpha
from lib.utils.pointnet_class_alpha_reporting import alpha_rows, best_modes, export_results
from lib.utils.pointnet_class_alpha_statistics import AlphaStatistics, MODES, confusion_metrics

torch.set_num_threads(2)


class ClassAlphaTests(unittest.TestCase):
    def test_all_twenty_learning_ids_and_unobserved_fallback(self):
        logits = torch.full((1, 20, 1, 20), -1000.)
        for cid in range(20):
            logits[0, cid, 0, cid] = 1000.
        mask = torch.ones(1, 1, 1, 20)
        hard, soft = build_alpha_maps(logits, mask)
        expected = torch.tensor(ALPHA_TABLE).view(1, 1, 1, 20)
        torch.testing.assert_close(hard, expected, atol=0, rtol=0)
        torch.testing.assert_close(soft, expected, atol=0, rtol=0)
        mask[..., ::2] = 0
        hard, soft = build_alpha_maps(logits, mask)
        for alpha in (hard, soft):
            torch.testing.assert_close(alpha[..., ::2], torch.full_like(alpha[..., ::2], 1.2), atol=0, rtol=0)
        self.assertEqual(set(inspect.signature(build_alpha_maps).parameters), {"stage1_logits", "observation_mask"})
        self.assertEqual(len(ALPHA_TABLE), 20)

    def test_person_truck_mixture_uniform_and_raw_logits_not_height_correction(self):
        for cid, expected in ((6, .6), (4, 1.6)):
            logits = torch.full((1, 20, 1, 1), -1000.)
            logits[:, cid] = 1000
            for alpha in build_alpha_maps(logits, torch.ones(1, 1, 1, 1)):
                self.assertAlmostEqual(alpha.item(), expected, places=6)
        logits = torch.full((1, 20, 1, 1), -1000.)
        logits[:, 4] = logits[:, 6] = 0
        self.assertAlmostEqual(build_alpha_maps(logits, torch.ones(1, 1, 1, 1))[1].item(), 1.1, places=6)
        uniform = build_alpha_maps(torch.zeros_like(logits), torch.ones(1, 1, 1, 1))[1]
        self.assertAlmostEqual(uniform.item(), sum(ALPHA_TABLE)/20, places=6)
        logits.fill_(-1000.)
        logits[:, 7] = 1000
        hard, _ = build_alpha_maps(logits, torch.ones(1, 1, 1, 1))
        self.assertAlmostEqual(hard.item(), .8, places=6)
        features = torch.zeros(1, 23, 1, 1)
        features[:, 0] = -.4
        self.assertEqual(evaluation.d1.original_predictions(logits, features).item(), 2)
        with self.assertRaises(ValueError):
            build_alpha_maps(logits.fill_(float('nan')), torch.ones(1, 1, 1, 1))

    def test_strict_state_constant_maps_all_outputs_masks_and_no_stale_state(self):
        options = dict(nclasses=20, base_ch=8, aspp_out=16, drop=0)
        reference = SJNet3_PointNet_Gated_Local3_FixedAlpha(**options, alpha=1.2).eval().requires_grad_(False)
        model = SJNet3_PointNet_Gated_Local3_ClassAlpha(**options).eval().requires_grad_(False)
        model.load_state_dict(reference.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in reference.parameters()))
        self.assertFalse(any('alpha' in name for name in model.state_dict()))
        digest = evaluation.d1.state_digest(model.state_dict())
        x = torch.randn(1, 24, 32, 320)
        for mask in (0., 1.):
            x[:, 23] = mask
            with torch.no_grad():
                expected = reference(x)
                actual = model(x, alpha_map=torch.full((1, 1, 32, 320), 1.2))
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
                changed = model(x, alpha_map=torch.full((1, 1, 32, 320), .6))
                self.assertFalse(torch.equal(changed['logits'], expected['logits']))
                restored = model(x)
                for name in expected:
                    torch.testing.assert_close(restored[name], expected[name], atol=0, rtol=0)
            self.assertIsNone(model.point_context._alpha_map)
        self.assertEqual(evaluation.d1.state_digest(model.state_dict()), digest)
        def fail(*args):
            raise RuntimeError('simulated forward failure')
        handle = model.point_context.register_forward_hook(fail)
        with self.assertRaises(RuntimeError):
            model(x, alpha_map=torch.ones(1, 1, 32, 320))
        handle.remove()
        self.assertIsNone(model.point_context._alpha_map)
        with self.assertRaises(ValueError):
            model(x, alpha_map=torch.ones(1, 16, 32, 320))
        with self.assertRaises(ValueError):
            model(x, alpha_map=torch.ones(1, 1, 32, 320, requires_grad=True))

    def test_table_matches_saved_d1b_and_rejects_mapping_changes(self):
        source = evaluation.ROOT/'logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity'
        classes = dict(enumerate(CLASS_NAMES))
        evaluation.validate_alpha_table(source, classes)
        with self.assertRaises(ValueError):
            evaluation.validate_alpha_table(source, {**classes, 0:'wrong'})
        with tempfile.TemporaryDirectory() as tmp:
            with (source/'best_alpha_by_class.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['best_alphas'] = '1.6'
            evaluation.d1.write_csv(Path(tmp)/'best_alpha_by_class.csv', list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, 'car'):
                evaluation.validate_alpha_table(Path(tmp), classes)

    def test_streaming_statistics_moments_quantiles_gt_diagnostics_and_resume(self):
        hard, soft = np.array([.6, 1.6, 1.2]), np.array([1.1, 1.3, 1.2])
        targets, raw = np.array([6, 4, 0]), np.array([4, 4, 0])
        finals = [raw, np.array([6, 4, 0]), np.array([6, 6, 0])]
        stats = AlphaStatistics()
        stats.update(hard, soft, targets, raw, finals, unobserved=2)
        self.assertEqual(stats.count[0], 3)
        self.assertAlmostEqual(stats.distribution(0)['mean'], hard.mean())
        self.assertAlmostEqual(stats.distribution(0)['std'], hard.std())
        self.assertEqual(stats.distribution(0, scope='unobserved')['std'], 0)
        self.assertAlmostEqual(stats.distribution(1, scope='all')['mean'], (soft.sum()+2.4)/5)
        self.assertEqual(stats.distribution(0)['p50'], 1.2)
        self.assertLessEqual(abs(stats.distribution(1)['p50']-1.2), .0001)
        diagnostic = stats.diagnostics()
        self.assertEqual(diagnostic['stage1_wrong_fraction'], .5)
        self.assertEqual(diagnostic['by_mode']['D2_Hard']['stage1_wrong']['final_correct_fraction'], 1)
        self.assertEqual(diagnostic['by_mode']['D2_Soft']['stage1_correct']['final_correct_fraction'], 0)
        restored = AlphaStatistics.restore(stats.snapshot())
        self.assertEqual(restored.snapshot(), stats.snapshot())
        merged = AlphaStatistics()
        merged.merge(restored)
        self.assertEqual(merged.snapshot(), stats.snapshot())
        bad = copy.deepcopy(stats.snapshot())
        bad['count'][0] += 1
        with self.assertRaises(ValueError):
            AlphaStatistics.restore(bad)
        self.assertEqual(len(alpha_rows(stats)), 90)

    def test_three_mode_partial_batch_is_not_committed_and_resume_no_duplicates(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
                self.point_context = type('Context', (), {'_alpha_map':None})()
                self.calls, self.fail = 0, True
            def set_alpha(self, alpha):
                self.fixed_alpha = alpha
            def forward(self, x, alpha_map=None):
                self.calls += 1
                if self.fail and self.calls == 5:
                    raise RuntimeError('simulated mid-batch interruption')
                logits = torch.zeros(x.shape[0], 20, 2, 2)
                logits[:, 1] = self.weight
                return {'logits':logits}
        class Loader:
            batch_size = 1
            def __len__(self):
                return 2
            def __iter__(self):
                for index in range(2):
                    batch = [None]*16
                    batch[:3] = [torch.zeros(1, 23, 2, 2), torch.ones(1, 1, 2, 2), torch.ones(1, 2, 2).long()]
                    batch[4:6] = [['08'], [f'{index:06d}.label']]
                    batch[14] = torch.zeros(1, 1, 2, 2)
                    yield batch
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): pass
            def synchronize(self): pass
            def elapsed_time(self, other): return 5.
        model = Model().eval()
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.cuda, 'Event', Event), \
                patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                patch.object(torch.cuda, 'max_memory_allocated', return_value=2**30):
            args = dict(model=model, loader=Loader(), device=torch.device('cpu'), classes=dict(enumerate(CLASS_NAMES)),
                scan_ids=['08/000000.label','08/000001.label'], output=Path(tmp), metadata_id='same', report_every=1)
            with self.assertRaises(RuntimeError):
                evaluation.evaluate_three_modes(**args)
            saved = json.loads((Path(tmp)/'progress_d2.json').read_text())
            self.assertEqual(saved['next_batch'], 1)
            for mode in MODES:
                self.assertEqual(sum(map(sum, saved['confusion_matrices'][mode])), 4)
            model.fail = False
            run, stats = evaluation.evaluate_three_modes(**args)
            for mode, result in run['results'].items():
                self.assertEqual(result['scans'], 2)
                self.assertEqual(sum(map(sum, result['confusion_matrix'])), 8)
                self.assertAlmostEqual(run['forward_seconds'][mode], .01)
            self.assertEqual(stats.count[0], 8)
            with self.assertRaisesRegex(ValueError, 'identity'):
                evaluation.evaluate_three_modes(**dict(args, metadata_id='different'))

    def test_exports_all_classes_three_modes_ties_na_and_four_plots(self):
        reference = dict(miou=.5, per_class_iou={n:.5 if n!='motorcyclist' else 0. for n in CLASS_NAMES[1:]})
        results = {mode:dict(miou=.5+i*.001, accuracy=.89,
                    per_class_iou={n:score+(i*.001 if n!='motorcyclist' else 0.) for n, score in reference['per_class_iou'].items()})
                   for i, mode in enumerate(MODES)}
        run = dict(status='complete', metadata={'checkpoint':'C.2/best_val.path'}, reference=reference,
                   baseline_passed=True, baseline_reproduction={'miou':.5}, results=results)
        stats = AlphaStatistics()
        stats.update(np.array([.6,1.6]), np.array([1.1,1.3]), np.array([6,4]), np.array([6,4]),
                     [np.array([6,4])]*3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            np.savez_compressed(p/'alpha_example_000000.npz', scan_id='08/000000.label',
                                mask=np.array([[1,0],[1,0]]), hard=np.ones((2,2)), soft=np.ones((2,2)))
            export_results(p,run,stats)
            with (p/'per_class_iou.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 19)
            self.assertEqual(next(r for r in rows if r['Class']=='motorcyclist')['Best'], 'N/A (all zero)')
            with (p/'alpha_statistics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 90)
            for filename in ('class_iou_comparison.png','class_delta_comparison.png','alpha_distribution.png','alpha_map_examples.png'):
                self.assertGreater((p/filename).stat().st_size,1000)
        self.assertIn('tie', best_modes([.2,.2,.1]))
        self.assertEqual(confusion_metrics([[0]*20 for _ in range(20)])['miou'],0)


if __name__ == '__main__':
    unittest.main()
