"""Regression checks: ground-truth zero is ignored, prediction zero is an error."""
import unittest
import torch
from .trainer import Trainer


class MetricsTests(unittest.TestCase):
    def metrics(self, prediction, target):
        trainer = Trainer.__new__(Trainer)
        trainer.nclasses = 3
        confusion = torch.zeros(3, 3, dtype=torch.int64)
        trainer._update_confusion(confusion, torch.tensor(prediction), torch.tensor(target))
        before = confusion.clone()
        result = trainer._metrics(confusion)
        torch.testing.assert_close(confusion, before)
        return result

    def test_prediction_zero_is_false_negative(self):
        result = self.metrics([1, 0, 2, 0], [1, 1, 2, 2])
        self.assertEqual(result['class_iou'], [0.5, 0.5])
        self.assertEqual(result['miou'], 0.5)
        self.assertEqual(result['accuracy'], 0.5)

    def test_ground_truth_zero_is_ignored(self):
        result = self.metrics([1, 2, 1, 2], [1, 2, 0, 0])
        self.assertEqual(result['miou'], 1.0)
        self.assertEqual(result['accuracy'], 1.0)

    def test_all_predictions_zero(self):
        result = self.metrics([0, 0], [1, 2])
        self.assertEqual(result['miou'], 0.0)
        self.assertEqual(result['accuracy'], 0.0)

    def test_wrong_nonzero_prediction(self):
        result = self.metrics([1, 2, 2], [1, 1, 2])
        self.assertEqual(result['class_iou'], [0.5, 0.5])
        self.assertAlmostEqual(result['accuracy'], 2 / 3, places=6)

    def test_only_ignored_targets(self):
        result = self.metrics([1, 0], [0, 0])
        self.assertEqual(result['miou'], 0.0)
        self.assertEqual(result['accuracy'], 0.0)


if __name__ == '__main__':
    unittest.main()
