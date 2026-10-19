"""Bounded-memory D.2 alpha histograms, moments, and post-hoc error analysis."""

import numpy as np

MODES = ("Baseline", "D2_Hard", "D2_Soft")
HIST_WIDTH = 0.0001
HIST_BINS = 10001


def confusion_metrics(matrix):
    clean = np.asarray(matrix, dtype=np.float64).copy()
    clean[0, :] = 0
    clean[:, 0] = 0
    tp = clean.diagonal()
    iou = tp / (clean.sum(0) + clean.sum(1) - tp + 1e-15)
    return dict(miou=float(iou[1:].mean()), accuracy=float(tp.sum() / (clean.sum() + 1e-15)),
                class_iou=iou[1:].tolist())


class AlphaStatistics:
    # Row 0: observed; 1:21: GT classes; 21:41: raw predicted classes;
    # rows 41/42: raw stage1 correct/wrong, restricted to observed labeled cells.
    def __init__(self):
        self.count = np.zeros(43, dtype=np.int64)
        self.sum = np.zeros((2, 43), dtype=np.float64)
        self.sumsq = np.zeros((2, 43), dtype=np.float64)
        self.minimum = np.full((2, 43), np.inf)
        self.maximum = np.full((2, 43), -np.inf)
        self.hist = np.zeros((2, 43, HIST_BINS), dtype=np.int64)
        self.unobserved = 0
        self.absolute_difference_sum = 0.
        self.partition_confusions = np.zeros((3, 2, 20, 20), dtype=np.int64)
        self.final_correct = np.zeros((3, 2), dtype=np.int64)

    def _add(self, mode, values, groups, offset, size):
        if not len(values):
            return
        values = np.asarray(values, dtype=np.float64)
        counts = np.bincount(groups, minlength=size)
        if mode == 0:
            self.count[offset:offset+size] += counts
        self.sum[mode, offset:offset+size] += np.bincount(groups, weights=values, minlength=size)
        self.sumsq[mode, offset:offset+size] += np.bincount(groups, weights=values**2, minlength=size)
        bins = np.clip(np.floor((values - .6) / HIST_WIDTH + 1e-3).astype(np.int64), 0, HIST_BINS-1)
        self.hist[mode, offset:offset+size] += np.bincount(
            groups * HIST_BINS + bins, minlength=size*HIST_BINS).reshape(size, HIST_BINS)
        np.minimum.at(self.minimum[mode, offset:offset+size], groups, values)
        np.maximum.at(self.maximum[mode, offset:offset+size], groups, values)

    def update(self, hard, soft, targets, raw_predictions, final_predictions, unobserved):
        """Only observed-cell vectors; GT is diagnostic input, never inference input."""
        hard, soft = np.asarray(hard), np.asarray(soft)
        targets, raw_predictions = np.asarray(targets, dtype=np.int64), np.asarray(raw_predictions, dtype=np.int64)
        if not (hard.shape == soft.shape == targets.shape == raw_predictions.shape):
            raise ValueError("Statistics vectors must have equal shapes")
        if not all(np.isfinite(v).all() for v in (hard, soft)):
            raise ValueError("Non-finite diagnostic alpha")
        labeled = targets > 0
        correct = raw_predictions == targets
        partition = (~correct[labeled]).astype(np.int64)
        self.unobserved += int(unobserved)
        self.absolute_difference_sum += np.abs(hard.astype(np.float64)-soft).sum()
        for mode, values in enumerate((hard, soft)):
            self._add(mode, values, np.zeros(len(values), dtype=np.int64), 0, 1)
            self._add(mode, values, targets, 1, 20)
            self._add(mode, values, raw_predictions, 21, 20)
            self._add(mode, values[labeled], partition, 41, 2)
        for mode, predictions in enumerate(final_predictions):
            predictions = np.asarray(predictions, dtype=np.int64)[labeled]
            target = targets[labeled]
            for part in (0, 1):
                select = partition == part
                self.partition_confusions[mode, part] += np.bincount(
                    predictions[select]*20+target[select], minlength=400).reshape(20, 20)
                self.final_correct[mode, part] += np.count_nonzero(predictions[select] == target[select])

    def snapshot(self):
        flat = self.hist.reshape(2, -1)
        indices = [np.flatnonzero(row) for row in flat]
        return dict(count=self.count.tolist(), sum=self.sum.tolist(), sumsq=self.sumsq.tolist(),
            minimum=np.where(self.count[None, :] > 0, self.minimum, 0).tolist(),
            maximum=np.where(self.count[None, :] > 0, self.maximum, 0).tolist(),
            histogram_indices=[row.tolist() for row in indices],
            histogram_counts=[row[index].tolist() for row, index in zip(flat, indices)],
            unobserved=self.unobserved, absolute_difference_sum=self.absolute_difference_sum,
            partition_confusions=self.partition_confusions.tolist(), final_correct=self.final_correct.tolist())

    def merge(self, other):
        for key in ("count", "sum", "sumsq", "hist", "partition_confusions", "final_correct"):
            getattr(self, key)[:] += getattr(other, key)
        np.minimum(self.minimum, other.minimum, out=self.minimum)
        np.maximum(self.maximum, other.maximum, out=self.maximum)
        self.unobserved += other.unobserved
        self.absolute_difference_sum += other.absolute_difference_sum

    @classmethod
    def restore(cls, saved):
        result = cls()
        for key in ("count", "sum", "sumsq", "minimum", "maximum", "partition_confusions", "final_correct"):
            value = np.asarray(saved[key], dtype=getattr(result, key).dtype)
            if value.shape != getattr(result, key).shape or not np.isfinite(value).all():
                raise ValueError("Invalid saved alpha statistic " + key)
            setattr(result, key, value)
        result.minimum[:, result.count == 0] = np.inf
        result.maximum[:, result.count == 0] = -np.inf
        for mode in (0, 1):
            indices = np.asarray(saved["histogram_indices"][mode], dtype=np.int64)
            counts = np.asarray(saved["histogram_counts"][mode], dtype=np.int64)
            if (len(indices) != len(counts) or (counts < 0).any() or (indices < 0).any() or
                    (indices >= 43*HIST_BINS).any() or len(np.unique(indices)) != len(indices)):
                raise ValueError("Invalid saved histogram")
            result.hist[mode].reshape(-1)[indices] = counts
            if not np.array_equal(result.hist[mode].sum(1), result.count):
                raise ValueError("Alpha histogram and sample counts differ")
        if (result.count < 0).any():
            raise ValueError("Negative saved statistic counts")
        result.unobserved = int(saved["unobserved"])
        result.absolute_difference_sum = float(saved["absolute_difference_sum"])
        if result.unobserved < 0 or result.absolute_difference_sum < 0:
            raise ValueError("Invalid unobserved/difference statistic")
        return result

    def distribution(self, mode, row=0, scope="observed"):
        count = int(self.count[row])
        total, squared = self.sum[mode, row], self.sumsq[mode, row]
        histogram = self.hist[mode, row].copy()
        minimum, maximum = self.minimum[mode, row], self.maximum[mode, row]
        if scope == "unobserved":
            count, total, squared = self.unobserved, self.unobserved*1.2, self.unobserved*1.44
            histogram[:] = 0
            histogram[6000] = self.unobserved
            minimum = maximum = 1.2
        elif scope == "all":
            if self.unobserved:
                minimum, maximum = min(minimum, 1.2), max(maximum, 1.2)
            count += self.unobserved
            total += self.unobserved*1.2
            squared += self.unobserved*1.44
            histogram[6000] += self.unobserved
        if not count:
            return dict(count=0, mean=None, std=None, min=None, max=None, p05=None, p50=None, p95=None)
        if scope == "unobserved":
            return dict(count=count, mean=1.2, std=0., min=1.2, max=1.2, p05=1.2, p50=1.2, p95=1.2)
        mean = float(total / count)
        quantiles = []
        cumulative = histogram.cumsum()
        for fraction in (.05, .5, .95):
            index = int(np.searchsorted(cumulative, max(1, int(np.ceil(fraction*count)))))
            quantiles.append(min(1.6, .6 + index*HIST_WIDTH + (HIST_WIDTH/2 if mode else 0)))
        return dict(count=count, mean=mean, std=float(np.sqrt(max(0., squared/count-mean**2))),
                    min=float(minimum), max=float(maximum), **dict(zip(("p05", "p50", "p95"), quantiles)))

    def diagnostics(self):
        counts = self.count[41:43]
        total = int(counts.sum())
        return dict(scope="observed cells with GT learning ID 1..19; raw stage1 argmax before height correction",
            count=total, stage1_correct_count=int(counts[0]), stage1_wrong_count=int(counts[1]),
            stage1_correct_fraction=float(counts[0]/total) if total else None,
            stage1_wrong_fraction=float(counts[1]/total) if total else None,
            by_mode={name:{label:dict(count=int(counts[part]), final_correct=int(self.final_correct[mode, part]),
                final_correct_fraction=float(self.final_correct[mode, part]/counts[part]) if counts[part] else None,
                **confusion_metrics(self.partition_confusions[mode, part]))
                for part, label in enumerate(("stage1_correct", "stage1_wrong"))} for mode, name in enumerate(MODES)},
            observed_hard_soft_mean_absolute_difference=float(self.absolute_difference_sum/self.count[0]) if self.count[0] else None,
            all_hard_soft_mean_absolute_difference=float(self.absolute_difference_sum/(self.count[0]+self.unobserved)) if self.count[0]+self.unobserved else None)
