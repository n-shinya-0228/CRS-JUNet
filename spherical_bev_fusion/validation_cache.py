"""Lossless, opt-in validation cache. Never caches augmented training samples."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


HOME = Path(__file__).resolve().parent
ARRAY_KEYS = ("bev", "sphere", "bev_labels", "sphere_labels", "bev_mask",
              "sphere_mask", "bev_to_sphere", "bev_to_sphere_valid")


class ValidationCache:
    def __init__(self, root, sensor, learning_map):
        self.root = Path(root).resolve()
        if HOME not in self.root.parents:
            raise ValueError("Validation cache must be below the training package")
        self.sensor = sensor
        self.learning_map = learning_map
        self.projection_hash = hashlib.sha256((HOME / 'projection.py').read_bytes()).hexdigest()

    def signature(self, scan):
        scan = Path(scan).resolve()
        label = scan.parent.parent / 'labels' / (scan.stem + '.label')
        sources = []
        for path in (scan, label):
            stat = path.stat()
            sources.append([str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
        metadata = dict(version=1, sources=sources, sensor=self.sensor,
                        learning_map=self.learning_map, projection=self.projection_hash)
        return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def path(self, scan, signature):
        scan = Path(scan)
        path = self.root / scan.parent.parent.name / (scan.stem + '_' + signature + '.npz')
        if HOME not in path.resolve().parents:
            raise ValueError("Cache path escapes the training package")
        return path

    def load(self, scan):
        signature = self.signature(scan)
        path = self.path(scan, signature)
        if not path.is_file():
            return None  # Missing/stale cache uses the original projection path.
        with np.load(path, allow_pickle=False) as archive:
            if archive['signature'].item() != signature:
                raise ValueError(f'Cache metadata mismatch: {path}')
            result = {key: archive[key] for key in ARRAY_KEYS}
        if result['bev'].shape != (7, 512, 512):
            raise ValueError(f'Invalid BEV cache shape: {path}')
        if result['sphere'].shape != (6, self.sensor['height'], self.sensor['width']):
            raise ValueError(f'Invalid sphere cache shape: {path}')
        bev_shape = (512, 512)
        sphere_shape = (self.sensor['height'], self.sensor['width'])
        expected = dict(bev=((7, *bev_shape), np.float32),
                        sphere=((6, *sphere_shape), np.float32),
                        bev_labels=(bev_shape, np.int64),
                        sphere_labels=(sphere_shape, np.int64),
                        bev_mask=(bev_shape, np.float32),
                        sphere_mask=(sphere_shape, np.float32),
                        bev_to_sphere=((*bev_shape, 2), np.float32),
                        bev_to_sphere_valid=(bev_shape, np.float32))
        for key, (shape, dtype) in expected.items():
            if result[key].shape != shape or result[key].dtype != dtype:
                raise ValueError(f'Invalid cached array {key}: {path}')
        return result

    def store(self, scan, sample):
        signature = self.signature(scan)
        path = self.path(scan, signature)
        if path.exists():
            existing = self.load(scan)
            for key in ARRAY_KEYS:
                np.testing.assert_array_equal(existing[key], sample[key], err_msg=key)
            return path  # Existing valid entries are verified, never overwritten.
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                # No float16 or lossy compression: preserve every original array.
                np.savez(stream, signature=np.array(signature),
                         **{key: sample[key] for key in ARRAY_KEYS})
                stream.flush()
                os.fsync(stream.fileno())
                os.link(temporary, path)  # Atomic publication, refuses overwrite.
            finally:
                temporary.unlink(missing_ok=True)
        return path
