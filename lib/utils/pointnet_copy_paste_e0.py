"""Read-only E.0 diagnostics. This module is not imported by training code."""

import ast
import gzip
import hashlib
import inspect
import random
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

from lib.dataset.SemanticKitti_PointNet import SemanticKitti
from lib.utils.laserscan_Polar4 import PRIORITY_MAP


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bev(path):
    with gzip.open(path, "rb") as stream:
        value = torch.load(stream, map_location="cpu", weights_only=True)
    return (value["proj_tensor"].float(), value["mask_t"].float(),
            value["labels_t"].long())


def cell_ids_numpy(points):
    radius = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    angle = np.arctan2(points[:, 1], points[:, 0])
    valid = (radius > 0) & (radius < 51.2)
    rows = np.clip(np.floor(radius / 0.1).astype(np.int64), 0, 511)
    cols = np.clip(np.floor((angle + np.pi) / (2 * np.pi / 512)).astype(np.int64), 0, 511)
    return rows * 512 + cols, valid


def project_handcrafted(points, packed_labels, learning_map):
    """Vectorized reference for Polar4, including its priority/tie ordering.

    Must be checked against SemLaserScan on real data before interpreting tests.
    The float32 mean reduction can differ by rounding from np.mean per cell.
    """
    semantic = packed_labels.astype(np.int64) & 0xFFFF
    order = np.argsort(PRIORITY_MAP[semantic])
    points, semantic = points[order], semantic[order]
    ids, valid = cell_ids_numpy(points)
    points, semantic, ids = points[valid], semantic[valid], ids[valid]
    order = np.argsort(ids, kind="stable")
    points, semantic, ids = points[order], semantic[order], ids[order]
    cells, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    feature = np.zeros((7, 512 * 512), dtype=np.float32)
    mask = np.zeros(512 * 512, dtype=bool)
    labels = np.zeros(512 * 512, dtype=np.int64)
    if len(cells):
        maxima = np.maximum.reduceat(points, starts, axis=0)
        minima = np.minimum.reduceat(points, starts, axis=0)
        feature[0, cells] = maxima[:, 2]
        feature[1, cells] = np.add.reduceat(points[:, 2], starts) / counts
        feature[2, cells] = maxima[:, 3]
        feature[3, cells] = counts / 100.0
        feature[4, cells] = maxima[:, 2] - minima[:, 2]
        feature[5, cells] = maxima[:, 0] - minima[:, 0]
        feature[6, cells] = maxima[:, 1] - minima[:, 1]
        lut = np.zeros(max(learning_map) + 100, dtype=np.int64)
        for key, value in learning_map.items():
            lut[key] = value
        labels[cells] = lut[semantic[starts + counts - 1]]
        mask[cells] = True
    return (torch.from_numpy(feature.reshape(7, 512, 512)),
            torch.from_numpy(mask.reshape(1, 512, 512)),
            torch.from_numpy(labels.reshape(512, 512)))


def normalize_handcrafted(value):
    value = value.clone()
    value[:2] = (value[:2].clamp(-5, 15) - 1) / 5
    maximum = value[2].max()
    if maximum > 0:
        value[2] /= maximum
    value[2] = value[2].clamp(0, 1)
    value[3] = value[3].clamp(0, 5) / 5
    value[4:] = value[4:].clamp(0, 10) / 10
    return value


def differences(actual, expected, region=None, atol=0, rtol=0):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if region is not None:
        a, b = a[..., region], b[..., region]
    a, b = a.double(), b.double()
    error = (a - b).abs()
    bad = error > (atol + rtol * b.abs())
    channel_axis = actual.ndim == 3 or (region is not None and actual.ndim > 2)
    return {"values": a.numel(), "different_values": int(bad.sum()),
            "different_cells": int(bad.any(dim=0).sum()) if channel_axis else int(bad.sum()),
            "max_absolute_error": float(error.max()) if error.numel() else 0.0,
            "mean_absolute_error": float(error.mean()) if error.numel() else 0.0,
            "atol": atol, "rtol": rtol}


def ambiguous_priority_cells(points, packed_labels, learning_map):
    """Cells with multiple learning IDs tied at the highest raw-label priority."""
    ids, valid = cell_ids_numpy(points)
    semantic = (packed_labels.astype(np.int64) & 0xFFFF)[valid]
    ids = ids[valid]
    priorities = PRIORITY_MAP[semantic]
    maximum = np.full(512 * 512, -1, dtype=np.int32)
    np.maximum.at(maximum, ids, priorities)
    highest = priorities == maximum[ids]
    lut = np.zeros(max(learning_map) + 100, dtype=np.int64)
    for key, value in learning_map.items():
        lut[key] = value
    mapped = lut[semantic[highest]]
    low = np.full(512 * 512, 1000, dtype=np.int64)
    high = np.full(512 * 512, -1, dtype=np.int64)
    np.minimum.at(low, ids[highest], mapped)
    np.maximum.at(high, ids[highest], mapped)
    return torch.from_numpy(((low != high) & (high >= 0)).reshape(512, 512))


class CandidateTrace:
    """Observe the *unmodified* production function, without replaying its logic."""

    def __init__(self):
        function = SemanticKitti.apply_bev_copy_paste
        source, first = inspect.getsourcelines(function)
        tree = ast.parse(textwrap.dedent("".join(source)))
        self.code = function.__code__
        self.begin = None
        self.class_line = None
        self.reasons = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "context_classes" for t in node.targets):
                self.class_line = first + node.lineno - 1
            if isinstance(node, ast.Try) and self.begin is None:
                self.begin = first + node.lineno - 1
            if isinstance(node, ast.If):
                for child in node.body:
                    if isinstance(child, ast.Continue):
                        self.reasons[first + child.lineno - 1] = ast.unparse(node.test)
            if isinstance(node, ast.ExceptHandler):
                for child in node.body:
                    if isinstance(child, ast.Continue):
                        self.reasons[first + child.lineno - 1] = "donor_read_error"
        self.rows = []
        self.success = None

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        local = frame.f_locals
        if event == "line" and frame.f_lineno == self.begin:
            self.rows.append({"donor": local["src_file"], "attempt": local["attempt"],
                              "reason": "pending", "success": False})
        if self.rows:
            row = self.rows[-1]
            if event == "line" and frame.f_lineno == self.class_line:
                row["class_id"] = local["obj_cls"]
            if event == "line" and frame.f_lineno in self.reasons:
                row["reason"] = self.reasons[frame.f_lineno]
            if event == "return" and arg is not None and int(arg[-1].sum()) > 0:
                row.update(reason="success", success=True, class_id=local["obj_cls"])
                keys = ["src_file", "x1", "x2", "y1", "y2", "ty", "tx", "h", "w", "obj_cls"]
                self.success = {key: local[key] for key in keys}
                self.success["m"] = local["m"].clone()
                self.success["visible_ratio"] = float(local["visible_ratio"])
                self.success["overlap_ratio"] = float(local["overlap_ratio"])
        return self.trace

    def call(self, dataset, inputs, **kwargs):
        old = sys.gettrace()
        start = time.perf_counter()
        try:
            sys.settrace(self.trace)
            output = dataset.apply_bev_copy_paste(*inputs, **kwargs)
        finally:
            sys.settrace(old)
        return output, time.perf_counter() - start


def transplant_cell_points(target_points, target_labels, donor_points, donor_labels, paste):
    """Diagnostic alternative, NOT a recovery of legacy CP's nonexistent raw state.

    Replace all target points in visible cells with all donor points from selected
    cells. Preserve within-cell polar offsets, z, remission and packed labels.
    No object completeness, rigid-body motion or 3D visibility is asserted.
    """
    source_cells = np.zeros((512, 512), dtype=bool)
    destination_cells = np.zeros_like(source_cells)
    m = paste["m"].squeeze(0).numpy()
    source_cells[paste["y1"]:paste["y2"], paste["x1"]:paste["x2"]] = m
    destination_cells[paste["ty"]:paste["ty"] + paste["h"],
                      paste["tx"]:paste["tx"] + paste["w"]] = m
    tid, tv = cell_ids_numpy(target_points)
    did, dv = cell_ids_numpy(donor_points)
    remove = tv & destination_cells.reshape(-1)[tid]
    select = dv & source_cells.reshape(-1)[did]
    moved = donor_points[select].copy()
    radius = np.hypot(moved[:, 0], moved[:, 1]).astype(np.float64)
    angle = np.arctan2(moved[:, 1], moved[:, 0]).astype(np.float64)
    radius += (paste["ty"] - paste["y1"]) * 0.1
    angle += (paste["tx"] - paste["x1"]) * (2 * np.pi / 512)
    moved[:, 0], moved[:, 1] = radius * np.cos(angle), radius * np.sin(angle)
    expected = ((did[select] // 512 + paste["ty"] - paste["y1"]) * 512
                + (did[select] % 512 + paste["tx"] - paste["x1"]) % 512)
    actual, valid = cell_ids_numpy(moved)
    info = {"target_points_removed": int(remove.sum()), "donor_points_added": int(select.sum()),
            "transformed_cell_mismatch_points": int((actual != expected).sum()),
            "invalid_transformed_points": int((~valid).sum()),
            "rigid_transform": False, "physical_occlusion_model": False}
    return (np.concatenate([target_points[~remove], moved]),
            np.concatenate([target_labels[~remove], donor_labels[select]]), info)


def off_reference(dataset, index, seed):
    """Independent CP-OFF array pipeline, retaining the baseline dropout contract."""
    seed_all(seed)
    torch.rand(1)
    path = dataset.scan_files[index]
    hand, mask, label = read_bev(path)
    point, occupied = dataset.load_pointnet_feature(path)
    point = dataset.align_pointnet_mask(path, point, occupied, mask)
    hand = normalize_handcrafted(hand)
    shift = int(torch.randint(0, 512, (1,)))
    hand, point, mask = [torch.roll(v, shift, 2) for v in (hand, point, mask)]
    label = torch.roll(label, shift, 1)
    if torch.rand(1) > .5:
        hand, point, mask = [v.flip([2]) for v in (hand, point, mask)]
        label = label.flip([1])
    if torch.rand(1) > .5:
        keep = (torch.rand(512, 512) > .10).unsqueeze(0)
        hand, point, mask = [v * keep for v in (hand, point, mask)]
    return torch.cat([hand, point]), mask, label, torch.zeros_like(mask)


def read_raw(bev_path):
    path = Path(bev_path)
    sequence, frame = path.parent.parent, path.stem
    return (np.fromfile(sequence / "velodyne" / (frame + ".bin"), dtype=np.float32).reshape(-1, 4),
            np.fromfile(sequence / "labels" / (frame + ".label"), dtype=np.uint32))
