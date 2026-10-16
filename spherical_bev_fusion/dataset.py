import glob
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .projection import build_dual_projection


class DualProjectionDataset(Dataset):
    def __init__(self, root, sequences, data, sensor, training=False):
        self.files = []
        for seq in sequences:
            folder = Path(root) / "sequences" / f"{int(seq):02d}" / "velodyne"
            files = sorted(folder.glob("*.bin"))
            if not files:
                raise FileNotFoundError(f"SemanticKITTI scans missing: {folder}")
            self.files.extend(files)
        self.training = training
        self.learning_map = data["learning_map"]
        self.database_files = glob.glob(str(Path(__file__).resolve().parents[1] / "copy_paste_database" / "*.npy"))
        self.sensor = dict(height=sensor["img_prop"]["height"],
                           width=sensor["img_prop"]["width"],
                           fov_up=sensor["fov_up"], fov_down=sensor["fov_down"],
                           means=sensor["img_means"], stds=sensor["img_stds"])

    def __len__(self):
        return len(self.files)

    def apply_copy_paste(self, points, remissions, labels):
        if not self.database_files:
            return points, remissions, labels

        num_paste = np.random.randint(1, 4) # 1〜3個の物体をランダムに貼り付け
        new_pts, new_rems, new_lbls = [points], [remissions], [labels]

        for _ in range(num_paste):
            npy_file = random.choice(self.database_files)
            obj_data = np.load(npy_file) # [N, 5] (X, Y, Z, rem, label)

            obj_pts = obj_data[:, :3].copy()
            obj_rem = obj_data[:, 3:4].copy()
            obj_lbl = obj_data[:, 4:5].astype(np.int32).copy()

            angle = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,             0,             1]
            ], dtype=np.float32)
            obj_pts = np.dot(obj_pts, rot_mat)

            r = np.random.uniform(5.0, 40.0)
            theta = np.random.uniform(0, 2 * np.pi)
            obj_pts[:, 0] += r * np.cos(theta)
            obj_pts[:, 1] += r * np.sin(theta)

            new_pts.append(obj_pts)
            new_rems.append(obj_rem)
            new_lbls.append(obj_lbl)

        return np.concatenate(new_pts, axis=0), np.concatenate(new_rems, axis=0), np.concatenate(new_lbls, axis=0)

    def __getitem__(self, index):
        path = self.files[index]
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        label_path = path.parent.parent / "labels" / (path.stem + ".label")
        labels = np.fromfile(label_path, dtype=np.uint32)
        if len(labels) != len(scan):
            raise ValueError(f"Scan/label length mismatch: {path}")
        points, remissions = scan[:, :3].copy(), scan[:, 3:4].copy()
        labels = labels.reshape(-1, 1)
        if self.training and torch.rand(()) > 0.5:
            points, remissions, labels = self.apply_copy_paste(points, remissions, labels)
        # Augment the shared cloud before either projection, preserving alignment.
        if self.training:
            angle = np.random.uniform(0, 2 * np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]], dtype=np.float32)
            points[:, :2] = points[:, :2] @ rotation.T
            if torch.rand(()) > 0.5:
                points[:, 1] *= -1
        semantic = labels.reshape(-1).astype(np.int64) & 0xFFFF
        mapped = np.zeros(len(semantic), dtype=np.int64)
        for raw_id, train_id in self.learning_map.items():
            mapped[semantic == int(raw_id)] = int(train_id)
        sample = build_dual_projection(points, remissions, mapped, self.sensor)
        keys = ("bev", "sphere", "bev_labels", "sphere_labels",
                "bev_to_sphere", "bev_to_sphere_valid")
        return {key: torch.from_numpy(sample[key]) for key in keys}


class ParserBEVSphere:
    def __init__(self, root, data_cfg, arch_cfg):
        options = dict(batch_size=arch_cfg["train"]["batch_size"],
                       num_workers=arch_cfg["train"]["workers"],
                       pin_memory=torch.cuda.is_available())
        self.train_dataset = DualProjectionDataset(root, data_cfg["split"]["train"],
            data_cfg, arch_cfg["dataset"]["sensor"], True)
        self.valid_dataset = DualProjectionDataset(root, data_cfg["split"]["valid"],
            data_cfg, arch_cfg["dataset"]["sensor"])
        self.trainloader = DataLoader(self.train_dataset, shuffle=True, **options)
        self.validloader = DataLoader(self.valid_dataset, shuffle=False, **options)
        self.nclasses = len(data_cfg["learning_map_inv"])

    def get_n_classes(self):
        return self.nclasses
