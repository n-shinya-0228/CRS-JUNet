import os
import torch
import numpy as np
import glob
import random
from torch.utils.data import Dataset
from lib.utils.laserscan_Polar3 import SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_EDGE = ['.png']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_edge(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_EDGE)

def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def load_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    return scan[:, 0:3], scan[:, 3:4]  # points(N,3), remissions(N,1)

def load_label(label_path):
    label = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
    return label


class SemanticKitti(Dataset):

    def __init__(self, root, sequences, labels, color_map,
                 learning_map, learning_map_inv, sensor,
                 max_points=150000, gt=True, skip=0,
                 is_train=False):
        super().__init__()
        self.root = os.path.join(root, "sequences")
        self.sequences = [f"{int(s):02d}" for s in sequences]
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.gt = gt
        self.is_train = is_train
        
        self.database_files = glob.glob("copy_paste_database/*.npy")

        self.scan_files = []
        for seq in self.sequences:
            velodyne_path = os.path.join(self.root, seq, "velodyne")
            if os.path.exists(velodyne_path):
                scans = [os.path.join(velodyne_path, f) for f in sorted(os.listdir(velodyne_path)) if f.endswith(".bin")]
                self.scan_files += scans

        if skip:
            self.scan_files = self.scan_files[::skip]

    def __len__(self):
        return len(self.scan_files)

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
        bin_file = self.scan_files[index]
        label_file = bin_file.replace("velodyne", "labels").replace(".bin", ".label")

        points, remissions = load_bin(bin_file)
        labels = load_label(label_file)

        if self.is_train and torch.rand(1) > 0.5:
            points, remissions, labels = self.apply_copy_paste(points, remissions, labels)

        labels = labels.flatten()

        scan = SemLaserScan(self.color_map, project=True)
        scan.set_points(points, remissions)
        scan.set_label(labels)

        proj_tensor = torch.from_numpy(scan.pseudo_image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(scan.proj_mask).unsqueeze(0).float()
        raw_labels = scan.proj_sem_label
        mapped_labels = self.map(raw_labels, self.learning_map)
        labels_t = torch.from_numpy(mapped_labels).long()

        # ch 0: max_z 
        proj_tensor[0] = torch.clamp(proj_tensor[0], -5.0, 15.0)
        proj_tensor[0] = (proj_tensor[0] - 1.0) / 5.0
        
        # ch 1: mean_z 
        proj_tensor[1] = torch.clamp(proj_tensor[1], -5.0, 15.0)
        proj_tensor[1] = (proj_tensor[1] - 1.0) / 5.0
        
        # ch 2: max_r 
        max_r = torch.max(proj_tensor[2])
        if max_r > 0.0:
            proj_tensor[2] = proj_tensor[2] / max_r
        proj_tensor[2] = torch.clamp(proj_tensor[2], 0.0, 1.0)
        
        # ch 3: density
        proj_tensor[3] = torch.clamp(proj_tensor[3], 0.0, 5.0) / 5.0
    
        # ch 4: z_diff
        proj_tensor[4] = torch.clamp(proj_tensor[4], 0.0, 10.0) / 10.0

        # ch 5: x_diff
        proj_tensor[5] = torch.clamp(proj_tensor[5], 0.0, 10.0) / 10.0

        # ch 6: y_diff 
        proj_tensor[6] = torch.clamp(proj_tensor[6], 0.0, 10.0) / 10.0

        if self.is_train:
            # 1. Azimuth Roll 
            roll_shift = torch.randint(0, proj_tensor.shape[2], (1,)).item()
            if roll_shift > 0:
                proj_tensor = torch.roll(proj_tensor, shifts=roll_shift, dims=2)
                mask_t = torch.roll(mask_t, shifts=roll_shift, dims=2)
                labels_t = torch.roll(labels_t, shifts=roll_shift, dims=1)

            # 2. ランダム水平反転
            if torch.rand(1) > 0.5:
                proj_tensor = torch.flip(proj_tensor, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                labels_t = torch.flip(labels_t, dims=[1])
                
            if torch.rand(1) > 0.5:
                drop_mask = (torch.rand(proj_tensor.shape[1:]) > 0.10).unsqueeze(0).float()
                proj_tensor = proj_tensor * drop_mask
                mask_t = mask_t * drop_mask

        dummy_list = []
        dummy_tensor = torch.tensor(0)

        path_norm = os.path.normpath(bin_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label") 

        return (
            proj_tensor, mask_t, labels_t, dummy_list, path_seq, path_name,
            dummy_list, dummy_list, dummy_tensor, dummy_list, 
            dummy_list, dummy_list, dummy_tensor, dummy_list, 
            torch.tensor(0), dummy_tensor
        )

    @staticmethod
    def map(label, mapdict):
        maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
        first_val = next(iter(mapdict.values()))
        if isinstance(first_val, list):
            lut = np.zeros((maxkey + 100, len(first_val)), dtype=np.float32)
        else:
            lut = np.zeros((maxkey + 100,), dtype=np.int32)
            
        for k, v in mapdict.items():
            lut[k] = v
            
        return lut[label]