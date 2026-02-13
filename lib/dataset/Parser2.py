# # Parser2_motorfocus.py
# import os
# import numpy as np
# import torch
# from torch.utils.data import WeightedRandomSampler, DataLoader

# from ..utils.laserscan3 import LaserScan, SemLaserScan
# from .SemanticKitti7 import SemanticKitti


# def kitti_collate_stack_only(batch):
#     """
#     SemanticKitti7.__getitem__ が返す 16 要素タプルを
#     バッチ化するための collate 関数（train/valid/test 共通）。

#     返り値も 16 要素で、既存の Trainer 側の unpack と
#     同じ並びになるように作る。
#     """
#     in_vol          = torch.stack([b[0] for b in batch], dim=0)   # [B,6,H,W]
#     proj_mask       = torch.stack([b[1] for b in batch], dim=0)   # [B,H,W]
#     proj_labels     = torch.stack([b[2] for b in batch], dim=0)   # [B,H,W]
#     unproj_labels   = [b[3] for b in batch]                       # list

#     path_seq        = [b[4] for b in batch]
#     path_name       = [b[5] for b in batch]

#     proj_x          = [b[6] for b in batch]                       # list of [M]
#     proj_y          = [b[7] for b in batch]                       # list of [M]

#     proj_range      = torch.stack([b[8] for b in batch], dim=0)   # [B,H,W]
#     unproj_range    = [b[9] for b in batch]                       # list

#     proj_xyz        = [b[10] for b in batch]                      # list of [H,W,3]
#     unproj_xyz      = [b[11] for b in batch]                      # list of [M,3]

#     proj_remission  = torch.stack([b[12] for b in batch], dim=0)  # [B,H,W]
#     unproj_remiss   = [b[13] for b in batch]                      # list

#     unproj_n_points = torch.tensor([b[14] for b in batch], dtype=torch.int32)  # [B]
#     edge            = torch.stack([b[15] for b in batch], dim=0)  # [B,H,W]

#     return (in_vol,
#             proj_mask,
#             proj_labels,
#             unproj_labels,
#             path_seq,
#             path_name,
#             proj_x,
#             proj_y,
#             proj_range,
#             unproj_range,
#             proj_xyz,
#             unproj_xyz,
#             proj_remission,
#             unproj_remiss,
#             unproj_n_points,
#             edge)


# class Parser:
#     """
#     SemanticKitti7 + motorcyclist 対策用の Parser

#     - train  : focus_motor=True で motor 周辺パッチ拡大
#                + WeightedRandomSampler で motor を含むフレームを重めに
#     - valid/test: focus_motor=False（augmentation なし）
#     """

#     def __init__(self, root, data_cfg, arch_cfg,
#                  gt=True, shuffle_train=True):

#         # 設定
#         self.root = root
#         self.data_cfg = data_cfg
#         self.arch_cfg = arch_cfg
#         self.gt = gt
#         self.shuffle_train = shuffle_train

#         # sensor / label 関連
#         self.sensor = arch_cfg["dataset"]["sensor"]
#         self.labels = data_cfg["labels"]
#         self.color_map = data_cfg["color_map"]
#         self.learning_map = data_cfg["learning_map"]
#         self.learning_map_inv = data_cfg["learning_map_inv"]
#         self.learning_ignore = data_cfg["learning_ignore"]

#         self.max_points = arch_cfg["dataset"]["max_points"]
#         self.batch_size = arch_cfg["train"]["batch_size"]
#         self.workers = arch_cfg["train"]["workers"]
#         self.DATASET_TYPE = SemanticKitti

#         # クラス数（xentropy 用）
#         self.nclasses = len(self.learning_map_inv)

#         # シーケンス分割
#         self.train_sequences = data_cfg["split"]["train"]
#         self.valid_sequences = data_cfg["split"]["valid"]
#         self.test_sequences  = data_cfg["split"]["test"]

#         # ============================================================
#         # 1. train dataset
#         # ============================================================
#         self.train_dataset = self.DATASET_TYPE(
#             root=self.root,
#             sequences=self.train_sequences,
#             labels=self.labels,
#             color_map=self.color_map,
#             learning_map=self.learning_map,
#             learning_map_inv=self.learning_map_inv,
#             sensor=self.sensor,
#             max_points=self.max_points,
#             gt=self.gt,
#             # 小穴 inpaint
#             max_gap_row=2,
#             max_gap_col=2,
#             inpaint_mode="linear",
#             median_ksize=0,
#             # ★ motor 周辺パッチを拡大学習
#             focus_motor=True,
#             focus_prob=0.7,
#             patch_h=64,
#             patch_w=256,
#         )

#         # ----------------------------
#         # motor を含むフレームのサンプリング重み付け
#         # ----------------------------
#         num_train = len(self.train_dataset)
#         motor_present = np.zeros(num_train, dtype=bool)

#         print("[Parser] scanning train labels for motorcyclist presence...")
#         MOTOR_ID = 8  # xentropy ID の motorcyclist

#         for idx in range(num_train):
#             label_path = self.train_dataset.label_files[idx]
#             labels_raw = np.fromfile(label_path, dtype=np.uint32)
#             sem = labels_raw & 0xFFFF
#             sem_x = self.DATASET_TYPE.map(sem, self.learning_map)
#             if (sem_x == MOTOR_ID).any():
#                 motor_present[idx] = True

#         # 重み設定（適宜いじってOK）
#         w_non = 1.0
#         w_mot = 5.0
#         weights = np.where(motor_present, w_mot, w_non).astype(np.float32)

#         sampler = WeightedRandomSampler(
#             weights,
#             num_samples=num_train,   # 1 epoch あたりほぼ全フレームぶんサンプル
#             replacement=True
#         )

#         self.trainloader = DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             sampler=sampler,
#             shuffle=False,          # sampler を使うので shuffle=False
#             num_workers=self.workers,
#             pin_memory=True,
#             drop_last=True,
#             collate_fn=kitti_collate_stack_only,
#         )

#         # ============================================================
#         # 2. valid dataset（augmentation なし, focus_motor=False）
#         # ============================================================
#         self.valid_dataset = self.DATASET_TYPE(
#             root=self.root,
#             sequences=self.valid_sequences,
#             labels=self.labels,
#             color_map=self.color_map,
#             learning_map=self.learning_map,
#             learning_map_inv=self.learning_map_inv,
#             sensor=self.sensor,
#             max_points=self.max_points,
#             gt=self.gt,
#             max_gap_row=2,
#             max_gap_col=2,
#             inpaint_mode="linear",
#             median_ksize=0,
#             focus_motor=False,    # ★ valid はそのまま
#         )

#         self.validloader = DataLoader(
#             self.valid_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.workers,
#             pin_memory=True,
#             drop_last=True,
#             collate_fn=kitti_collate_stack_only,
#         )

#         # ============================================================
#         # 3. test dataset（augmentation なし, focus_motor=False）
#         # ============================================================
#         if self.test_sequences:
#             self.test_dataset = self.DATASET_TYPE(
#                 root=self.root,
#                 sequences=self.test_sequences,
#                 labels=self.labels,
#                 color_map=self.color_map,
#                 learning_map=self.learning_map,
#                 learning_map_inv=self.learning_map_inv,
#                 sensor=self.sensor,
#                 max_points=self.max_points,
#                 gt=False,
#                 max_gap_row=2,
#                 max_gap_col=2,
#                 inpaint_mode="linear",
#                 median_ksize=0,
#                 focus_motor=False,
#             )

#             self.testloader = DataLoader(
#                 self.test_dataset,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=self.workers,
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=kitti_collate_stack_only,
#             )
#         else:
#             self.test_dataset = None
#             self.testloader = None

#     # ===== interface =====
#     def get_train_set(self):
#         return self.trainloader

#     def get_valid_set(self):
#         return self.validloader

#     def get_test_set(self):
#         return self.testloader

#     def get_train_size(self):
#         return len(self.trainloader)

#     def get_valid_size(self):
#         return len(self.validloader)

#     def get_test_size(self):
#         return len(self.testloader) if self.testloader is not None else 0

#     def get_n_classes(self):
#         return self.nclasses

#     def get_xentropy_class_string(self, idx):
#         orig_id = self.learning_map_inv[idx]
#         return self.labels[orig_id]

#     def to_original(self, label):
#         return self.DATASET_TYPE.map(label, self.learning_map_inv)

#     def to_xentropy(self, label):
#         return self.DATASET_TYPE.map(label, self.learning_map)

#     def to_color(self, label):
#         label = self.DATASET_TYPE.map(label, self.learning_map_inv)
#         return self.DATASET_TYPE.map(label, self.color_map)



# Parser2_nomotorfocus.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.laserscan3 import LaserScan, SemLaserScan
from .SemanticKitti7 import SemanticKitti


def kitti_collate_stack_only(batch):
    """
    SemanticKitti7.__getitem__ が返す 16 要素タプルを
    バッチ化するための collate 関数（train/valid/test 共通）。

    返り値も 16 要素で、Trainer 側の unpack と
    同じ並びになるように作る。
    """
    in_vol          = torch.stack([b[0] for b in batch], dim=0)   # [B,6,H,W]
    proj_mask       = torch.stack([b[1] for b in batch], dim=0)   # [B,H,W]
    proj_labels     = torch.stack([b[2] for b in batch], dim=0)   # [B,H,W]
    unproj_labels   = [b[3] for b in batch]                       # list

    path_seq        = [b[4] for b in batch]
    path_name       = [b[5] for b in batch]

    proj_x          = [b[6] for b in batch]                       # list of [M]
    proj_y          = [b[7] for b in batch]                       # list of [M]

    proj_range      = torch.stack([b[8] for b in batch], dim=0)   # [B,H,W]
    unproj_range    = [b[9] for b in batch]                       # list

    proj_xyz        = [b[10] for b in batch]                      # list of [H,W,3]
    unproj_xyz      = [b[11] for b in batch]                      # list of [M,3]

    proj_remission  = torch.stack([b[12] for b in batch], dim=0)  # [B,H,W]
    unproj_remiss   = [b[13] for b in batch]                      # list

    unproj_n_points = torch.tensor([b[14] for b in batch], dtype=torch.int32)  # [B]
    edge            = torch.stack([b[15] for b in batch], dim=0)  # [B,H,W]

    return (in_vol,
            proj_mask,
            proj_labels,
            unproj_labels,
            path_seq,
            path_name,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remiss,
            unproj_n_points,
            edge)


class Parser:
    """
    SemanticKitti7 用のシンプル Parser（motor 強調なし）

    - train  : focus_motor=False, 通常の shuffle DataLoader
    - valid  : focus_motor=False, shuffle=False
    - test   : focus_motor=False, shuffle=False
    """

    def __init__(self, root, data_cfg, arch_cfg,
                 gt=True, shuffle_train=True):

        # 設定
        self.root = root
        self.data_cfg = data_cfg
        self.arch_cfg = arch_cfg
        self.gt = gt
        self.shuffle_train = shuffle_train

        # sensor / label 関連
        self.sensor = arch_cfg["dataset"]["sensor"]
        self.labels = data_cfg["labels"]
        self.color_map = data_cfg["color_map"]
        self.learning_map = data_cfg["learning_map"]
        self.learning_map_inv = data_cfg["learning_map_inv"]
        self.learning_ignore = data_cfg["learning_ignore"]

        self.max_points = arch_cfg["dataset"]["max_points"]
        self.batch_size = arch_cfg["train"]["batch_size"]
        self.workers = arch_cfg["train"]["workers"]
        self.DATASET_TYPE = SemanticKitti

        # クラス数（xentropy 用）
        self.nclasses = len(self.learning_map_inv)

        # シーケンス分割
        self.train_sequences = data_cfg["split"]["train"]
        self.valid_sequences = data_cfg["split"]["valid"]
        self.test_sequences  = data_cfg["split"]["test"]

        # ============================================================
        # 1. train dataset（motor 特別扱いなし）
        # ============================================================
        self.train_dataset = self.DATASET_TYPE(
            root=self.root,
            sequences=self.train_sequences,
            labels=self.labels,
            color_map=self.color_map,
            learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv,
            sensor=self.sensor,
            max_points=self.max_points,
            gt=self.gt,
            # 小穴 inpaint
            max_gap_row=2,
            max_gap_col=2,
            inpaint_mode="linear",
            median_ksize=0,
            # ★ motor 周辺パッチ拡大はオフ
            focus_motor=False,
            focus_prob=0.0,
            patch_h=64,
            patch_w=256,
        )

        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,   # ふつうの shuffle
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=kitti_collate_stack_only,
        )

        # ============================================================
        # 2. valid dataset（augmentation なし, focus_motor=False）
        # ============================================================
        self.valid_dataset = self.DATASET_TYPE(
            root=self.root,
            sequences=self.valid_sequences,
            labels=self.labels,
            color_map=self.color_map,
            learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv,
            sensor=self.sensor,
            max_points=self.max_points,
            gt=self.gt,
            max_gap_row=2,
            max_gap_col=2,
            inpaint_mode="linear",
            median_ksize=0,
            focus_motor=False,
            focus_prob=0.0,
            patch_h=64,
            patch_w=256,
        )

        self.validloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=kitti_collate_stack_only,
        )

        # ============================================================
        # 3. test dataset（augmentation なし, focus_motor=False）
        # ============================================================
        if self.test_sequences:
            self.test_dataset = self.DATASET_TYPE(
                root=self.root,
                sequences=self.test_sequences,
                labels=self.labels,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                sensor=self.sensor,
                max_points=self.max_points,
                gt=False,
                max_gap_row=2,
                max_gap_col=2,
                inpaint_mode="linear",
                median_ksize=0,
                focus_motor=False,
                focus_prob=0.0,
                patch_h=64,
                patch_w=256,
            )

            self.testloader = DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=kitti_collate_stack_only,
            )
        else:
            self.test_dataset = None
            self.testloader = None

    # ===== interface =====
    def get_train_set(self):
        return self.trainloader

    def get_valid_set(self):
        return self.validloader

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader) if self.testloader is not None else 0

    def get_n_classes(self):
        return self.nclasses

    def get_xentropy_class_string(self, idx):
        orig_id = self.learning_map_inv[idx]
        return self.labels[orig_id]

    def to_original(self, label):
        return self.DATASET_TYPE.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        return self.DATASET_TYPE.map(label, self.learning_map)

    def to_color(self, label):
        label = self.DATASET_TYPE.map(label, self.learning_map_inv)
        return self.DATASET_TYPE.map(label, self.color_map)
