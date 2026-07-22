# Parser_Polar2.py
import os
import torch
from torch.utils.data import DataLoader

from .SemanticKitti_step4_5 import SemanticKitti

def bev_collate_fn(batch):
    """
    SemanticKitti_Polar1 が返す16要素のタプルをバッチ化する関数。
    テンソル化すべき最初の3つ（特徴量、マスク、ラベル）だけスタックし、
    残りのダミーやパス情報はリストのまま返す。
    """
    # ★ ここが 5ch (B, 5, H, W) になります
    proj_tensor   = torch.stack([b[0] for b in batch], dim=0)
    # [B, 1, H, W]
    mask_t        = torch.stack([b[1] for b in batch], dim=0)
    # [B, H, W]
    labels_t      = torch.stack([b[2] for b in batch], dim=0)

    paste_mask_t = torch.stack([b[14] for b in batch], dim=0)
    
    # 互換性のためのリスト
    unproj_labels = [b[3] for b in batch]
    path_seq      = [b[4] for b in batch]
    path_name     = [b[5] for b in batch]
    
    # trainer.py の unpack に合わせるためのダミー (Noneや0で埋める)
    # trainer.py 側で unpack しているのは:
    # in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, proj_range, _, _, _, _, _, _, edge
    
    dummy_list = [None] * len(batch)
    dummy_tensor = torch.zeros(len(batch))
    
    return (
        proj_tensor, mask_t, labels_t, unproj_labels, path_seq, path_name,
        dummy_list, dummy_list, dummy_tensor, dummy_list, 
        dummy_list, dummy_list, dummy_tensor, dummy_list, 
        paste_mask_t, dummy_tensor
    )


class Parser:
    """
    BEV事前計算データ (.pt) を学習ループに供給するモダンなParser
    """
    def __init__(self, root, data_cfg, arch_cfg, gt=True, shuffle_train=True):
        self.root = root
        self.data_cfg = data_cfg
        self.arch_cfg = arch_cfg
        self.gt = gt
        self.shuffle_train = shuffle_train

        self.labels = data_cfg["labels"]
        self.color_map = data_cfg["color_map"]
        self.learning_map = data_cfg["learning_map"]
        self.learning_map_inv = data_cfg["learning_map_inv"]
        
        self.sensor = arch_cfg["dataset"]["sensor"]
        self.batch_size = arch_cfg["train"]["batch_size"]
        self.workers = arch_cfg["train"]["workers"]
        self.copy_paste = bool(arch_cfg["train"].get("copy_paste", True))
        
        self.nclasses = len(self.learning_map_inv)

        self.train_sequences = data_cfg["split"]["train"]
        self.valid_sequences = data_cfg["split"]["valid"]
        self.test_sequences  = data_cfg["split"]["test"]

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=bev_collate_fn,
        )

        if self.workers > 0:
            loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4
            )


        # ============================================================
        # 1. Train Dataset & Loader
        # ============================================================
        self.train_dataset = SemanticKitti(
            root=self.root, sequences=self.train_sequences, labels=self.labels,
            color_map=self.color_map, learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv, sensor=self.sensor,
            gt=self.gt, is_train=True, copy_paste=self.copy_paste # ★ Data Augmentation をオンにする
        )

        self.trainloader = DataLoader(
            self.train_dataset,shuffle=self.shuffle_train,drop_last=True,**loader_kwargs
            )

        # ============================================================
        # 2. Valid Dataset & Loader
        # ============================================================
        self.valid_dataset = SemanticKitti(
            root=self.root, sequences=self.valid_sequences, labels=self.labels,
            color_map=self.color_map, learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv, sensor=self.sensor,
            gt=self.gt, is_train=False, copy_paste=False # ★ 検証時は Augmentation オフ
        )

        self.validloader = DataLoader(
            self.valid_dataset,shuffle=False,drop_last=False,**loader_kwargs
            )

        # ============================================================
        # 3. Test Dataset & Loader
        # ============================================================
        test_loader_kwargs = dict(
            batch_size=1,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=bev_collate_fn,
        )

        if self.workers > 0:
            test_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4
            )

        if self.test_sequences:
            self.test_dataset = SemanticKitti(
                root=self.root, sequences=self.test_sequences, labels=self.labels,
                color_map=self.color_map, learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv, sensor=self.sensor,
                gt=False, is_train=False, copy_paste=False
            )
            self.testloader = DataLoader(
                self.test_dataset,shuffle=False,drop_last=False,**test_loader_kwargs
                )
        else:
            self.testloader = None

    # ===== Interface (モダンな呼び出し方) =====
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

    def get_n_classes(self):
        return self.nclasses

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        label = SemanticKitti.map(label, self.learning_map_inv)
        return SemanticKitti.map(label, self.color_map)
