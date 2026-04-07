# Parser_BEV2.py
import os
import torch
from torch.utils.data import DataLoader

# ★ 直交座標系の SemanticKitti をインポートする (ファイル名に合わせてください)
from .SemanticKitti_BEV13 import SemanticKitti 

def bev_collate_fn(batch):
    """
    SemanticKitti が返すタプルをバッチ化する関数。
    テンソル化すべき最初の3つ（特徴量、マスク、ラベル）だけスタックし、
    残りのダミーやパス情報はリストのまま返す。
    """
    # ★ ここが 5ch (B, 5, H, W) になります
    proj_tensor   = torch.stack([b[0] for b in batch], dim=0)
    # [B, 1, H, W]
    mask_t        = torch.stack([b[1] for b in batch], dim=0)
    # [B, H, W]
    labels_t      = torch.stack([b[2] for b in batch], dim=0)
    
    # 互換性のためのリスト
    unproj_labels = [b[3] for b in batch]
    path_seq      = [b[4] for b in batch]
    path_name     = [b[5] for b in batch]
    
    # trainer.py の unpack に合わせるためのダミー (Noneや0で埋める)
    dummy_list = [None] * len(batch)
    dummy_tensor = torch.zeros(len(batch))
    
    return (
        proj_tensor, mask_t, labels_t, unproj_labels, path_seq, path_name,
        dummy_list, dummy_list, dummy_tensor, dummy_list, 
        dummy_list, dummy_list, dummy_tensor, dummy_list, 
        dummy_tensor, dummy_tensor
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
        
        self.nclasses = len(self.learning_map_inv)

        self.train_sequences = data_cfg["split"]["train"]
        self.valid_sequences = data_cfg["split"]["valid"]
        self.test_sequences  = data_cfg["split"]["test"]

        # ============================================================
        # 1. Train Dataset & Loader
        # ============================================================
        self.train_dataset = SemanticKitti(
            root=self.root, sequences=self.train_sequences, labels=self.labels,
            color_map=self.color_map, learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv, sensor=self.sensor,
            gt=self.gt, is_train=True # ★ Data Augmentation をオンにする
        )

        self.trainloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train,
            num_workers=self.workers, pin_memory=True, drop_last=True,
            collate_fn=bev_collate_fn # ★ 専用の結合関数を使う
        )

        # ============================================================
        # 2. Valid Dataset & Loader
        # ============================================================
        self.valid_dataset = SemanticKitti(
            root=self.root, sequences=self.valid_sequences, labels=self.labels,
            color_map=self.color_map, learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv, sensor=self.sensor,
            gt=self.gt, is_train=False # ★ 検証時は Augmentation オフ
        )

        self.validloader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True, drop_last=False,
            collate_fn=bev_collate_fn
        )

        # ============================================================
        # 3. Test Dataset & Loader
        # ============================================================
        if self.test_sequences:
            self.test_dataset = SemanticKitti(
                root=self.root, sequences=self.test_sequences, labels=self.labels,
                color_map=self.color_map, learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv, sensor=self.sensor,
                gt=False, is_train=False
            )
            self.testloader = DataLoader(
                self.test_dataset, batch_size=1, shuffle=False,
                num_workers=self.workers, pin_memory=True, drop_last=False,
                collate_fn=bev_collate_fn
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