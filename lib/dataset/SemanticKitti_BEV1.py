import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.laserscan_BEV1 import LaserScan, SemLaserScan  # ★ laserscan.py をインポート

def _build_lut(mapdict: dict) -> np.ndarray:
    maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for k, v in mapdict.items():
        lut[k] = v
    return lut

class SemanticKitti(Dataset):
    """
    laserscan4.py の Pseudo-image (BEV) をそのままニューラルネットに入力するデータセット。
    """
    def __init__(self, root, sequences, labels, color_map,
                 learning_map, learning_map_inv, sensor,
                 max_points=150000, gt=True, skip=0):
        super().__init__()
        self.root = os.path.join(root, "sequences")
        self.sequences = [f"{int(s):02d}" for s in sequences]
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.gt = gt

        self.lut_map = _build_lut(self.learning_map)

        # ファイル列挙
        self.scan_files, self.label_files = [], []
        for seq in self.sequences:
            vpath = os.path.join(self.root, seq, "velodyne")
            lpath = os.path.join(self.root, seq, "labels")
            scans = [os.path.join(vpath, f) for f in sorted(os.listdir(vpath)) if f.endswith(".bin")]
            self.scan_files += scans
            if self.gt:
                lbls = [os.path.join(lpath, f) for f in sorted(os.listdir(lpath)) if f.endswith(".label")]
                self.label_files += lbls

        if skip:
            self.scan_files = self.scan_files[::skip]
            if self.gt:
                self.label_files = self.label_files[::skip]

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        label_file = self.label_files[index] if self.gt else None

        # ==========================================
        # 1. laserscan4.py の機能を使って読み込み＆BEV投影
        # project=True にすると自動で do_pseudo_image_projection() が走る
        # ==========================================
        scan = SemLaserScan(self.color_map, project=True)
        scan.open_scan(scan_file)
        
        if self.gt:
            scan.open_label(label_file)
            # 生のSemanticKITTIラベルを、学習用(0〜19等)のIDにマッピング
            scan.sem_label = self.lut_map[scan.sem_label]
            scan.proj_sem_label = self.lut_map[scan.proj_sem_label]

        # ==========================================
        # 2. BEV特徴量（pseudo_image）の取得
        # laserscan4.py は [H, W, 4] (max_z, mean_z, max_r, density) を作る
        # ==========================================
        # PyTorchのConv2dに入力するため [4, H, W] の形に並べ替える
        pseudo_image_np = scan.pseudo_image.transpose(2, 0, 1) 
        
        # テンソル化
        proj_tensor = torch.from_numpy(pseudo_image_np).float()
        
        # マスク（点が存在するピクセルが 1）
        mask_t = torch.from_numpy(scan.proj_mask).unsqueeze(0).float()

        if self.gt:
            # [H, W] のBEV用正解ラベル
            labels_t = torch.from_numpy(scan.proj_sem_label).long()
        else:
            labels_t = torch.zeros((scan.proj_H, scan.proj_W)).long()

        # Parserの戻り値と合わせるためのダミー変数
        dummy_list = []
        dummy_tensor = torch.tensor(0)

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        # Parser2.py の kitti_collate_stack_only と unpack の形に合わせる
        return (
            proj_tensor,      # [4, H, W] (BEV入力: 4チャネル)
            mask_t,           # [1, H, W] (マスク)
            labels_t,         # [H, W]    (正解ラベル)
            dummy_list,       # unproj_labels
            path_seq, path_name,
            dummy_list, dummy_list, dummy_tensor, dummy_list, 
            dummy_list, dummy_list, dummy_tensor, dummy_list, 
            torch.tensor(0), dummy_tensor
        )
    @staticmethod
    def map(label, mapdict):
        # ラベルIDを学習用IDやカラーコードに変換するための関数
        maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
        
        # 辞書の中身がスカラー(ID)かリスト(RGB)かを判定
        first_val = next(iter(mapdict.values()))
        if isinstance(first_val, list):
            lut = np.zeros((maxkey + 100, len(first_val)), dtype=np.float32)
        else:
            lut = np.zeros((maxkey + 100,), dtype=np.int32)
            
        for k, v in mapdict.items():
            lut[k] = v
            
        return lut[label]