import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.laserscan3 import SemLaserScan

def _build_lut(mapdict: dict) -> np.ndarray:
    maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for k, v in mapdict.items():
        lut[k] = v
    return lut

class SemanticKittiBEV(Dataset):
    """
    LiDAR点群を鳥瞰図（Bird's Eye View）に投影するデータセット。
    
    デフォルト設定:
    - 前方(X): 0m 〜 50m
    - 左右(Y): -25m 〜 25m
    - 高さ(Z): -3m 〜 3m （地面の少し下から車のルーフあたりまで）
    - 解像度: 0.1m / ピクセル
    -> 結果として 500 x 500 ピクセルのBEV画像が生成されます。
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
        self.max_points = max_points
        self.gt = gt

        # ==========================================
        # BEV用のパラメータ設定（適宜変更してください）
        # ==========================================
        self.res = 0.1  # 1ピクセルあたり0.1m
        self.x_range = (0.0, 50.0)   # LiDARより前方を抽出
        self.y_range = (-25.0, 25.0) # 左右25mずつ
        self.z_range = (-3.0, 3.0)   # 不要な高所や地下ノイズをカット

        self.grid_W = int((self.y_range[1] - self.y_range[0]) / self.res) # 500
        self.grid_H = int((self.x_range[1] - self.x_range[0]) / self.res) # 500

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

        # LaserScanでの読み込み（投影はここでは行わず、生データを取得）
        scan = SemLaserScan(self.color_map, project=False)
        scan.open_scan(scan_file)
        
        points = scan.points  # (N, 3)
        remissions = scan.remissions  # (N,)
        
        if self.gt:
            scan.open_label(label_file)
            labels = self.lut_map[scan.sem_label]  # 学習用IDに変換済みのラベル
        else:
            labels = None

        # ==========================================
        # 1. 指定範囲外の点群をカット (Cropping)
        # ==========================================
        mask = (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1]) & \
               (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1]) & \
               (points[:, 2] >= self.z_range[0]) & (points[:, 2] < self.z_range[1])

        points = points[mask]
        remissions = remissions[mask]
        if self.gt:
            labels = labels[mask]

        # ==========================================
        # 2. X, Y 座標をグリッド（ピクセル）のインデックスに変換
        # ==========================================
        grid_x = np.floor((points[:, 0] - self.x_range[0]) / self.res).astype(np.int32)
        grid_y = np.floor((points[:, 1] - self.y_range[0]) / self.res).astype(np.int32)

        # ==========================================
        # 3. Z座標（高さ）でソート
        # ※ これにより、同じピクセルに複数の点が落ちた場合、
        # 配列の後ろにある「一番高い点」が上書きされてBEV画像に残ります。
        # ==========================================
        sort_idx = np.argsort(points[:, 2])
        grid_x = grid_x[sort_idx]
        grid_y = grid_y[sort_idx]
        points = points[sort_idx]
        remissions = remissions[sort_idx]
        if self.gt:
            labels = labels[sort_idx]

        # ==========================================
        # 4. BEV画像の作成 (チャンネルの組み立て)
        # ==========================================
        # ch0: Z座標 (一番高い点)
        bev_z = np.full((self.grid_H, self.grid_W), self.z_range[0], dtype=np.float32)
        # ch1: 反射強度 (一番高い点の強度)
        bev_rem = np.zeros((self.grid_H, self.grid_W), dtype=np.float32)
        # ch2: 観測マスク (点が存在すれば 1.0)
        bev_mask = np.zeros((self.grid_H, self.grid_W), dtype=np.float32)
        
        if self.gt:
            bev_labels = np.zeros((self.grid_H, self.grid_W), dtype=np.int32)

        # インデックスを使って一括で配列に書き込み
        bev_z[grid_x, grid_y] = points[:, 2]
        bev_rem[grid_x, grid_y] = remissions
        bev_mask[grid_x, grid_y] = 1.0
        if self.gt:
            bev_labels[grid_x, grid_y] = labels

        # ==========================================
        # 5. 正規化とテンソル化
        # ==========================================
        # Z座標を 0.0 ~ 1.0 に正規化
        bev_z_norm = (bev_z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        
        # 3チャンネル構成: [Height(Z), Remission, Mask]
        img3 = np.stack([bev_z_norm, bev_rem, bev_mask], axis=0)
        proj_tensor = torch.from_numpy(img3).float()
        
        mask_t = torch.from_numpy(bev_mask).unsqueeze(0).float()
        
        # Parserとの互換性を保つため、ダミー変数を含めて返す（Trainer側のunpackエラーを防ぐため）
        dummy_list = []
        dummy_tensor = torch.zeros(1)
        
        if self.gt:
            labels_t = torch.from_numpy(bev_labels).long()
        else:
            labels_t = torch.zeros((self.grid_H, self.grid_W)).long()

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return (
            proj_tensor,      # [3, H, W] (BEV入力画像)
            mask_t,           # [1, H, W] (観測マスク)
            labels_t,         # [H, W] (BEV用セマンティックラベル)
            dummy_list,       # unproj_labels (以下Parser互換用ダミー)
            path_seq, path_name,
            dummy_list, dummy_list, dummy_tensor, dummy_list, 
            dummy_list, dummy_list, dummy_tensor, dummy_list, 
            torch.tensor(0), dummy_tensor
        )