import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SemanticKitti(Dataset):
    """
    事前計算された BEV (.pt) を超高速で読み込むデータセット
    """
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

        # 事前計算した 'bev' フォルダ内の .pt ファイルを列挙
        self.scan_files = []
        for seq in self.sequences:
            bev_path = os.path.join(self.root, seq, "bev_512")
            if os.path.exists(bev_path):
                scans = [os.path.join(bev_path, f) for f in sorted(os.listdir(bev_path)) if f.endswith(".pt")]
                self.scan_files += scans

        if skip:
            self.scan_files = self.scan_files[::skip]

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        pt_file = self.scan_files[index]

        # 1. 事前計算されたテンソルを爆速でロード
        data = torch.load(pt_file, weights_only=True)
        
        proj_tensor = data['proj_tensor'] # [4, 256, 256]
        mask_t = data['mask_t']           # [1, 256, 256]
        labels_t = data['labels_t']       # [256, 256]

        # ★ 異常値によるNaNを完全に防ぐための安全対策（クリッピングと正規化）
        # ch 0: max_z (高さの最大値)
        proj_tensor[0] = torch.clamp(proj_tensor[0], -5.0, 15.0)
        proj_tensor[0] = (proj_tensor[0] - 1.0) / 5.0
        
        # ch 1: mean_z (高さの平均値)
        proj_tensor[1] = torch.clamp(proj_tensor[1], -5.0, 15.0)
        proj_tensor[1] = (proj_tensor[1] - 1.0) / 5.0
        
        # ch 2: 反射強度 (ゼロ割りを絶対に防ぐ)
        max_r = torch.max(proj_tensor[2])
        if max_r > 0.0:
            proj_tensor[2] = proj_tensor[2] / max_r
        proj_tensor[2] = torch.clamp(proj_tensor[2], 0.0, 1.0)
        
        # ch 3: Density (点の密度)
        proj_tensor[3] = torch.clamp(proj_tensor[3], 0.0, 5.0) / 5.0

        # ★ ch 4: 高低差 (z_diff)
        proj_tensor[4] = torch.clamp(proj_tensor[4], 0.0, 10.0) / 10.0

        if self.is_train:
            # 1. ランダム水平反転 (50%の確率)
            if torch.rand(1) > 0.5:
                proj_tensor = torch.flip(proj_tensor, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                labels_t = torch.flip(labels_t, dims=[1])

            # 2. ランダム垂直反転 (50%の確率)
            if torch.rand(1) > 0.5:
                proj_tensor = torch.flip(proj_tensor, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
                labels_t = torch.flip(labels_t, dims=[0])

            # 3. ランダム90度回転 (0度, 90度, 180度, 270度)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                proj_tensor = torch.rot90(proj_tensor, k, [1, 2])
                mask_t = torch.rot90(mask_t, k, [1, 2])
                labels_t = torch.rot90(labels_t, k, [0, 1])

            # 点群消去は「超マイルド」にする（Cartesianのピクセルは貴重なため）
            # ★ 確率50%で、たった 5% だけ消す
            if torch.rand(1) > 0.5:
                drop_mask = (torch.rand(proj_tensor.shape[1:]) > 0.05).unsqueeze(0).float()
                proj_tensor = proj_tensor * drop_mask
                mask_t = mask_t * drop_mask

            # Feature Jittering (特徴量のガウシアンノイズ) はそのまま残す！
            # これが直交座標系における最強の正則化になります
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(proj_tensor) * 0.02
                proj_tensor = (proj_tensor + noise) * mask_t

        # Parserの戻り値と合わせるためのダミー変数
        dummy_list = []
        dummy_tensor = torch.tensor(0)

        path_norm = os.path.normpath(pt_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".pt", ".label")

        return (
            proj_tensor,      # [4, H, W]
            mask_t,           # [1, H, W]
            labels_t,         # [H, W]
            dummy_list,       # unproj_labels
            path_seq, path_name,
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