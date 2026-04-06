import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SemanticKitti(Dataset):
    """
    極座標 (Polar) BEV (.pt) を読み込むデータセット
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

        # ★ 読み込み先を Polar 用のフォルダに変更
        self.scan_files = []
        for seq in self.sequences:
            bev_path = os.path.join(self.root, seq, "polar_512_8ch")
            if os.path.exists(bev_path):
                scans = [os.path.join(bev_path, f) for f in sorted(os.listdir(bev_path)) if f.endswith(".pt")]
                self.scan_files += scans

        if skip:
            self.scan_files = self.scan_files[::skip]

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        pt_file = self.scan_files[index]

        data = torch.load(pt_file, weights_only=True)
        
        proj_tensor = data['proj_tensor'] # [5, 512, 512]
        mask_t = data['mask_t']           # [1, 512, 512]
        labels_t = data['labels_t']       # [512, 512]

        # --- 安全クリッピング付き正規化 (Polar BEV 5ch仕様) ---
        # ch 0: max_z (高さの最大値。地面付近-2m 〜 建物10mくらいまで)
        proj_tensor[0] = torch.clamp(proj_tensor[0], -5.0, 15.0)
        proj_tensor[0] = (proj_tensor[0] - 1.0) / 5.0
        
        # ch 1: mean_z (高さの平均値)
        proj_tensor[1] = torch.clamp(proj_tensor[1], -5.0, 15.0)
        proj_tensor[1] = (proj_tensor[1] - 1.0) / 5.0
        
        # ch 2: max_r (反射強度。0〜1なのでそのままか、少しクリップ)
        max_r = torch.max(proj_tensor[2])
        if max_r > 0.0:
            proj_tensor[2] = proj_tensor[2] / max_r
        proj_tensor[2] = torch.clamp(proj_tensor[2], 0.0, 1.0)
        
        # ch 3: density (密度。すでに laserscan.py で counts/100 されているので0〜数倍)
        proj_tensor[3] = torch.clamp(proj_tensor[3], 0.0, 5.0) / 5.0
        
        # ch 4: z_diff (高さの差。平坦0m 〜 建物数m)
        proj_tensor[4] = torch.clamp(proj_tensor[4], 0.0, 10.0) / 10.0

        # ch 5: x_diff (X方向の広がり)
        proj_tensor[5] = torch.clamp(proj_tensor[5], 0.0, 10.0) / 10.0

        # ch 6: y_diff (Y方向の広がり)
        proj_tensor[6] = torch.clamp(proj_tensor[6], 0.0, 10.0) / 10.0

        # ★★★ 極座標 (Polar) 専用の Data Augmentation ★★★
        if self.is_train:
            # 1. Azimuth Roll (ヨー回転のシミュレート)
            # 画像を横軸(角度)に沿ってランダムにスライドさせる。はみ出た部分は反対側からループして戻ってくる。
            roll_shift = torch.randint(0, proj_tensor.shape[2], (1,)).item()
            if roll_shift > 0:
                proj_tensor = torch.roll(proj_tensor, shifts=roll_shift, dims=2)
                mask_t = torch.roll(mask_t, shifts=roll_shift, dims=2)
                labels_t = torch.roll(labels_t, shifts=roll_shift, dims=1)

            # 2. ランダム水平反転 (50%の確率で鏡映しにする)
            if torch.rand(1) > 0.5:
                proj_tensor = torch.flip(proj_tensor, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                labels_t = torch.flip(labels_t, dims=[1])
                
            # ※ 上下反転と90度回転は、極座標の物理法則を壊すため削除！

            # 3. マイルドな DropBlock に戻す (本来の形を学ばせるため)
            # ★ 確率50%で、10%の点群だけを隠す
            if torch.rand(1) > 0.5:
                drop_mask = (torch.rand(proj_tensor.shape[1:]) > 0.10).unsqueeze(0).float()
                proj_tensor = proj_tensor * drop_mask
                mask_t = mask_t * drop_mask

            # # 4. Feature Jittering (特徴量のガウシアンノイズ)
            # # ★ 確率50%で、すべての特徴量にランダムな微小ノイズを加える
            # if torch.rand(1) > 0.5:
            #     # mean=0, std=0.02 のノイズを作成
            #     noise = torch.randn_like(proj_tensor) * 0.02
            #     # ノイズを足す（マスクされている真空地帯にはノイズを乗せない）
            #     proj_tensor = (proj_tensor + noise) * mask_t

        dummy_list = []
        dummy_tensor = torch.tensor(0)

        path_norm = os.path.normpath(pt_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".pt", ".label")

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