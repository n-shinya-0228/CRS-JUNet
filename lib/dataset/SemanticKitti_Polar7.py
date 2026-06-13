import os
import torch
import numpy as np
import random
import gzip
from torch.utils.data import Dataset

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

        self.scan_files = []
        for seq in self.sequences:
            bev_path = os.path.join(self.root, seq, "polar_512_prlb")
            if os.path.exists(bev_path):
                scans = [os.path.join(bev_path, f) for f in sorted(os.listdir(bev_path)) if f.endswith(".pt")]
                self.scan_files += scans

        if skip:
            self.scan_files = self.scan_files[::skip]

        self.bev_files = self.scan_files

    def __len__(self):
        return len(self.scan_files)
    
    def get_context_classes(self, obj_cls):
        """
        obj_cls: int, learning label
        return: torch.Tensor of allowed context classes
        """

        # bicycle, motorcycle, truck, other-vehicle
        # 車両系は road / parking に貼る
        if obj_cls in [2, 3, 4, 5]:
            return torch.tensor([9, 10], dtype=torch.long)

        # person, bicyclist, motorcyclist
        # 人・乗車人物系は sidewalk / road に貼る
        elif obj_cls in [6, 7, 8]:
            return torch.tensor([11, 9], dtype=torch.long)

        # pole, traffic-sign
        # 静的な小物体は sidewalk / road / terrain に貼る
        elif obj_cls in [18, 19]:
            return torch.tensor([11, 9, 17], dtype=torch.long)

        # fallback
        else:
            return torch.tensor([9, 10, 11], dtype=torch.long)

    def get_occlusion_threshold(self, obj_cls):
        # bicycle, motorcycle, person, bicyclist, motorcyclist
        if obj_cls in [2, 3, 6, 7, 8]:
            return 0.2

        # truck, other-vehicle
        elif obj_cls in [4, 5]:
            return 0.3

        # pole, traffic-sign
        elif obj_cls in [18, 19]:
            return 0.4

        else:
            return 0.3
    
    def apply_bev_copy_paste(self, proj_tensor, mask_t, labels_t):
        paste_mask_t = torch.zeros_like(mask_t).float()

        if len(self.bev_files) == 0:
            return proj_tensor, mask_t, labels_t, paste_mask_t

        # carは除外。少数クラス・小物体中心
        object_classes = torch.tensor([2, 3, 4, 5, 6, 7, 8, 18, 19], dtype=torch.long)

        H, W = labels_t.shape
        crop_h = 64
        crop_w = 64

        for attempt in range(12):
            src_file = random.choice(self.bev_files)

            try:
                with gzip.open(src_file, "rb") as f:
                    src = torch.load(f, map_location="cpu", weights_only=True)
            except Exception:
                continue

            src_feat = src["proj_tensor"].float()
            src_mask = src["mask_t"].float()
            src_label = src["labels_t"].long()

            obj_mask = torch.isin(src_label, object_classes) & (src_mask.squeeze(0) > 0)

            ys, xs = torch.where(obj_mask)
            if ys.numel() == 0:
                continue

            k = torch.randint(0, ys.numel(), (1,)).item()
            cy = ys[k].item()
            cx = xs[k].item()

            # 選ばれた画素のクラスを取得
            obj_cls = int(src_label[cy, cx].item())

            # クラスごとのcontextを取得
            context_classes = self.get_context_classes(obj_cls)

            y1 = max(0, cy - crop_h // 2)
            x1 = max(0, cx - crop_w // 2)
            y2 = min(H, y1 + crop_h)
            x2 = min(W, x1 + crop_w)

            y1 = max(0, y2 - crop_h)
            x1 = max(0, x2 - crop_w)

            patch_feat = src_feat[:, y1:y2, x1:x2]
            patch_label = src_label[y1:y2, x1:x2]

            # patch内は選ばれたクラスだけ貼る
            # これにより、bicycleを選んだのにpoleも一緒に貼る、みたいな混在を避ける
            patch_obj_mask = ((patch_label == obj_cls) & (src_mask.squeeze(0)[y1:y2, x1:x2] > 0)).unsqueeze(0).float()

            _, h, w = patch_feat.shape

            if h <= 0 or w <= 0:
                continue

            # 小物体用なので少し緩め
            if patch_obj_mask.sum() < 10:
                continue

            if h >= H or w >= W:
                continue

            # 距離方向は元の位置から大きく変えない
            ty = int(np.clip(y1 + np.random.randint(-20, 21), 0, H - h))

            # 方位方向はランダム
            tx = int(np.random.randint(0, W - w))

            m = patch_obj_mask > 0.5
            m2d = m.squeeze(0)

            target_region = labels_t[ty:ty+h, tx:tx+w]
            target_under_obj = target_region[m2d]

            if target_under_obj.numel() == 0:
                continue

            context_mask = torch.isin(target_under_obj, context_classes)

            # Step2: クラスごとのcontext条件
            if context_mask.float().mean() < 0.1:
                continue

            # ==============================
            # Step3: Occlusion-aware check
            # ==============================
            target_label_region = labels_t[ty:ty+h, tx:tx+w]
            target_mask_region = mask_t[:, ty:ty+h, tx:tx+w]
            
            m = patch_obj_mask > 0.5
            m2d = m.squeeze(0)
            
            occupied = (target_mask_region.squeeze(0) > 0) & m2d
            
            foreground_classes = torch.tensor(
                [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 18, 19],
                dtype=torch.long
                )
                
            target_foreground = torch.isin(target_label_region, foreground_classes) & occupied

            # 貼り付ける物体領域の中で、既存前景と重なる割合を計算
            if m2d.sum() > 0:
                overlap_ratio = target_foreground[m2d].float().mean()
            else:
                continue
                
            occ_th = self.get_occlusion_threshold(obj_cls)
            
            # クラスごとの閾値で判定
            if overlap_ratio > occ_th:
                continue

            #step3 一部だけ隠したcopy&paste
            visible_m2d = m2d & (~target_foreground)
            visible_ratio = visible_m2d.sum().float() / m2d.sum().float()

            if visible_ratio < 0.3:
                continue

            if visible_ratio > 0.95 and torch.rand(1).item() > 0.3:
                continue

            m = visible_m2d.unsqueeze(0)
            

            proj_tensor[:, ty:ty+h, tx:tx+w] = torch.where(
                m.expand_as(patch_feat),
                patch_feat,
                proj_tensor[:, ty:ty+h, tx:tx+w]
            )

            labels_t[ty:ty+h, tx:tx+w] = torch.where(
                m.squeeze(0),
                patch_label,
                labels_t[ty:ty+h, tx:tx+w]
            )

            mask_t[:, ty:ty+h, tx:tx+w] = torch.where(
                m,
                torch.ones_like(mask_t[:, ty:ty+h, tx:tx+w]),
                mask_t[:, ty:ty+h, tx:tx+w]
            )

            paste_mask_t[:, ty:ty+h, tx:tx+w] = torch.where(
                m,
                torch.ones_like(paste_mask_t[:, ty:ty+h, tx:tx+w]),
                paste_mask_t[:, ty:ty+h, tx:tx+w]
            )

            return proj_tensor, mask_t, labels_t, paste_mask_t

        return proj_tensor, mask_t, labels_t, paste_mask_t

    def __getitem__(self, index):
        pt_file = self.scan_files[index]

        with gzip.open(pt_file, 'rb') as f:
            data = torch.load(f, weights_only=True)

        proj_tensor = data['proj_tensor'].float() # [7, H, W]
        mask_t = data['mask_t'].float()           # [1, H, W]
        labels_t = data['labels_t'].long()        # [H, W]

        paste_mask_t = torch.zeros_like(mask_t).float()

        if self.is_train and torch.rand(1) > 0.25:
            num_paste = np.random.randint(1, 4)  # 2〜4回試す
            max_paste_pixels = 1500

            for _ in range(num_paste):
                proj_tensor_new, mask_t_new, labels_t_new, paste_mask_new = self.apply_bev_copy_paste(proj_tensor, mask_t, labels_t)

                if paste_mask_new.sum() > 0:
                    proj_tensor = proj_tensor_new
                    mask_t = mask_t_new
                    labels_t = labels_t_new
                    paste_mask_t = torch.maximum(paste_mask_t, paste_mask_new)

                if paste_mask_t.sum() > max_paste_pixels:
                    break
              
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
                paste_mask_t = torch.roll(paste_mask_t, shifts=roll_shift, dims=2)

            # 2. ランダム水平反転
            if torch.rand(1) > 0.5:
                proj_tensor = torch.flip(proj_tensor, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                labels_t = torch.flip(labels_t, dims=[1])
                paste_mask_t = torch.flip(paste_mask_t, dims=[2])
                
            if torch.rand(1) > 0.5:
                drop_mask = (torch.rand(proj_tensor.shape[1:]) > 0.10).unsqueeze(0).float()
                proj_tensor = proj_tensor * drop_mask
                mask_t = mask_t * drop_mask
                paste_mask_t = paste_mask_t * drop_mask

            # # 4. Feature Jittering 
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
            paste_mask_t, dummy_tensor
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