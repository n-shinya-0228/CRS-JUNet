import os
import torch
import numpy as np
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

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        pt_file = self.scan_files[index]

        with gzip.open(pt_file, 'rb') as f:
            data = torch.load(f, weights_only=True)

        proj_tensor = data['proj_tensor'].float() # [7, H, W]
        mask_t = data['mask_t'].float()           # [1, H, W]
        labels_t = data['labels_t'].long()        # [H, W]

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