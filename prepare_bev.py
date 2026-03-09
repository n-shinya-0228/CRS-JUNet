# prepare_bev.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import argparse

# 既存のモジュールをインポート
from lib.utils.laserscan_BEV2 import SemLaserScan
from lib.dataset.SemanticKitti_BEV1 import _build_lut

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SemanticKitti/', help='データセットのルート')
    parser.add_argument('--data_cfg', type=str, default='config/labels/semantic-kitti.yaml', help='YAMLパス')
    args = parser.parse_args()

    # YAMLから設定を読み込む
    with open(args.data_cfg, 'r') as f:
        data_cfg = yaml.safe_load(f)

    color_map = data_cfg["color_map"]
    learning_map = data_cfg["learning_map"]
    lut_map = _build_lut(learning_map)

    # 処理するシーケンス (train, valid, test)
    sequences = data_cfg["split"]["train"] + data_cfg["split"]["valid"] + data_cfg["split"]["test"]
    sequences = [f"{int(s):02d}" for s in sequences]

    print("=== BEV事前計算を開始します ===")
    
    for seq in sequences:
        vpath = os.path.join(args.dataset, "sequences", seq, "velodyne")
        lpath = os.path.join(args.dataset, "sequences", seq, "labels")
        
        # 保存先フォルダを作成（sequences/00/bev_512 など）
        save_path = os.path.join(args.dataset, "sequences", seq, "bev_512_6ch")
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(vpath):
            continue

        scan_files = sorted([f for f in os.listdir(vpath) if f.endswith(".bin")])
        
        print(f"Processing Sequence {seq} ...")
        for scan_file in tqdm(scan_files):
            bin_path = os.path.join(vpath, scan_file)
            label_path = os.path.join(lpath, scan_file.replace(".bin", ".label"))
            
            # 保存ファイル名
            save_file = os.path.join(save_path, scan_file.replace(".bin", ".pt"))
            if os.path.exists(save_file):
                continue # 既に作成済みの場合はスキップ

            # 1. laserscan4.py で読み込み & BEV投影
            scan = SemLaserScan(color_map, project=True)
            scan.open_scan(bin_path)
            
            has_label = os.path.exists(label_path)
            if has_label:
                scan.open_label(label_path)
                scan.proj_sem_label = lut_map[scan.proj_sem_label] # 学習用IDに変換
            else:
                scan.proj_sem_label = np.zeros((scan.proj_H, scan.proj_W), dtype=np.int32)

            # 2. テンソル化
            pseudo_image_np = scan.pseudo_image.transpose(2, 0, 1) # [4, H, W]
            proj_tensor = torch.from_numpy(pseudo_image_np).float()
            mask_t = torch.from_numpy(scan.proj_mask).unsqueeze(0).float() # [1, H, W]
            labels_t = torch.from_numpy(scan.proj_sem_label).long() # [H, W]

            # 3. 1つの辞書にまとめて保存 (ディスク容量節約のため圧縮は不要、読み込み速度重視)
            save_dict = {
                'proj_tensor': proj_tensor.clone(),
                'mask_t': mask_t.clone(),
                'labels_t': labels_t.clone()
            }
            torch.save(save_dict, save_file)

if __name__ == '__main__':
    main()