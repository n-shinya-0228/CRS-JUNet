# prepare_bev2.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import argparse
import gc
import gzip  # ★ Gzip圧縮モジュールを追加

# 既存のモジュールをインポート
from lib.utils.laserscan_Polar3 import SemLaserScan

def _build_lut(learning_map):
    maxkey = max(learning_map.keys())
    lut = np.zeros((maxkey + 100,), dtype=np.int32)
    for k, v in learning_map.items():
        lut[k] = v
    return lut

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

    print("=== BEV事前計算を開始します (Gzip圧縮 & 16bit軽量化モード) ===")
    
    for seq in sequences:
        vpath = os.path.join(args.dataset, "sequences", seq, "velodyne")
        lpath = os.path.join(args.dataset, "sequences", seq, "labels")
        
        # 保存先フォルダを作成（sequences/00/polar_512_8ch など）
        save_path = os.path.join(args.dataset, "sequences", seq, "polar_512_8ch")
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

            # 1. laserscan_Polar3.py で読み込み & BEV投影
            scan = SemLaserScan(color_map, project=True)
            scan.open_scan(bin_path)
            
            has_label = os.path.exists(label_path)
            if has_label:
                scan.open_label(label_path)
                # ★ 追加: lut_mapのサイズ(最大インデックス)をはみ出さないように安全装置を入れる
                safe_label = np.clip(scan.proj_sem_label, 0, lut_map.shape[0] - 1)
                scan.proj_sem_label = lut_map[safe_label] # 安全に変換
            else:
                scan.proj_sem_label = np.zeros((scan.proj_H, scan.proj_W), dtype=np.int32)

            # 2. テンソル化 ＆ ★ データ型の縮小（メモリ節約）
            pseudo_image_np = scan.pseudo_image.transpose(2, 0, 1) # [7, H, W]
            
            # ★ 特徴量は float16 (半精度) にして容量半減
            proj_tensor = torch.from_numpy(pseudo_image_np).half() 
            
            # ★ マスクは 0 or 1 なので uint8 (1/4サイズ) に縮小
            mask_t = torch.from_numpy(scan.proj_mask).unsqueeze(0).to(torch.uint8) 
            
            # ★ ラベルは 0~300 程度なので int16 (1/2サイズ) に縮小
            labels_t = torch.from_numpy(scan.proj_sem_label).to(torch.int16) 

            # 3. 1つの辞書にまとめる
            save_dict = {
                'proj_tensor': proj_tensor.clone(),
                'mask_t': mask_t.clone(),
                'labels_t': labels_t.clone()
            }
            
            # ★ 修正: Gzipで強烈に圧縮して保存
            with gzip.open(save_file, 'wb') as f:
                torch.save(save_dict, f)

            # メモリを強制解放する
            del scan, pseudo_image_np, proj_tensor, mask_t, labels_t, save_dict
            gc.collect()

if __name__ == '__main__':
    main()