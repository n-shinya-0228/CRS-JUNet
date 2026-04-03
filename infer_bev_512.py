#!/usr/bin/env python3
import argparse
import yaml
import os
import torch
import numpy as np
from lib.models import get_model
from tqdm import tqdm
from scipy.spatial import cKDTree

# ★ BEV投影の心臓部（学習時と同じもの）をインポート
from lib.utils.laserscan_BEV10 import SemLaserScan

def remap_to_original_labels(predictions, learning_map_inv):
    """モデル出力のラベルをSemanticKITTIの元のラベルIDに戻す"""
    lut = np.zeros(256, dtype=np.uint32)
    for k, v in learning_map_inv.items():
        lut[k] = v
    return lut[predictions]

def main():
    parser = argparse.ArgumentParser("./infer_bev_512v3.py")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--arch_cfg', type=str, required=True)
    parser.add_argument('--data_cfg', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='predictions')
    FLAGS = parser.parse_args()

    # Load config files
    with open(FLAGS.arch_cfg, 'r') as f:
        ARCH = yaml.safe_load(f)
    with open(FLAGS.data_cfg, 'r') as f:
        DATA = yaml.safe_load(f)

    # Load model
    model = get_model(ARCH['model']['name'])(
        ARCH['model']['in_channels'],
        len(DATA["learning_map_inv"]),
        ARCH['model']['dropout']
    )
    checkpoint = torch.load(FLAGS.pretrained, weights_only=False)
    
    # ★ `_orig_mod.` のクリーニング処理
    state_dict = checkpoint['model']
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        clean_state_dict[new_key] = v

    model.load_state_dict(clean_state_dict)
    model = model.eval().cuda()

    # Output folders & Sequences
    sequences = DATA["split"][FLAGS.split]
    sequences = [f"{int(s):02d}" for s in sequences]

    for seq in sequences:
        os.makedirs(os.path.join(FLAGS.save_path, "sequences", seq, "predictions"), exist_ok=True)
        
        vpath = os.path.join(FLAGS.dataset, "sequences", seq, "velodyne")
        if not os.path.exists(vpath):
            continue
            
        scan_files = sorted([f for f in os.listdir(vpath) if f.endswith(".bin")])
        print(f"Processing Sequence {seq} (Total: {len(scan_files)} scans)")

        with torch.no_grad():
            for scan_file in tqdm(scan_files):
                bin_path = os.path.join(vpath, scan_file)
                
                # 1. BEV画像の生成（学習時と全く同じ処理）
                scan = SemLaserScan(DATA["color_map"], project=True)
                scan.open_scan(bin_path)
                
                # 正規化処理（データセットクラスに書いてあったものと同じ）
                pseudo_image_np = scan.pseudo_image.transpose(2, 0, 1) # [4, 512, 512]
                proj_tensor = torch.from_numpy(pseudo_image_np).float()
                proj_tensor[0] = (proj_tensor[0] - 1.0) / 3.0
                proj_tensor[1] = (proj_tensor[1] + 0.5) / 2.0
                proj_tensor[2] = proj_tensor[2] / (torch.max(proj_tensor[2]) + 1e-5) 
                proj_tensor[3] = torch.clamp(proj_tensor[3], 0.0, 5.0) / 5.0

                # ★ この下にもう1行、追加した高低差チャネルの正規化を足す必要があります
                proj_tensor[4] = proj_tensor[4] / 3.0  # (0〜3m程度と想定して3で割る)

                # ★ 1. マスク(1チャネル)を作成
                mask_t = torch.from_numpy(scan.proj_mask).unsqueeze(0).float() # [1, 512, 512]
                
                # ★ 2. 画像(4チャネル)と合体させて、5チャネルのテンソルを作る！
                combined_tensor = torch.cat([proj_tensor, mask_t], dim=0)      # [5, 512, 512]
                
                # ★ 3. バッチ次元を追加してGPUへ
                in_vol = combined_tensor.unsqueeze(0).cuda()                   # [1, 5, 512, 512]
                
                # 2. モデル推論
                outputs = model(in_vol)
                # ★ 戻り値が辞書(dict)の場合、メインの出力だけを取り出す
                if isinstance(outputs, dict):
                    if 'out' in outputs:
                        main_out = outputs['out']
                    elif 'final_logits' in outputs:
                        main_out = outputs['final_logits']
                    elif 'logits' in outputs:
                        main_out = outputs['logits']
                    else:
                        # 該当する名前がなければ、一番最初の結果を使う
                        main_out = list(outputs.values())[0]
                elif isinstance(outputs, (tuple, list)):
                    main_out = outputs[0]
                else:
                    main_out = outputs
                
                pred_2d = main_out.argmax(dim=1).cpu().squeeze().numpy()  # shape (512, 512)
                # 3. 逆投影（BEVのピクセルから元の3D点群へラベルを戻す）
                # scan.proj_idx は「各3D点がBEV画像のどのXYピクセルに入ったか」のリスト（N行2列）
                # ※laserscan_BEV1.py の仕様に依存しますが、通常は [x_idx, y_idx] が入っています
                unproj_n_points = scan.points.shape[0]
                final_pred = np.zeros(unproj_n_points, dtype=np.uint32)
                
                # proj_idx を使って、2Dの推論結果を1Dの点群リストにマッピング
                # ★ scan.proj_x と scan.proj_y から直接取得し、整数に変換して平坦化する
                proj_x = scan.proj_x.flatten().astype(np.int32)
                proj_y = scan.proj_y.flatten().astype(np.int32)
                
                # 範囲外のアクセスを防ぐためのマスク
                valid_mask = (proj_x >= 0) & (proj_x < 512) & (proj_y >= 0) & (proj_y < 512)
                
                # 有効な点にラベルを割り当て
                final_pred[valid_mask] = pred_2d[proj_y[valid_mask], proj_x[valid_mask]]

                # 4. 改良版KNN補完 (k=1, 距離制限付き)
                missing_mask = final_pred == 0
                if np.any(missing_mask):
                    ref_xy = scan.points[~missing_mask, :2] 
                    ref_lbl = final_pred[~missing_mask]
                    
                    if len(ref_lbl) > 0: 
                        tree = cKDTree(ref_xy)
                        qry_xy = scan.points[missing_mask, :2]
                        
                        # ★ k=1 に変更し、0.5メートル(50cm)以内の点だけを参照する制限を追加！
                        dists, idxs = tree.query(qry_xy, k=1, distance_upper_bound=0.5, workers=-1)
                        
                        # 制限距離内（dists が inf でない）の有効な点だけを抽出
                        valid_knn = dists != np.inf
                        
                        # missing_mask のインデックスを取得し、有効な箇所だけを上書きする
                        missing_indices = np.where(missing_mask)[0]
                        valid_missing_indices = missing_indices[valid_knn]
                        
                        final_pred[valid_missing_indices] = ref_lbl[idxs[valid_knn]]

                # 5. ラベルをSemanticKITTI形式（250番台など）に戻す
                final_pred = remap_to_original_labels(final_pred, DATA["learning_map_inv"])

                # 6. 保存
                save_file = os.path.join(
                    FLAGS.save_path, "sequences", seq, "predictions", scan_file.replace(".bin", ".label")
                )
                final_pred.tofile(save_file)

if __name__ == '__main__':
    main()