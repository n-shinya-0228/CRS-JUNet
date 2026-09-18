#!/usr/bin/env python3
import argparse
import yaml
import os
import torch
import numpy as np
from lib.models import get_model
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.dataset.SemanticKitti3 import SemanticKitti
import torch.nn.functional as F          # 既にあれば不要
from scipy.spatial import cKDTree        # 追加：高速 k‑d 木


def remap_to_original_labels(predictions, learning_map_inv):
    """モデル出力のラベルをSemanticKITTIの元のラベルIDに戻す"""
    lut = np.zeros(256, dtype=np.uint32)
    for k, v in learning_map_inv.items():
        lut[k] = v
    return lut[predictions]

def main():
    parser = argparse.ArgumentParser("./modified_infer.py")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--arch_cfg', type=str, required=True)
    parser.add_argument('--data_cfg', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='predictions')
    FLAGS = parser.parse_args()                                      #FLAGS に格納(FLAGS.pretrainedのように使える)

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
    checkpoint = torch.load(FLAGS.pretrained, weights_only=False)               #torch.load でチェックポイントを読み込み

    # ★ 追加: `_orig_mod.` という余分な文字を削って、元の名前に戻す！
    state_dict = checkpoint['model']
    clean_state_dict = {}

    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')  # _orig_mod. を消す
        clean_state_dict[new_key] = v

    model.load_state_dict(clean_state_dict) # ✅ 綺麗にした辞書を読み込む

    model = model.eval().cuda()                                                 #eval() にして推論モード、.cuda() で GPU へ

    # Output folders
    for seq in DATA["split"][FLAGS.split]:
        seq_id = f"{int(seq):02d}"
        os.makedirs(os.path.join(FLAGS.save_path, "sequences", seq_id, "predictions"), exist_ok=True)

    # Load dataset
    test_dataset = SemanticKitti(
        root=FLAGS.dataset,
        sequences=DATA["split"][FLAGS.split],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        gt=False,
        max_gap_row=0,
        max_gap_col=0,
        median_ksize=0
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    img_W = ARCH["dataset"]["sensor"]["img_prop"]["width"]

    with torch.no_grad():
        for (
            in_vol, _, _, unproj_labels, path_seq, path_name,
            proj_x, proj_y, proj_range, unproj_range, _, unproj_xyz, _, _, unproj_n_points, _
        ) in tqdm(test_loader):

            in_vol = in_vol.cuda()
            outputs = model(in_vol)
            if isinstance(outputs, dict):
                if 'out' in outputs:
                    output = outputs['out']
                elif 'logits' in outputs:
                    output = outputs['logits']
                else:
                    output = list(outputs.values())[0]
            elif isinstance(outputs, (tuple, list)):
                output = outputs[0]
            else:
                output = outputs

            # 1. 2D予測を取得
            pred_2d = output.argmax(dim=1).cpu().squeeze().numpy()

            # GPUテンソルをNumpy配列に変換（バッチ次元[0]を外す）
            proj_x_np = proj_x[0][:unproj_n_points].cpu().numpy().astype(np.int32)
            proj_y_np = proj_y[0][:unproj_n_points].cpu().numpy().astype(np.int32)
            unproj_xyz_np = unproj_xyz[0][:unproj_n_points].cpu().numpy()
            unproj_range_np = unproj_range[0][:unproj_n_points].cpu().numpy()
            proj_range_np = proj_range[0].cpu().numpy()

            # 2. とりあえず全点に2Dの色を塗る（一瞬でコピー）
            final_pred = pred_2d[proj_y_np, proj_x_np].astype(np.uint32)

            # 3. 【影のノイズ検出】
            # 「点の本当の距離」と「2D画像のピクセルの距離」を比較する
            pixel_depth = proj_range_np[proj_y_np, proj_x_np]
            
            # 本当の距離のほうが 0.5m 以上遠い点は「手前の物体に隠れた影（オクルージョン）」と判定
            is_shadow = (unproj_range_np - pixel_depth) > 0.5

            # 4. 【超高速 3D KNN】影の点だけ、周囲の「正しい3D点」から色をもらう
            valid_mask = ~is_shadow
            if np.any(is_shadow) and np.any(valid_mask):
                # 正しい点（影じゃない点）だけで3D空間の検索ツリーを作る
                tree = cKDTree(unproj_xyz_np[valid_mask])
                # 影の点に一番近い「正しい点」を1つ（k=1）探す（マルチスレッド workers=-1 で爆速化）
                _, idxs = tree.query(unproj_xyz_np[is_shadow], k=1, workers=-1)
                # 正しい点の色で、影の点を上書きして修復！
                final_pred[is_shadow] = final_pred[valid_mask][idxs]

            # 5. SemanticKITTIの提出用IDに戻す
            final_pred = remap_to_original_labels(final_pred, DATA["learning_map_inv"])

            # 6. 保存
            unproj_pred = final_pred.astype(np.int32)
            
            # ★ ここから追加：SemanticKITTI指定のフォルダ階層に合わせてパスを作る
            save_file = os.path.join(
                FLAGS.save_path,
                "sequences",
                path_seq[0],
                "predictions",
                path_name[0]
            )
            unproj_pred.tofile(save_file)

if __name__ == '__main__':
    main()
