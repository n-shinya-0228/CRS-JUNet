#!/usr/bin/env python3
import argparse
import yaml
import os
import torch
import numpy as np
from lib.models import get_model
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.dataset.SemanticKitti import SemanticKitti
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
    model.load_state_dict(checkpoint['model'])
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
        gt=True
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    img_W = ARCH["dataset"]["sensor"]["img_prop"]["width"]

    with torch.no_grad():               #勾配計算オフ
        for (
            in_vol, _, _, unproj_labels, path_seq, path_name,             #in_vol（2D 投影入力）、unproj_n_points（元点群数）、proj_x, proj_y（3D→2D 座標配列）
            proj_x, proj_y, _, _, _, _, _, _, unproj_n_points, _          #path_seq/path_name（出力用パス）など
        ) in tqdm(test_loader):

            in_vol = in_vol.cuda()
            output, _ = model(in_vol)
            pred_2d = output.argmax(dim=1).cpu().squeeze().numpy()  # shape (H, W)

            # 投影インデックス計算
            proj_x_np = proj_x[0][:unproj_n_points].cpu().numpy().astype(np.int32)     #proj_x, proj_y：各 3D 点が投影された 2D 座標リスト
            proj_y_np = proj_y[0][:unproj_n_points].cpu().numpy().astype(np.int32)
            proj_idx = proj_y_np * img_W + proj_x_np
            pred_flat = pred_2d.flatten()

            # 出力：点群の数に対応するよう初期化
            final_pred = np.zeros(unproj_n_points, dtype=np.uint32)

            # 有効長に基づいて代入
            valid_len = min(len(proj_idx), len(pred_flat))
            final_pred[:valid_len] = pred_flat[proj_idx[:valid_len]]        #pred_flat[proj_idx] で 2D 予測を対応する 3D 点にコピー



            # ------ KNN 補間を追加 ------
            missing_mask = final_pred == 0                      # ❶ 先に作る
            miss_rate = missing_mask.mean() * 100              # ❷
            if miss_rate > 0:
                print(f"{miss_rate:.2f}% points filled by KNN")

            if np.any(missing_mask):                           # ❸
                # KNN は 2D 投影座標で実施
                ref_xy  = np.column_stack((proj_x_np[~missing_mask],
                                        proj_y_np[~missing_mask]))
                ref_lbl = final_pred[~missing_mask]
                tree    = cKDTree(ref_xy)

                qry_xy  = np.column_stack((proj_x_np[missing_mask],
                                        proj_y_np[missing_mask]))
                _, idxs = tree.query(qry_xy, k=3, workers=-1)
                voted   = ref_lbl[idxs]
                maj     = np.apply_along_axis(
                            lambda row: np.bincount(row,
                                                    minlength=ref_lbl.max()+1).argmax(),
                            1, voted)
                final_pred[missing_mask] = maj.astype(np.uint32)
            # ------ ここまで ------

    

            # ラベルをSemanticKITTI形式に戻す
            final_pred = remap_to_original_labels(final_pred, DATA["learning_map_inv"])

            # Save
            save_file = os.path.join(
                FLAGS.save_path,
                "sequences",
                path_seq[0],
                "predictions",
                path_name[0]
            )
            final_pred.tofile(save_file)

if __name__ == '__main__':
    main()
