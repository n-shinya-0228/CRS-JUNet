import os
import numpy as np
import glob
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# あなたの作ったパーサーを読み込む
from lib.utils.laserscan_Polar3 import SemLaserScan

def build_copy_paste_database():
    dataset_path = "SemanticKitti/sequences/"
    output_dir = "copy_paste_database/"
    os.makedirs(output_dir, exist_ok=True)

    # 抽出したいレアクラスのID（SemanticKITTIの元ラベル）
    # 11: bicycle, 15: motorcycle, 18: truck, 20: other-vehicle, 30: person など
    target_classes = [11, 15, 18, 20, 30, 31, 32, 253, 254, 255, 257, 259]

    # 学習用シーケンス（00〜07, 09, 10など）を指定
    train_seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]

    # パーサーの初期化（投影は不要なので project=False）
    scan = SemLaserScan(project=False)

    saved_count = 0

    for seq in train_seqs:
        print(f"Processing Sequence {seq}...")
        bin_paths = sorted(glob.glob(os.path.join(dataset_path, seq, "velodyne", "*.bin")))
        label_paths = sorted(glob.glob(os.path.join(dataset_path, seq, "labels", "*.label")))

        for bin_path, label_path in tqdm(zip(bin_paths, label_paths), total=len(bin_paths)):
            
            # 1. あなたのクラスを使ってデータを読み込む
            scan.open_scan(bin_path)
            scan.open_label(label_path)

            points = scan.points         # (N, 3)
            remissions = scan.remissions # (N, 1)
            labels = scan.sem_label      # (N, 1)

            # 2. ターゲットとなるレアクラスの点だけをマスク（抽出）する
            mask = np.isin(labels.flatten(), target_classes)
            
            if not np.any(mask):
                continue # レア物体が1つもないフレームはスキップ
            
            target_pts = points[mask]
            target_rems = remissions[mask]
            target_lbls = labels[mask]

            # 3. 抽出した点群を「物体ごと」に分割する（DBSCANクラスタリング）
            # eps=0.5: 50cm以内の点は「同じ物体」とみなす
            # min_samples=10: 最低でも10点以上集まっていないとノイズとして捨てる
            clustering = DBSCAN(eps=0.5, min_samples=10).fit(target_pts)
            
            unique_ids = np.unique(clustering.labels_)

            # 4. 物体ごとに別々のファイルとして保存する
            for obj_id in unique_ids:
                if obj_id == -1:
                    continue # ノイズは無視

                obj_mask = (clustering.labels_ == obj_id)
                obj_points = target_pts[obj_mask]
                
                # ★ ここを修正！縦長の2次元配列 (N, 1) に変換してあげる
                obj_rem = target_rems[obj_mask].reshape(-1, 1)
                obj_lbl = target_lbls[obj_mask].reshape(-1, 1)

                # 物体の中心を (0,0,0) に移動させておく（後で貼り付けやすくするため）
                center = np.mean(obj_points, axis=0)
                obj_points -= center

                # [X, Y, Z, 反射強度, ラベル] の5次元配列にまとめる
                obj_data = np.concatenate([obj_points, obj_rem, obj_lbl], axis=1).astype(np.float32)

                # クラス名と連番をつけて .npy として保存
                class_id = int(obj_lbl[0][0])
                save_path = os.path.join(output_dir, f"class_{class_id}_{saved_count:06d}.npy")
                np.save(save_path, obj_data)
                
                saved_count += 1

    print(f"✅ データベース作成完了！ 合計 {saved_count} 個の物体を保存しました。")

if __name__ == "__main__":
    build_copy_paste_database()