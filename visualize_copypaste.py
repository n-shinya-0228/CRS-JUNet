#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import open3d as o3d

def main():
    parser = argparse.ArgumentParser("SemanticKITTI Copy-Paste Visualizer")
    parser.add_argument('--bin_path', type=str, required=True, help='Path to the .bin file')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the .label file')
    parser.add_argument('--config', type=str, default='config/labels/semantic-kitti.yaml')
    
    # --- Copy-Paste用の追加引数 ---
    parser.add_argument('--paste_npy', type=str, default=None, help='Path to the extracted object .npy file')
    parser.add_argument('--offset', type=str, default='0,0,0', help='X,Y,Z offset for the pasted object (e.g., 5.0,-2.0,0.0)')
    args = parser.parse_args()

    # 1. configからカラーマップを読み込む
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    color_map = cfg['color_map']

    # 2. 点群データの読み込み
    scan = np.fromfile(args.bin_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3] # x, y, z

    # 3. ラベルデータの読み込み
    labels = np.fromfile(args.label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF 

    # 4. ラベルIDをRGBカラーに変換
    colors = np.zeros((len(sem_labels), 3), dtype=np.float64)
    for label_id, bgr in color_map.items():
        rgb = np.array([bgr[2], bgr[1], bgr[0]]) / 255.0
        colors[sem_labels == label_id] = rgb

    # ====================================================
    # 5. Copy-Paste 処理 (Afterシーンの作成)
    # ====================================================
    if args.paste_npy:
        # npyを読み込む (N, 3) または (N, 4) を想定
        paste_data = np.load(args.paste_npy)
        paste_points = paste_data[:, :3] 

        # オフセット（移動量）を適用して空きスペースに配置
        offset = np.array([float(x) for x in args.offset.split(',')])
        paste_points += offset

        # スライド映えするように、追加した物体を「真っ赤」にする
        paste_colors = np.zeros((len(paste_points), 3), dtype=np.float64)
        paste_colors[:, 0] = 1.0  # R=1, G=0, B=0

        # 元のシーンと結合
        points = np.vstack((points, paste_points))
        colors = np.vstack((colors, paste_colors))
        print(f"[*] Pasted object from {args.paste_npy} with offset {offset}")

    # 6. Open3Dで点群オブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 7. 可視化ウィンドウの起動
    print("操作方法: 左クリックで回転, 右クリックで平行移動, ホイールで拡大縮小")
    print("【重要】Ctrl+C でカメラ視点をコピーし、別の画面で Ctrl+V を押すと視点が完全に一致します！")
    
    o3d.visualization.draw_geometries([pcd], window_name='Copy-Paste Viewer', width=1280, height=720)

if __name__ == '__main__':
    main()