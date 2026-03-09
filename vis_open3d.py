#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import os
import open3d as o3d

def main():
    parser = argparse.ArgumentParser("SemanticKITTI Visualizer")
    parser.add_argument('--bin_path', type=str, required=True, help='Path to the .bin file')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the .label file')
    parser.add_argument('--config', type=str, default='config/labels/semantic-kitti.yaml')
    args = parser.parse_args()

    # 1. configからカラーマップを読み込む
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    color_map = cfg['color_map']

    # 2. 点群データ (.bin) の読み込み [x, y, z, remission]
    # SemanticKITTIのbinはfloat32形式
    scan = np.fromfile(args.bin_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3] # x, y, z のみ抽出

    # 3. ラベルデータ (.label) の読み込み
    # SemanticKITTIのlabelはuint32形式。下位16ビットがセマンティクスID
    labels = np.fromfile(args.label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF 

    # 4. ラベルIDをRGBカラーに変換
    colors = np.zeros((len(sem_labels), 3), dtype=np.float64)
    for label_id, bgr in color_map.items():
        # カラーマップはBGR順なので、RGBに反転して0~1に正規化
        rgb = np.array([bgr[2], bgr[1], bgr[0]]) / 255.0
        colors[sem_labels == label_id] = rgb

    # 5. Open3Dで点群オブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 6. 可視化ウィンドウの起動
    print(f"Visualizing:\n - Point Cloud: {args.bin_path}\n - Labels: {args.label_path}")
    print("操作方法: 左クリックで回転, 右クリックで平行移動, ホイールで拡大縮小")
    
    # エラーに強いシンプルな描画関数を使用
    o3d.visualization.draw_geometries([pcd], window_name='CRS-JUNet 3D Viewer', width=1280, height=720)

if __name__ == '__main__':
    main()