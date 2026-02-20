import os
import numpy as np
import cv2
from tqdm import tqdm

# SemanticKitti.py のスタイルに合わせた拡張子チェック
EXTENSIONS_SCAN = ['.bin']

def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def generate_edge_from_bin(bin_path, save_path, H=64, W=2048, fov_up=3.0, fov_down=-23.0):
    # バイナリ読み込み
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]
    depth = np.linalg.norm(points, 2, axis=1)
    
    mask = depth > 0
    points = points[mask]
    depth = depth[mask]

    scan_x, scan_y, scan_z = points[:, 0], points[:, 1], points[:, 2]
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(np.clip(scan_z / depth, -1, 1))
    
    fov_up_rad = fov_up / 180.0 * np.pi
    fov_down_rad = fov_down / 180.0 * np.pi
    fov_rad = abs(fov_down_rad) + abs(fov_up_rad)
    
    proj_x = 0.5 * (yaw / np.pi + 1.0) * W
    proj_y = (1.0 - (pitch + abs(fov_down_rad)) / fov_rad) * H
    
    proj_x = np.clip(np.floor(proj_x), 0, W - 1).astype(np.int32)
    proj_y = np.clip(np.floor(proj_y), 0, H - 1).astype(np.int32)
    
    indices = np.argsort(depth)[::-1]
    proj_range = np.zeros((H, W), dtype=np.float32)
    proj_range[proj_y[indices], proj_x[indices]] = depth[indices]
    
    if proj_range.max() > 0:
        norm_range = (proj_range / proj_range.max() * 255).astype(np.uint8)
        edges = cv2.Canny(norm_range, 50, 150)
    else:
        edges = np.zeros((H, W), dtype=np.uint8)
    
    cv2.imwrite(save_path, edges)

def main():
    # SemanticKitti.py と同じルートパス指定
    root = "SemanticKitti/sequences"
    
    # 00から21までをループ
    sequences = [f"{i:02d}" for i in range(22)]
    
    for seq in sequences:
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir):
            continue

        print(f"--- Processing sequence {seq} ---")
        scan_path = os.path.join(seq_dir, "velodyne")
        edge_path = os.path.join(seq_dir, "edge")

        # 保存先フォルダを確実に作成
        if not os.path.exists(edge_path):
            os.makedirs(edge_path)

        # SemanticKitti.py のスタイルでファイルをリストアップ
        scan_files = []
        for dp, dn, fn in os.walk(os.path.expanduser(scan_path)):
            for f in fn:
                if is_scan(f):
                    scan_files.append(os.path.join(dp, f))
        
        # 対応関係を保つためにソート
        scan_files.sort()

        # 進捗表示付きで生成
        for bin_file in tqdm(scan_files, desc=f"Seq {seq}"):
            # ファイル名を取得（000000.binなど）
            basename = os.path.basename(bin_file)
            # 保存先のパスを作成（000000.pngなど）
            save_name = os.path.splitext(basename)[0] + ".png"
            save_path = os.path.join(edge_path, save_name)

            # 生成実行
            generate_edge_from_bin(bin_file, save_path)

if __name__ == "__main__":
    main()