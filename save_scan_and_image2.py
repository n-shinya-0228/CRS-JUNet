import numpy as np
import cv2
import os

from lib.utils.laserscan4 import LaserScan

# --------- 設定 ---------
bin_path = "/home/ns/CRS-JUNet/SemanticKitti/sequences/08/velodyne/000001.bin"
save_dir = "output"
H = 64
W = 512
fov_up = 3.0
fov_down = -23.0
# ------------------------

# 1. 保存先ディレクトリ作成
os.makedirs(save_dir, exist_ok=True)

# 2. 点群の読み込みと投影設定
# project=True にしているため、open_scan内で自動的に do_range_projection() が呼ばれます
scan = LaserScan(project=True, H=H, W=W, fov_up=fov_up, fov_down=fov_down)
scan.open_scan(bin_path)

# 3. .pcd 保存（Open3Dなし）
points = scan.points[:, :3]
pcd_path = os.path.join(save_dir, "pointcloud.pcd")
with open(pcd_path, 'w') as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z\n")
    f.write("SIZE 4 4 4\n")
    f.write("TYPE F F F\n")
    f.write("COUNT 1 1 1\n")
    f.write(f"WIDTH {points.shape[0]}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {points.shape[0]}\n")
    f.write("DATA ascii\n")
    for pt in points:
        f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n") # 小数点以下を少し丸めるとファイルサイズが軽くなります
print(f"Saved: 3D point cloud -> {pcd_path}")

# 4. range image を保存（OpenCVでpng出力）
# 元の配列を書き換えないようにコピーを取得
range_img = np.copy(scan.proj_range)

# 無効値（-1など）やNaNの処理
range_img = np.nan_to_num(range_img, nan=0.0)
range_img[range_img <= 0] = 0.0 # 0以下の無効ピクセルをすべて0にする

# 正規化 (0 ~ 255)
max_val = np.max(range_img)
if max_val > 0:
    range_img_normalized = (range_img / max_val * 255).astype(np.uint8)
else:
    range_img_normalized = np.zeros_like(range_img, dtype=np.uint8)

# カラーマップ (JET) の適用
color_img = cv2.applyColorMap(range_img_normalized, cv2.COLORMAP_JET)

# 無効データ（距離が0のピクセル）をグレーアウト
zero_mask = (range_img_normalized == 0)
gray_val = 192
color_img[zero_mask] = (gray_val, gray_val, gray_val)

# 保存
img_path = os.path.join(save_dir, "range_image_output.png")
cv2.imwrite(img_path, color_img)
print(f"Saved: 2D range image -> {img_path}")