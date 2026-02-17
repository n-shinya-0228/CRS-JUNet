import numpy as np
import cv2
import os

from lib.utils.laserscan import LaserScan
import scipy.ndimage as ndimage

# --------- 設定 ---------
bin_path = "/home/ns/SemanticKitti/sequences/08/velodyne/000001.bin"
save_dir = "output"
H = 64
W = 512
fov_up = 3.0
fov_down = -23.0
# ------------------------

# 保存先ディレクトリ作成
os.makedirs(save_dir, exist_ok=True)

# 点群の読み込みと投影設定
scan = LaserScan(project=True, H=H, W=W, fov_up=fov_up, fov_down=fov_down)
scan.open_scan(bin_path)
# scan.do_range_projection()

# ① .pcd 保存（Open3Dなし）
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
        f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
print(f"Saved: 3D point cloud -> {pcd_path}")

# ② range image を保存（OpenCVでpng出力）
range_img = scan.proj_range
# proj_mask = scan.proj_mask
# proj_range_np = range_img
# # 修正箇所: proj_mask は既に numpy.ndarray なので .numpy() を削除
# proj_mask_np = proj_mask.astype(bool) # 修正後
# # proj_mask_np = proj_mask.numpy().astype(bool) # 修正前

# missing_mask = ~proj_mask_np
# proj_range_np[missing_mask] = np.nan # proj_range_np は range_img と同じオブジェクトを指しているので、range_img自体が変更されます。

# proj_range_temp = proj_range_np.copy()
# proj_range_temp[~proj_mask_np] = 0 # 欠損を0で埋めて一時的にぼかす

# flag = np.sum(proj_range_temp == 0)

# # Gaussian filter (sigma: ぼかしの強度)
# sigma = 1.0 # 調整してください
# blurred_range = ndimage.gaussian_filter(proj_range_temp, sigma=sigma, mode='nearest')

# # 補間された値 (blurred_range) を欠損部分に適用し、有効な部分は元の値を保持
# proj_range_filled_np = np.where(proj_mask_np, proj_range_np, blurred_range)

# range_img_normalized = (proj_range_filled_np / np.max(proj_range_filled_np) * 255).astype(np.uint8)
# print(range_img_normalized)
# color_img = cv2.applyColorMap(range_img_normalized, cv2.COLORMAP_JET)
# zero_mask = (range_img_normalized == 0)
# gray_val = 192
# color_img[zero_mask] = (gray_val, gray_val, gray_val)
# img_path = os.path.join(save_dir, "range7_image.png")
# cv2.imwrite(img_path, color_img)
# print(f"Saved: 2D range image -> {img_path}")

# flag1 = np.sum(proj_range_filled_np == 0)

# print(flag, flag1)

range_img = np.nan_to_num(range_img, nan=0.0)

range_img[range_img == -1] = 0.0

range_img_normalized = (range_img / np.max(range_img) * 255).astype(np.uint8)
print(range_img_normalized)
color_img = cv2.applyColorMap(range_img_normalized, cv2.COLORMAP_JET)
zero_mask = (range_img_normalized == 0)
gray_val = 192
color_img[zero_mask] = (gray_val, gray_val, gray_val)
img_path = os.path.join(save_dir, "range8_image.png")
cv2.imwrite(img_path, color_img)
print(f"Saved: 2D range image -> {img_path}")
