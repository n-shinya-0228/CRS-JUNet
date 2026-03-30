#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
import cv2

# LiDARスキャンデータ（xyz座標と反射強度r）を保持するクラス。
class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

  def reset(self):
    """Reset scan members."""
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m,3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m,1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m,1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask

    self.pseudo_image = None

  def size(self):
    """Return the size of the point cloud."""
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """Open raw scan and fill in attributes."""
    self.reset()
    if not isinstance(filename, str):
      raise TypeError(f"Filename should be string type, but was {type(filename)}")
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]
    remissions = scan[:, 3]
    self.set_points(points, remissions)

  def open_edge(self, path):
    self.edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if self.edge is None:
      self.edge = np.zeros((self.proj_H, self.proj_W), dtype=np.uint8)
    self.edge = (self.edge > 0)

  def set_points(self, points, remissions=None):
    """Set scan attributes (instead of opening from file)."""
    self.reset()
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    self.points = points
    if remissions is not None:
      self.remissions = remissions
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    if self.project:
      self.do_pseudo_image_projection()

  def do_pseudo_image_projection(self):
    """Project a pointcloud into a multi-channel Pseudo-image (BEV grid)."""
    
    # 1. グリッドの解像度と範囲
    res = 0.1  # 10cm grid resolution
    x_min, x_max = 0.0, 51.2
    y_min, y_max = -25.6, 25.6
    
    W = int((y_max - y_min) / res)
    H = int((x_max - x_min) / res)

    # ★【修正1】データローダーが要求するすべての配列を正しいサイズで初期化
    self.proj_range = np.full((H, W), -1, dtype=np.float32)
    self.proj_xyz = np.full((H, W, 3), -1, dtype=np.float32)
    self.proj_remission = np.full((H, W), -1, dtype=np.float32)
    self.proj_idx = np.full((H, W), -1, dtype=np.int32)
    self.proj_mask = np.zeros((H, W), dtype=np.int32)
    pseudo_image = np.zeros((H, W, 4), dtype=np.float32)

    # ラベル用の配列も新しい H, W (256x256) で初期化し直す！
    if hasattr(self, 'proj_sem_label'):
        self.proj_sem_label = np.zeros((H, W), dtype=np.int32)
        self.proj_sem_color = np.zeros((H, W, 3), dtype=float)
        self.proj_inst_label = np.zeros((H, W), dtype=np.int32)
        self.proj_inst_color = np.zeros((H, W, 3), dtype=float)

    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]
    remissions = self.remissions.reshape(-1)

    depth = np.linalg.norm(self.points, 2, axis=1)
    self.unproj_range = np.copy(depth)

    # 2. 範囲内の点をフィルタリング
    mask = (scan_x >= x_min) & (scan_x < x_max) & \
           (scan_y >= y_min) & (scan_y < y_max)

    grid_x = np.floor((scan_y - y_min) / res).astype(np.int32)
    grid_y = np.floor((x_max - scan_x) / res).astype(np.int32)

    self.proj_x = np.zeros(self.points.shape[0], dtype=np.int32)
    self.proj_y = np.zeros(self.points.shape[0], dtype=np.int32)
    self.proj_x[mask] = np.clip(grid_x[mask], 0, W - 1)
    self.proj_y[mask] = np.clip(grid_y[mask], 0, H - 1)

    # 3. ボクセル（ピクセル）ごとの計算用データ抽出
    valid_indices = np.arange(self.points.shape[0])[mask]
    gx = self.proj_x[mask]
    gy = self.proj_y[mask]
    
    grid_indices = gy * W + gx
    unique_indices, inverse_indices, counts = np.unique(grid_indices, return_inverse=True, return_counts=True)
    u_y = unique_indices // W
    u_x = unique_indices % W

    z_f = scan_z[mask]
    r_f = remissions[mask]
    pts_f = self.points[mask]

    # 4. グリッドごとに特徴量を計算し、ダミー配列にも格納する
    for i, idx in enumerate(unique_indices):
        cell_mask = (inverse_indices == i)
        
        cell_z = z_f[cell_mask]
        cell_r = r_f[cell_mask]
        
        pseudo_image[u_y[i], u_x[i], 0] = np.max(cell_z)
        pseudo_image[u_y[i], u_x[i], 1] = np.mean(cell_z)
        pseudo_image[u_y[i], u_x[i], 2] = np.max(cell_r)
        pseudo_image[u_y[i], u_x[i], 3] = counts[i] / 100.0

        # BEVにおいては「そのピクセルの中で一番高い位置にある点」を代表点として扱うのが一般的です
        max_idx_in_cell = np.argmax(cell_z)
        orig_idx = valid_indices[cell_mask][max_idx_in_cell]

        self.proj_range[u_y[i], u_x[i]] = np.max(cell_z) # rangeの代わりに高さを格納
        self.proj_xyz[u_y[i], u_x[i]] = pts_f[cell_mask][max_idx_in_cell]
        self.proj_remission[u_y[i], u_x[i]] = cell_r[max_idx_in_cell]
        self.proj_idx[u_y[i], u_x[i]] = orig_idx
        self.proj_mask[u_y[i], u_x[i]] = 1

    self.pseudo_image = pseudo_image
    self.proj_H = H
    self.proj_W = W


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, sem_color_dict=None, project=False, H=64, W=1024,
               fov_up=3.0, fov_down=-25.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    if sem_color_dict:
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0, high=1.0, size=(max_sem_key, 3))
      self.sem_color_lut[0] = np.full((3), 0.1)

    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    super(SemLaserScan, self).reset()
    self.sem_label = np.zeros((0, 1), dtype=np.int32)
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)
    self.inst_label = np.zeros((0, 1), dtype=np.int32)
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)

    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=float)
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=float)

  def open_label(self, filename):
    if not isinstance(filename, str):
      raise TypeError(f"Filename should be string type, but was {type(filename)}")
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")
    label = np.fromfile(filename, dtype=np.int32).reshape((-1))
    self.set_label(label)

  def set_label(self, label):
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF
      self.inst_label = label >> 16
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    assert ((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    self.sem_label_color = self.sem_color_lut[self.sem_label].reshape((-1, 3))
    self.inst_label_color = self.inst_color_lut[self.inst_label].reshape((-1, 3))

  def do_label_projection(self):
    mask = self.proj_idx >= 0
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
