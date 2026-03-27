import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.laserscan2 import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_EDGE = ['.png']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_edge(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_EDGE)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


# LiDARの点群データ（.bin）とセマンティックラベル（.label）などを読み込み、学習・評価に使える形式のテンソルに整形
class SemanticKitti(Dataset):

  def __init__(self, root,
               sequences,
               labels,
               color_map,
               learning_map,
               learning_map_inv,
               sensor,
               max_points=150000,
               gt=True, skip=0):
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)  # 5ch用
    self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)    # 5ch用
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt

    # number of classes for CE
    self.nclasses = len(self.learning_map_inv)

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    assert(isinstance(self.labels, dict))
    assert(isinstance(self.color_map, dict))
    assert(isinstance(self.learning_map, dict))
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.edge_files = []
    self.label_files = []

    # fill lists
    for seq in self.sequences:
      seq = '{0:02d}'.format(int(seq))
      print("parsing seq {}".format(seq))

      scan_path = os.path.join(self.root, seq, "velodyne")
      edge_path = os.path.join(self.root, seq, "edge")
      label_path = os.path.join(self.root, seq, "labels")

      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      edge_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(edge_path)) for f in fn if is_edge(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_path)) for f in fn if is_label(f)]

      if self.gt:
        assert(len(scan_files) == len(edge_files))
        assert(len(scan_files) == len(label_files))

      self.scan_files.extend(scan_files)
      self.edge_files.extend(edge_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()
    self.edge_files.sort()
    self.label_files.sort()

    if skip != 0:
      self.scan_files = self.scan_files[::skip]
      self.edge_files = self.edge_files[::skip]
      self.label_files = self.label_files[::skip]

    print("Using {} scans from sequences {}".format(len(self.scan_files), self.sequences))

  def __getitem__(self, index):
    # get paths
    scan_file = self.scan_files[index]
    edge_file = self.edge_files[index] if hasattr(self, 'edge_files') and len(self.edge_files) > index else None
    if self.gt:
      label_file = self.label_files[index]

    # open a (semantic) laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # edge
    if edge_file is not None and os.path.exists(edge_file):
      scan.open_edge(edge_file)
    else:
      scan.edge = np.zeros((scan.proj_H, scan.proj_W), dtype=np.uint8)

    # read scan (+ projection)
    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # unprojected tensors (padded)
    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # projected tensors
    proj_range = torch.from_numpy(scan.proj_range).clone()      # [H,W]
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()          # [H,W,3]
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)                # [H,W] {0,1}
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []

    # ---------- ここから前処理の見直し（インペインティング削除／マスク前提） ----------
    # 入力5ch（range, x, y, z, remission）を正規化し、未観測は0化。その後にmaskチャネルを追加して6chに。
    mask_f = proj_mask.float().unsqueeze(0)  # [1,H,W]
    img5 = torch.cat([
        proj_range.unsqueeze(0),             # 1
        proj_xyz.permute(2, 0, 1),          # 3
        proj_remission.unsqueeze(0)         # 1
    ], dim=0)  # -> [5,H,W]

    # 正規化 → 無効画素0化
    img5 = (img5 - self.sensor_img_means[:, None, None]) / (self.sensor_img_stds[:, None, None])
    img5 = img5 * mask_f

    # マスクチャネルを結合（合計6ch）
    proj = torch.cat([img5, mask_f], dim=0)  # [6,H,W]
    # ---------- ここまで ----------

    # projection indices of original points
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x.squeeze())
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y.squeeze())

    # name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")

    return (proj,                  # [6,H,W]  ← 変更点：maskを追加した6チャネル
            proj_mask,             # [H,W]
            proj_labels,           # [H,W] long (masked)
            unproj_labels,         # [M] or []
            path_seq, path_name,
            proj_x, proj_y,
            proj_range, unproj_range,
            proj_xyz, unproj_xyz,
            proj_remission, unproj_remissions,
            unproj_n_points,
            scan.edge)

  def __len__(self):
    return len(self.scan_files)

  def get_pointcloud(self, seq, name):
      base = os.path.splitext(name)[0]
      bin_path = os.path.join(self.root, f"{int(seq):02d}", "velodyne", base + ".bin")
      pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
      return pts

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy (or vice-versa)
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    return lut[label]
