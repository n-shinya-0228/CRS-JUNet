#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
import cv2

#LiDARスキャンデータ（xyz座標と反射強度r）を保持するクラス。
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
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission(反射強度)

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
    self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)

  def open_edge(self, path):
      self.edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      if self.edge is None:
          self.edge = np.zeros((self.proj_H, self.proj_W), dtype=np.uint8)
      self.edge = (self.edge > 0)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file) """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points
    if remissions is not None:
      self.remissions = remissions
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image. """
    fov_up = self.proj_fov_up
    fov_down = self.proj_fov_down
    fov = abs(fov_down) + abs(fov_up)

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points (degreeベースの既存実装を維持)
    yaw = -np.arctan2(scan_y, scan_x) * 180 / np.pi
    pitch = np.arcsin(scan_z / depth) * 180 / np.pi

    # projections in image coords
    proj_x = 0.5 * (yaw / 150 + 1.0)   # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov

    # scale to image size
    proj_x *= self.proj_W
    proj_y *= self.proj_H

    # round and clamp
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)
    self.proj_x = np.copy(proj_x)

    proj_y = np.round(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)
    self.proj_y = np.copy(proj_y)

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assign to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices

    # ★ 修正ポイント：index 0 も有効扱いに
    self.proj_mask = (self.proj_idx >= 0).astype(np.int32)


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    # make semantic colors
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

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=float)

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=float)

  def open_label(self, filename):
    """ Open raw scan and fill in attributes """
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, but was {type}".format(type=str(type(filename))))
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np """
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic
      self.inst_label = label >> 16    # instance
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    assert((self.sem_label + (self.inst_label << 16) == label).all())

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
