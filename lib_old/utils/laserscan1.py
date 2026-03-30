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
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z     np.zeros(0, 3)はまだ点がないけど、あとで 3次元点を追加したい
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission(反射強度)

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,         #投影された画像のレンジ（距離画像）を -1 で初期化
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)      #元の3D点群の深度（距離）を格納する配列

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,          #投影後の各画素に対応する3D位置、リミッション、元点群中のインデックスを保持。
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x 各点がどの画素(x, y)に対応しているかを記録。（後で使用）
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),              #その画素に点が投影されたかどうかのマスク。
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):                #binファイルの読み込み（重要）
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)        #4列（x, y, z, remission）として読み込み（例：120000点 → shape=(120000, 4)）
    scan = scan.reshape((-1, 4))                          #-1 は「自動計算してね」という意味で,4列に直したい

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)             #点群と反射強度に分けて set_points() に渡す

  # def open_edge(self, path):
  #     self.edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  #     self.edge = (self.edge > 0)

  def open_edge(self, path):
      self.edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      if self.edge is None:
          # エッジファイルが存在しない or 読み込めなかった場合、ゼロで埋める
          self.edge = np.zeros((self.proj_H, self.proj_W), dtype=np.uint8)
      self.edge = (self.edge > 0)


  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz     （メンバーに保存。リミッションがなければゼロ埋め。）
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()                              


  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.              ##点群を球面画像（range image）に投影するメソッド
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    # fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    # fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov_up = self.proj_fov_up       # field of view up in rad
    fov_down = self.proj_fov_down   # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    # yaw = -np.arctan2(scan_y, scan_x)
    # pitch = np.arcsin(scan_z / depth)
    yaw = -np.arctan2(scan_y, scan_x)* 180/np.pi
    pitch = np.arcsin(scan_z / depth)* 180/np.pi
    # pitch_deg = pitch 
    # mask = (pitch_deg >= -23.0) & (pitch_deg <= 3.0)
    # mask = (pitch_deg >= -150.0) & (pitch_deg <= 150.0)
    # points, remissions, depth, scan_* の順に同じマスクを適用
    # self.points    = self.points[mask]
    # self.remissions= self.remissions[mask]
    # depth          = depth[mask]
    # scan_x         = scan_x[mask]
    # scan_y         = scan_y[mask]
    # scan_z         = scan_z[mask]
    # yaw            = yaw[mask]
    # pitch          = pitch[mask]
    # ——————————《ここまで追加》
    # get projections in image coords

    # yaw = np.round(yaw)
    # pitch = np.round(pitch)
    # proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_x = 0.5 * (yaw / 150 + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]　　　　　　　　　proj_x, proj_yは座標

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.round(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)       #depth = [1.0, 2.0, 1.414, 1.414, 3.0]

    # order in decreasing depth
    #indices = [0, 1, 2, 3, 4]各点に「元の点群中で何番目か」を示すラベルを付与しています。
    indices = np.arange(depth.shape[0])
    #np.argsort(depth) → [0, 2, 3, 1, 4] （小さい順に並んだインデックス）これを [::-1] で反転 → order = [4, 1, 3, 2, 0]
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]                #各点には点群中のインデックス番号を記録(10万個くらい)
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices                       #proj_idx[y, x] に点群中のインデックスが格納されます（-1なら点がない）
                                                                    #例えば、点 #45678 が画像上の (x=320, y=15) に投影されたとすると proj_idx[15, 320] = 45678
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)         #proj_idx > 0 → 点が投影された画素（0以上のインデックス）,無効画素（マスク0）は損失に含めない。


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    # make semantic colors
    if sem_color_dict:
      # if I have a dict, make it
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.1)

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
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
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=float)              # [H,W,3] color

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.int32)              #ファイルを1D整数配列として読み込み
    label = label.reshape((-1))

    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np            #ラベル配列を直接設定する
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half　　　　　　　　　　下位16ビットをsemantic、上位16ビットをinstance IDとして抽出。
      self.inst_label = label >> 16    # instance id in upper half                     labelの値が1300283(semantic+instance)のような時でも確実にsemanticIDを取得
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]                     #2D画像（range image）上のラベル
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]          #セマンティック/インスタンスラベルと色を各画素にマッピング。
