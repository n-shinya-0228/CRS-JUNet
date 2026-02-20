import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.laserscan import LaserScan, SemLaserScan
import scipy.ndimage as ndimage

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_EDGE = ['.png']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)       #引数filenameが.binで終わるかチェック

def is_edge(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_EDGE)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


#LiDARの点群データ（.bin）とセマンティックラベル（.label）などを読み込み、学習・評価に使える形式のテンソルに整形する
class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is　　　　データセット
               sequences,     # sequences for this data (e.g. [1,3,4,6])　　　　データセットのシーケンス(Unpyamlに書かれているsplit部分)
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0]) 
               learning_map,  # classes to learn (0 to N-1 for xentropy)    0~178 to 0~19
               learning_map_inv,    # inverse of previous (recover labels)     0~19 to 0~178
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True, skip=0):            # send ground truth?
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]      #Unpyamlの高さ
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],       #tensor型に変換
                                         dtype=torch.float)          
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],         #tensor型に変換
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)       #sequencesフォルダが存在するかを確認。
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))   #辞書型かどうか確認(train.pyでyamlファイルを辞書やリストなどの構造に変換していた)

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))      #list型か確認(train.pyでyamlファイルを辞書やリストなどの構造に変換していた)

    # placeholder for filenames
    self.scan_files = []                #スキャン・エッジ画像・ラベルファイルのパスを格納するリスト。
    self.edge_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")      #/home/jun/src/UnpNet/home/jun/src/SemanticKitti/sequences
      edge_path = os.path.join(self.root, seq, "edge")
      label_path = os.path.join(self.root, seq, "labels")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(             #先ほど作ったからのリストに,scan_path 以下のすべてのファイルの中から .bin 拡張子のファイルを探して、その絶対パスを入れる
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]            #dp: 現在のディレクトリパス,dn: サブディレクトリ名のリスト（使っていない）,fn: 現在のディレクトリ内のファイル名リスト
      edge_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(             #scan_files = [ "/home/user/SemanticKITTI/sequences/00/velodyne/000000.bin",
          os.path.expanduser(edge_path)) for f in fn if is_edge(f)]             #             "/home/user/SemanticKITTI/sequences/00/velodyne/000001.bin",....]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      if self.gt:
        assert(len(scan_files) == len(edge_files))


      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)        
      self.edge_files.extend(edge_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()              #ファイル名順にソート（ファイル名が連番で対応するため重要）
    self.edge_files.sort()
    self.label_files.sort()

    if skip != 0:                                         #フレームをスキップ。例：skip=2なら1つおきに使用。
        self.scan_files = self.scan_files[::skip]
        self.edge_files = self.edge_files[::skip]
        self.label_files = self.label_files[::skip]


#ロードしたスキャン数を出力。
    print("Using {} scans from sequences {}".format(len(self.scan_files),               
                                                    self.sequences))                      


#PyTorchの Dataset クラスでは、__getitem__(self, index) を定義することで、dataset[i] のように書いたときに、i番目のデータをどのように返すかを指定できます。
  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]                #index 番目のスキャンファイル（.bin）とエッジ画像ファイル（.png）を取得。
    # edge_file = self.edge_files[index]
    edge_file = self.edge_files[index] if hasattr(self, 'edge_files') and len(self.edge_files) > index else None          #edge_files が存在しない場合は None


    if self.gt:
      label_file = self.label_files[index]      #gt=True の場合は、ラベルファイル（.label）も取得。


    # open a semantic laserscan
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
      
       
    # edgeを読み込む（scan定義後）
    if edge_file is not None and os.path.exists(edge_file):
      scan.open_edge(edge_file)
    else:
      scan.edge = np.zeros((scan.proj_H, scan.proj_W), dtype=np.uint8)

    # open and obtain scan
    scan.open_scan(scan_file)                       #ファイルの読み込み(投影まで終わる)
    # scan.open_edge(edge_file)
    # scan.open_edge(edge_file)
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)            #sem_labelはSemLaserScanクラスのラベルIDが入っているところ、proj_sem_labelは2D画像（range image）上のラベル
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)  #map関数(static)で細かいラベル（例：34クラス）を学習用クラス（例：20）に変換。

    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan.points.shape[0]                                          #元の点群数（1スキャン内にある点の数）を取得。例えば 123456 点など。
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)          #点の座標（X, Y, Z）を max_points 個まで用意。足りない分は -1 で埋める（パディング）
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)                    # この代入で点群の数（unproj_n_points）だけ(scan.points)を代入
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)          
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()            #投影された画像の各チャネルを PyTorchの Tensor に変換
    proj_mask = torch.from_numpy(scan.proj_mask)                              #これらは13万個ほどの量ではなく64*1024。マッピングされていない部分は-1
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    
    # SemanticKitti.py の __getitem__ 内の補間部分

    # まず、NumPy 配列に変換
    proj_range_np = proj_range.numpy()
    proj_xyz_np = proj_xyz.numpy()
    proj_remission_np = proj_remission.numpy()
    proj_mask_np = proj_mask.numpy().astype(bool) # Trueが有効なデータ

    # 欠損部分 (maskがFalseの箇所) を補間
    # scipy.ndimage を使った簡易的なインペインティング
    # from scipy import ndimage

    # マスクを反転させて、補間対象の領域を定義 (Trueが補間対象)
    # ここでは、元のデータが存在しない部分(proj_mask_npがFalse)を補間対象とします
    missing_mask = ~proj_mask_np

    # 各チャンネルに対して補間を適用
    # 例: ndimage.binary_erosion + ndimage.gaussian_filter を使ったインペインティング
    # これは「最も近い非欠損値で埋める」ような効果を狙ったもので、
    # 欠損領域の内部が広すぎると効果が薄いです。
    # より複雑なインペインティングにはOpenCVのcv2.inpaintなどが必要です。

    # 簡易的な例: 欠損部分を周囲の有効な値で埋める
    # 欠損部分を0として、有効な部分を埋める
    # この処理の前に、一旦欠損部分をNaNにしておくと良い場合もありますが、
    # 現在は0なのでそのまま処理します。

    # 例：OpenCVのinpaintを使う場合（別途OpenCVのimportが必要）
    # import cv2
    # # `cv2.inpaint` は8ビット画像と浮動小数点画像をサポート
    # # proj_range_np はfloat32なのでそのまま使える。
    # # ただし、`inpaint`のマスクは255で有効、0で無効
    # inpaint_mask = (~proj_mask_np * 255).astype(np.uint8) # 欠損箇所が255になるマスク

    # proj_range_filled_np = cv2.inpaint(proj_range_np, inpaint_mask, 3, cv2.INPAINT_TELEA)
    # # 同様にxyzとremissionも処理

    # シンプルに、隣接ピクセルの平均で埋める再帰的な処理
    # これは計算コストが高いので、大規模なデータには向きません。

    # 欠損ピクセルに印をつける (例: NaN)
    proj_range_np[missing_mask] = np.nan
    proj_remission_np[missing_mask] = np.nan
    for i in range(3):
        proj_xyz_np[:,:,i][missing_mask] = np.nan

    # NaNを埋めるための関数 (例: pandasのfillnaに似た概念)
    # SciPyのndimage.map_coordinatesやgriddataが本格的ですが、
    # 簡略化して最近傍または双線形的にNaNを埋める関数が必要です。
    # これを自前で実装するか、外部ライブラリに頼るかになります。

    # ここでは、scipy.ndimage.gaussian_filterの応用例（簡易的補間）
    # 欠損箇所を0としたままでぼかすと、0が周囲に広がるので、これは不適切。
    # 欠損箇所を補間する際は、有効な箇所から情報を持ってくる必要があります。

    # 一つのアイデアとして、有効なピクセルのみを使い、ガウシアンフィルターをかけ、
    # その後元の有効なピクセルを戻す方法
    # まず、有効なピクセルのみの値を複製
    proj_range_temp = proj_range_np.copy()
    proj_range_temp[~proj_mask_np] = 0 # 欠損を0で埋めて一時的にぼかす

    # Gaussian filter (sigma: ぼかしの強度)
    # 欠損領域が広い場合、sigmaを大きくすると良いが、境界が曖昧になる
    sigma = 1.0 # 調整してください
    blurred_range = ndimage.gaussian_filter(proj_range_temp, sigma=sigma, mode='nearest')

    # 補間された値 (blurred_range) を欠損部分に適用し、有効な部分は元の値を保持
    proj_range_filled_np = np.where(proj_mask_np, proj_range_np, blurred_range)

    # 同様に、proj_remission と proj_xyz にも適用
    proj_remission_temp = proj_remission_np.copy()
    proj_remission_temp[~proj_mask_np] = 0
    blurred_remission = ndimage.gaussian_filter(proj_remission_temp, sigma=sigma, mode='nearest')
    proj_remission_filled_np = np.where(proj_mask_np, proj_remission_np, blurred_remission)

    proj_xyz_filled_np = np.zeros_like(proj_xyz_np)
    for i in range(3): # X, Y, Z 各チャンネルで独立して処理
        proj_xyz_channel_temp = proj_xyz_np[:,:,i].copy()
        proj_xyz_channel_temp[~proj_mask_np] = 0
        blurred_channel = ndimage.gaussian_filter(proj_xyz_channel_temp, sigma=sigma, mode='nearest')
        proj_xyz_filled_np[:,:,i] = np.where(proj_mask_np, proj_xyz_np[:,:,i], blurred_channel)

    # NumPy 配列から PyTorch Tensor に戻す
    proj_range = torch.from_numpy(proj_range_filled_np).clone()
    proj_xyz = torch.from_numpy(proj_xyz_filled_np).clone()
    proj_remission = torch.from_numpy(proj_remission_filled_np).clone()


    proj_x = torch.full([self.max_points], -1, dtype=torch.long)            #各点が画像のどのピクセルに投影されたかを保存。復元やunprojectionの際に使用。
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

    edge_tensor = torch.from_numpy(scan.edge).float().unsqueeze(0)

    if edge_tensor.max() > 1.0:
        edge_tensor = edge_tensor / 255.0  # 0.0~1.0 に正規化

    proj = torch.cat([proj_range.unsqueeze(0).clone(),                  #画像としてまとめた [range, x, y, z, remission] を正規化。
                      proj_xyz.clone().permute(2, 0, 1),                
                      proj_remission.unsqueeze(0).clone(),
                      edge_tensor],dim=0)
    # proj = torch.cat([proj_xyz.clone().permute(2, 0, 1),                
    #               proj_remission.unsqueeze(0).clone()])
    proj[:5] = (proj[:5] - self.sensor_img_means[:, None, None]                 #画像全体に対して平均を引き、標準偏差で割る（画像の前処理）
            ) / self.sensor_img_stds[:, None, None]
    # proj = proj * proj_mask.float()                                     #無効なピクセルは proj_mask で0にして無視。

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")                  #/path/to/sequences/08/velodyne/003452.bin → "08"と "003452.label" に変換。
    # print("path_norm: ", path_norm)
    # print("path_seq", path_seq)
    # print("path_name", path_name)

    # return
    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points, scan.edge

  def __len__(self):
    return len(self.scan_files)
  
  def get_pointcloud(self, seq, name):
      # name が "000000.label" で来る想定
      base = os.path.splitext(name)[0]
      bin_path = os.path.join(
          self.root, f"{int(seq):02d}", "velodyne", base + ".bin"
      )
      pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
      return pts

  @staticmethod
  def map(label, mapdict):                                    #ラベルIDを変えるメソッド
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


