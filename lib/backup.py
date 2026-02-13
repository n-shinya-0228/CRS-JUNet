#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.serialization
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
from tensorboardX import SummaryWriter        
import torch.nn.functional as F
import imp
import yaml
import time
import collections
import copy
import cv2
import os
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, StepLR, LambdaLR

# from .utils.logger import Logger
from .utils.avgmeter import *
# from .utils.sync_batchnorm.batchnorm import convert_model
from .utils.warmupLR import *
from .utils.ioueval import *
from .dataset.Parser import Parser
from .models import *
from .losses import *


# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.weight = weight
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

#     def forward(self, inputs, targets):
#         ce_loss = self.ce(inputs, targets)       # shape = (B,H,W)
#         pt = torch.exp(-ce_loss)                # pt = e^{-CE}  (予測確率)
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0,
                 ignore_index=-100, reduction='mean'):
        super().__init__()
        # ① ここでは **属性を作らない**
        if weight is not None:
            # ② register_buffer に登録（GPU にも一緒に移動してくれる）
            self.register_buffer('class_weight', weight.clone().detach())
        else:
            self.class_weight = None

        self.gamma  = gamma
        self.ignore = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp  = F.log_softmax(inputs, dim=1)                # (B,C,H,W)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt    = logpt.exp()                                 # p_t

        ce_loss = -logpt                                    # base CE

        # --- クラス重みを掛ける ---
        if self.class_weight is not None:
            w = self.class_weight[targets]                  # (B,H,W) すでに同デバイス
            ce_loss = ce_loss * w

        # --- Focal 係数 ---
        loss = ((1.0 - pt) ** self.gamma) * ce_loss

        # --- ignore_index と平均 ---
        if self.ignore is not None:
            valid = targets != self.ignore
            loss  = loss[valid]
            if self.class_weight is not None:
                w = w[valid]

        if self.reduction == 'mean':
            if self.class_weight is not None:
                return loss.sum() / w.sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:                # 'none'
            return loss




def set_tensorboard(path):            #モデルの学習過程を可視化
    writer = SummaryWriter(path)
    return writer

class Trainer():
  def __init__(self, ARCH, DATA, datadir, logdir, logger, pretrained=None, use_mps=True):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.log = logdir
    self.logger = logger
    self.pretrained = pretrained
    self.use_mps = use_mps

    torch.manual_seed(0)                              #PyTorchで使う乱数を固定（学習データのシャッフルや初期重みなど）
    torch.backends.cudnn.deterministic = True         #	CUDAの動作を決定論的にする（ランダム性をなくす）
    torch.backends.cudnn.benchmark = False            #最適化探索によるランダムな動作を避ける（↑とセットで安定性確保）
    np.random.seed(0)                                 #	NumPyの乱数を固定

    self.writer = set_tensorboard(osp.join(logdir, 'tfrecord'))       #TensorBoardログ記録のセットアップ

    # get the data
    self.parser = Parser(root=self.datadir,
                         data_cfg = DATA,
                         arch_cfg = ARCH,
                         gt=True,
                         shuffle_train=True)

    # weights for loss (and bias)
    # weights for loss (and bias)
    epsilon_w = self.ARCH["train"]["epsilon_w"]                             # 安定化のためのε項
    content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)   #各クラスの頻度を格納するテンソル
    for cl, freq in DATA["content"].items():
      x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class        #実際のクラスをクロスエントロピー用クラスにマッピングして頻度を集計
      content[x_cl] += freq
    self.loss_w = 1 / (content + epsilon_w)   # get weights                         #クラスの出現頻度の逆数を重みに設定
    for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
      if DATA["learning_ignore"][x_cl]:                                                   #学習に使用しないクラスの重みを0に設定
        # don't weigh
        self.loss_w[x_cl] = 0
    self.logger.info("Loss weights from content: ", self.loss_w.data)                     # 損失重みをログ出力

    self.model = get_model(ARCH['model']['name'])(ARCH['model']['in_channels'], self.parser.get_n_classes(), ARCH["model"]["dropout"])    #モデルのインスタンス化（例：UnpNet）
    weights_total = sum(p.numel() for p in self.model.parameters())
    weights_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)           #全パラメータ数・学習対象パラメータ数を計算
    self.logger.info("Total number of parameters: " + str(weights_total))
    self.logger.info("Total number of parameters requires_grad: " + str(weights_grad))          #パラメータ数をログ出力

    # GPU?
    self.gpu = False
    self.multi_gpu = False
    self.n_gpus = 0
    # self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          #使用可能なGPUがあれば"cuda"を、なければ"cpu"を使用
    self.logger.info("Training in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:                     #CUDA GPUがあるなら self.gpu = True にし、モデルを .cuda() でGPUに移す
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.n_gpus = 1
      self.model.cuda()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:                    #2枚以上GPUがある場合は nn.DataParallel でマルチGPU学習を有効化
      self.logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)   # spread in gpus
      # self.model = convert_model(self.model).cuda()  # sync batchnorm
      # self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True
      self.n_gpus = torch.cuda.device_count()

    # loss
    if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":     #設定ファイルで "xentropy" が指定されていれば...
      self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)          #重み付きクロスエントロピー（NLLLoss(weight=…)） 
      self.criterion1 = nn.CrossEntropyLoss(weight=self.loss_w).to(self.device)
      self.ls = Lovasz_softmax(ignore=0).to(self.device)                       #クラス0を無視した Lovasz-Softmax損失（IoU最適化用）
    
    elif "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "focal_gamma":
      self.criterion1 = FocalLoss(weight=self.loss_w, gamma=2.0).to(self.device)
    
    else:
      raise Exception('Loss not defined in config file')

    if self.use_mps:                                                           # エッジ損失 BCEWithLogitsLoss、深度損失 Depth_Loss を追加で定義
      self.criterion_e = nn.BCEWithLogitsLoss().to(self.device)
      self.criterion_d = Depth_Loss().to(self.device)


    # loss as dataparallel too (more images in batch)
    # if self.n_gpus > 1:                                                         #複数GPUで損失計算を並列化
    #   self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpu
    #   self.criterion1 = nn.DataParallel(self.criterion1).cuda()
    #   self.ls = nn.DataParallel(self.ls).cuda()
    #   if self.use_mps:
    #     self.criterion_e = nn.DataParallel(self.criterion_e).cuda()        #エッジ損失、深度損失も並列化
    #     self.criterion_d = nn.DataParallel(self.criterion_d).cuda()

    # Use SGD optimizer to train
    # self.optimizer = optim.SGD(self.model_single.parameters(),

    #adam
    self.optimizer = optim.AdamW(self.model.parameters(),                     #学習率、モーメンタム、重み減衰を指定してSGDオプティマイザを設定
                               lr=self.ARCH["train"]["lr"],
                               weight_decay=self.ARCH["train"]["w_decay"])
    

    # self.optimizer = optim.SGD(self.model.parameters(),                     #学習率、モーメンタム、重み減衰を指定してSGDオプティマイザを設定
    #                            lr=self.ARCH["train"]["lr"],
    #                            momentum=self.ARCH["train"]["momentum"],
    #                            weight_decay=self.ARCH["train"]["w_decay"])

    # Use warmup learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = self.parser.get_train_size()                         #1エポックに何ステップ（イテレーション）があるかを取得
    up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)     #warmup期間を何ステップにするか計算
    stay_steps =  int((self.ARCH["train"]["max_epochs"] - self.ARCH["train"]["wup_epochs"] - self.ARCH["train"]["cos_epochs"])* steps_per_epoch)
    down_steps = int(self.ARCH["train"]["cos_epochs"] * steps_per_epoch)
    final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)    #学習率の減衰率をステップ単位に換算している。
    self.scheduler = warmupLR(optimizer=self.optimizer,                    #上記設定に基づいて、warmup → decay という段階的学習率スケジューラを構築
                              lr=self.ARCH["train"]["lr"],
                              warmup_steps=up_steps,
                              momentum=self.ARCH["train"]["momentum"],
                              decay=final_decay)
    
    # warmup_scheduler = LambdaLR(optimizer=self.optimizer,
    #                             lr_lambda=lambda step: 0.1 + 0.9 * (step / up_steps)if step < up_steps else 1.0)
    
    # const_scheduler = StepLR(optimizer=self.optimizer, step_size=stay_steps, gamma=1.0)       
    
    # cosine_scheduler = CosineAnnealingLR(optimizer=self.optimizer,
    #                                       T_max=down_steps,
    #                                       eta_min=self.ARCH["train"]["lr"] * 0.0001  )
    
    # self.scheduler = SequentialLR(optimizer=self.optimizer,
    #                               schedulers=[warmup_scheduler, const_scheduler, cosine_scheduler],
    #                               milestones=[up_steps, stay_steps + up_steps])                        #どのステップで次のスケジューラに切り替えるか

    # self.start_epoch = 0
    # if self.pretrained is not None:
    #   try:
    #     w_dict = torch.load(self.pretrained)
    #     self.model.load_state_dict(w_dict['model'])
    #     self.optimizer.load_state_dict(w_dict['optim'])
    #     self.scheduler.load_state_dict(w_dict['scheduler'])
    #     self.start_epoch = w_dict['epoch']
    #     self.logger.info("Successfully loaded model weights")
    #   except Exception as e:
    #     self.logger.warning("Couldn't load parameters, using random weights. Error: ", e)
    #     raise e

    # self.start_epoch = 0
    # if self.pretrained is not None:                   #--pretrained オプションなどで .path ファイルが指定されていれば以下の処理へ
    #   try:
    #     # PyTorch 2.6以降の制約に対応
    #     torch.serialization.add_safe_globals([torch.optim.lr_scheduler.CyclicLR])
    #     w_dict = torch.load(self.pretrained, weights_only=False)  # ← 修正(保存された .path ファイル（実態は辞書）をロード：)
    #     self.model.load_state_dict(w_dict['model'])                              #それぞれの state_dict() を使って、保存された状態に復元（= 学習を中断したところから再開できる）
    #     self.optimizer.load_state_dict(w_dict['optim'])
    #     self.scheduler.load_state_dict(w_dict['scheduler'])
    #     self.start_epoch = w_dict['epoch']                        #再開用にエポック数を上書き。学習ループではこの値からスタート
    #     self.logger.info("Successfully loaded model weights")
    #   except Exception as e:
    #     self.logger.warning("Couldn't load parameters, using random weights. Error: ", e)
    #     raise e

    self.start_epoch = 0
    if self.pretrained is not None:
        # Resume training from checkpoint
        try:
            # Safely register CyclicLR for PyTorch versions that require it
            try:
                torch.serialization.add_safe_globals([torch.optim.lr_scheduler.CyclicLR])
            except AttributeError:
                pass

            # Load the checkpoint dict
            w_dict = torch.load(self.pretrained, weights_only=False)       # ← 修正(保存された .path ファイル（実態は辞書）をロード：)

            # Restore model, optimizer, and scheduler states
            self.model.load_state_dict(w_dict['model'])
            self.optimizer.load_state_dict(w_dict['optim'])             #それぞれの state_dict() を使って、保存された状態に復元（= 学習を中断したところから再開できる）
            self.scheduler.load_state_dict(w_dict['scheduler'])

            # Set start epoch for resuming training loop
            self.start_epoch = w_dict.get('epoch', 0)           # 中断したエポック数を取得して再開
            self.start_epoch += 1
            self.logger.info(f"Successfully loaded model weights. Resuming from epoch {self.start_epoch}")    

        except Exception as e:
            # If loading fails, warn and continue with random weights
            self.logger.warning(f"Couldn't load parameters, using random weights. Error: {e}")     # 読み込みに失敗してもランダム初期化で学習を続行
            # Do not raise here; continue training from scratch


  def train(self):

    # accuracy and IoU stuff
    best_train_iou = 0.0           #最も良いIoUスコアを記録しておき、モデル保存の基準にする
    best_val_iou = 0.0

    self.ignore_class = []                                       #学習対象外のクラス（loss weightが0に近いもの）を IoU 評価から除外。
    for i, w in enumerate(self.loss_w):
      if w < 1e-10:
        self.ignore_class.append(i)
        self.logger.info("Ignoring class ", i, " in IoU evaluation")
    self.evaluator = iouEval(self.parser.get_n_classes(),            #クラス数とデバイスを使って IoU 評価器を初期化
                             self.device, self.ignore_class)

    # train for n epochs
    for epoch in range(self.start_epoch, self.ARCH["train"]["max_epochs"]):           #指定したエポック数（たとえば 10〜100）まで学習を繰り返す。
      # get info for learn rate currently

      # train for 1 epoch
      acc, iou, loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),         #1エポック分の訓練を実行。精度・IoU・損失・重みの更新比を取得。
                                                     model=self.model,
                                                     optimizer=self.optimizer,
                                                     epoch=epoch,
                                                     evaluator=self.evaluator,
                                                     scheduler=self.scheduler,
                                                     color_fn=self.parser.to_color,
                                                     report=self.ARCH["train"]["report_batch"])
      
      self.writer.add_scalar('training/acc', acc, epoch)                    #TensorBoard に訓練結果を出力（可視化用ログ）
      self.writer.add_scalar('training/mIoU', iou, epoch)
      self.writer.add_scalar('training/loss', loss, epoch)
      self.writer.add_scalar('training/update_mean', update_mean, epoch)

      # remember best iou and save checkpoint
      if iou > best_train_iou:                                                         #新しいベストIoUを得たら、そのモデルを保存
        self.logger.info("Best mean iou in training set so far, save model!")
        best_train_iou = iou
        # torch.save({ 'epoch': epoch, 
        #      'optim': self.optimizer.state_dict(),                              #.state_dict()はPyTorch におけるモデルやオプティマイザの「学習パラメータ」だけを辞書形式で取得する関数
        #      'scheduler': self.scheduler.state_dict(), 
        #      'model': self.model.state_dict() }, 
        #      osp.join(self.log, 'epoch-' + str(epoch).zfill(4) + '.path'))


      if epoch % self.ARCH["train"]["report_epoch"] == 0:               #指定した周期（たとえば5エポックごと）に検証データで評価。
        # evaluate on validation set
        self.logger.info("*" * 80)
        acc, iou, loss = self.validate(val_loader=self.parser.get_valid_set(),                      #検証セットで精度・IoU・損失を計算
                                                 model=self.model,
                                                 evaluator=self.evaluator,
                                                 class_func=self.parser.get_xentropy_class_string)
      
        self.writer.add_scalar('validating/acc', acc, epoch)
        self.writer.add_scalar('validating/mIoU', iou, epoch)
        self.writer.add_scalar('validating/loss', loss, epoch)                    # validationの結果もTensorBoardに出力

        # remember best iou and save checkpoint
        if iou > best_val_iou:
          self.logger.info("Best mean iou in validation so far, save model!")
          self.logger.info("*" * 80)
          best_val_iou = iou

          # save the weights!
          torch.save({ 'epoch': epoch, 
               'optim': self.optimizer.state_dict(), 
               'scheduler': self.scheduler.state_dict(), 
               'model': self.model.state_dict() }, 
              #  osp.join(self.log, 'best_val-epoch-' + str(epoch).zfill(4) + '.path'))
               osp.join(self.log, f'best_val.path'))

        self.logger.info("*" * 80)

    torch.save({
    'epoch': epoch,
    'optim': self.optimizer.state_dict(),
    'scheduler': self.scheduler.state_dict(),   #自分でつけた最後のエポックのモデル
    'model': self.model.state_dict()
    }, osp.join(self.log, 'final.path'))


    self.logger.info('Finished Training')

    return

  def train_epoch(self, train_loader, model, optimizer, epoch, evaluator, scheduler, color_fn, report=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    fslosses = AverageMeter()
    slosses = AverageMeter()                  # 時間、損失、精度、IoU、更新比などの平均値を記録するためのメーターを準備。
    elosses = AverageMeter()
    dlosses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empty the cache to train now
    if self.gpu:
      torch.cuda.empty_cache()          #GPUのキャッシュを削除し、モデルを「訓練モード」に。

    # switch to train mode
    model.train()          #Dropout レイヤーを「訓練モード」に, BatchNorm レイヤーを「訓練モード」に
    #model.eval() を呼ぶと Dropout を無効化し、BatchNorm は訓練中の統計値ではなく「蓄積した」移動平均値を使うようになります。

    end = time.time()

# バッチごとにデータを読み込む。train_loader は DataLoader で定義されている。データローダ(parser.py)
#enumerate(train_loader) で、バッチインデックス i と、上記のタプル全体を順に取り出します。Python のタプルアンパックで「この値はいらない」という意味
#DataLoaderが利用しているデータセットクラス(semantickitti.py)の__getitem__メソッドが何を返しているかに依存します。
    for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, proj_range, _, _, _, _, _, _, edge) in enumerate(train_loader): 
        # measure data loading time
      data_time.update(time.time() - end)
      if not self.multi_gpu and self.gpu:
        in_vol = in_vol.cuda()                    # 入力点群のボリューム
        proj_mask = proj_mask.cuda()              # 有効画素マスク

      if self.gpu:
        proj_labels = proj_labels.cuda(non_blocking=True).long()       # 正解ラベル

      # compute output
      output, skips = model(in_vol)           #UnpNet実行！！！ output：最終的なクラス確率分布（softmax 後）skips：スキップ接続層の中間出力（補助的に損失をかけるために後で使う）
      # loss = self.criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())   #出力とスキップ接続（補助用）を得て、クロスエントロピー＋Lovasz損失を合算。
      loss = self.criterion1(output, proj_labels)     ##CrossEntropy
      if self.use_mps:
        orignal_size = (proj_labels.shape[1], proj_labels.shape[2])
        fsloss = loss

        proj_labels_small = F.interpolate(proj_labels.unsqueeze(1).float(), size=(int(orignal_size[0]/4), int(orignal_size[1]/4)), mode='nearest').long().squeeze()
        sloss = 0
        for s in skips['seg']:
          l = self.criterion(torch.log(F.softmax(s, dim=1).clamp(min=1e-8)), proj_labels_small)
          sloss = sloss + l
        sloss *= 0.1

        edge = edge.cuda()           # エッジラベル
        edge = F.interpolate(edge.unsqueeze(1).float(), size=(int(orignal_size[0]/2), int(orignal_size[1]/2)), mode='nearest')
        eloss = 0
        for e in skips['edge']:
          l = self.criterion_e(e, edge.float())
          # l[l > 1] = 0
          eloss = eloss + l
  
        dloss = 0
        proj_range = proj_range.cuda()               # 距離マップ
        for d in skips['depth']:
          l = self.criterion_d(d, proj_range)
          dloss = dloss + l
        dloss *= 0.01
          
        fslosses.update(fsloss.mean().item(), in_vol.size(0))
        slosses.update(sloss.mean().item(), in_vol.size(0))
        elosses.update(eloss.mean().item(), in_vol.size(0))
        dlosses.update(dloss.mean().item(), in_vol.size(0))

        loss = fsloss + sloss + dloss + eloss 

      # compute gradient and do SGD step
      optimizer.zero_grad()                     # 勾配を計算し、モデルパラメータを更新。
      if self.n_gpus > 1:
        idx = torch.ones(self.n_gpus).cuda()
        loss.backward(idx)                           # 勾配計算
      else:
        loss.backward()
      optimizer.step()                      # パラメータ更新

      # measure accuracy and record loss
      loss = loss.mean()
      with torch.no_grad():                   #精度とIoUを評価器で測定し、バッチごとに更新。勾配計算をやめる
        evaluator.reset()
        argmax = output.argmax(dim=1)
        evaluator.addBatch(argmax, proj_labels)       #logに表示させるaccなどを計算
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
      losses.update(loss.item(), in_vol.size(0))
      acc.update(accuracy.item(), in_vol.size(0))
      iou.update(jaccard.item(), in_vol.size(0))


      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # get gradient updates and weights, so I can print the relationship of
      # their norms
      update_ratios = []
      for g in self.optimizer.param_groups:
        lr = g["lr"]
        for value in g["params"]:
          if value.grad is not None:
            w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
            update = np.linalg.norm(-max(lr, 1e-10) *
                                    value.grad.cpu().numpy().reshape((-1)))
            update_ratios.append(update / max(w, 1e-10))
      update_ratios = np.array(update_ratios)
      update_mean = update_ratios.mean()
      update_std = update_ratios.std()
      update_ratio_meter.update(update_mean)  # over the epoch


      if i % self.ARCH["train"]["report_batch"] == 0:                   #一定バッチごとに進捗ログ（学習率、損失、精度、IoU、更新比）を出力。
        self.logger.info('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'acc {acc.val:.3f} ({acc.avg:.3f}) | '
              'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=acc, iou=iou, lr=lr,
                  umean=update_mean, ustd=update_std))

      # step scheduler
      scheduler.step()                #学習率スケジューラを１ステップ進め
      torch.cuda.empty_cache()

    if self.use_mps:
      self.writer.add_scalar('training/fsloss', fslosses.avg, epoch)
      self.writer.add_scalar('training/sloss', slosses.avg, epoch)
      self.writer.add_scalar('training/eloss', elosses.avg, epoch)
      self.writer.add_scalar('training/dloss', dlosses.avg, epoch)

    return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg           #1エポック分の平均精度、IoU、損失、更新比を返す。

  def validate(self, val_loader, model, evaluator, class_func):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()

    # switch to evaluate mode
    model.eval()                 #モデルを評価モードに（DropoutやBatchNormが無効になる）
    evaluator.reset()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      #検証用のデータローダーからバッチごとにデータを取得
      for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, proj_range, _, _, _, _, _, _, edge) in enumerate(val_loader):
        if not self.multi_gpu and self.gpu:
          in_vol = in_vol.cuda()
          proj_mask = proj_mask.cuda()
        if self.gpu:
          proj_labels = proj_labels.cuda(non_blocking=True).long()

        # compute output
        output, skips = model(in_vol)       #softmax出力（クラスごとの確率）
        # loss = self.criterion(torch.log(output.clamp(min=1e-8)), proj_labels)         #NLLLoss（負の対数尤度損失）
        loss = self.criterion1(output, proj_labels)         #CrossEntropy
        # measure accuracy and record loss
        argmax = output.argmax(dim=1)
        evaluator.addBatch(argmax, proj_labels)
        losses.update(loss.mean().item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)        #各バッチの処理時間を計測
        end = time.time()

      accuracy = evaluator.getacc()
      jaccard, class_jaccard = evaluator.getIoU()
      acc.update(accuracy.item(), in_vol.size(0))
      iou.update(jaccard.item(), in_vol.size(0))

      self.logger.info('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'Acc avg {acc.avg:.3f}\n'
            'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time,
                                           loss=losses,
                                           acc=acc, iou=iou))
      # print also classwise
      for i, jacc in enumerate(class_jaccard):
        self.logger.info('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=class_func(i), jacc=jacc))

    return acc.avg, iou.avg, losses.avg