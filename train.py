#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse   #comandline
import subprocess   #not use
import datetime   #get datetime
import yaml   #read yamlfile
from shutil import copyfile   #copy file and directory
import sys
import os
import os.path as osp
import shutil
from loguru import logger   #log library

from lib.trainer import Trainer
# from lib.backup import Trainer


# loguru: https://github.com/Delgan/loguru
def set_logger(log_path):   
    logger.add(log_path)
    #logger.add(sys.stdout, colorize=True, format="{message}")
    logger.add(sys.stderr, format="{time} {level} {message}", level="WARNING")

    return logger

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train.py")    #引数指定
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--arch_cfg', '-ac',
      type=str,
      required=True,
      help='Architecture yaml cfg file. See /config/arch for sample. No default!',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default='config/labels/semantic-kitti.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      required=True,
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
  )
  FLAGS, unparsed = parser.parse_known_args()       #引数を解析

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):      #ログディレクトリがすでにあれば削除し、新しく作成。
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  logger = set_logger(osp.join(FLAGS.log, 'log.txt'))     #log.txt にログを保存する設定

  # print summary of what we will do
  logger.info("----------")
  logger.info("INTERFACE:")
  logger.info("dataset", FLAGS.dataset)
  logger.info("arch_cfg", FLAGS.arch_cfg)
  logger.info("data_cfg", FLAGS.data_cfg)
  logger.info("log", FLAGS.log)
  logger.info("pretrained", FLAGS.pretrained)
  logger.info("----------\n")

  # open arch config file
  try:
    logger.info("Opening arch config file %s" % FLAGS.arch_cfg)     #開いた YAML の内容を読み込み、Python の辞書やリストなどの構造に変換する。
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
  except Exception as e:
    logger.warning(e)
    logger.warning("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    logger.info("Opening data config file %s" % FLAGS.data_cfg)
    DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
  except Exception as e:
    logger.warning(e)
    logger.warning("Error opening data yaml file.")
    quit()

  # does model folder exist?
  if FLAGS.pretrained is not None:      #指定された事前学習モデルのフォルダが存在するかチェック。
    if os.path.isdir(FLAGS.pretrained):
      logger.info("model folder exists! Using model from %s" % (FLAGS.pretrained))
    else:
      logger.warning("model folder doesnt exist! Start with random weights...")
  else:
    logger.warning("No pretrained directory found.")

  # copy all files to log folder (to remember what we did, and make inference
  # easier). Also, standardize name to be able to open it later
  try:
    logger.info("Copying files to %s for further reference." % FLAGS.log)     #実行時の設定を保存するため、yamlファイルをログディレクトリにコピー。
    copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
    copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")
  except Exception as e:
    logger.warning(e)
    logger.warning("Error copying files, check permissions. Exiting...")
    quit()

  # create trainer and start the training
  trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log, logger, FLAGS.pretrained, ARCH['model']['use_mps'])
  trainer.train()
