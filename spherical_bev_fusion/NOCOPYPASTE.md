# Copy-Pasteなしの比較実験

実験名: `SphericalBEVFusion_NoCopyPaste_bs2`

前回の成績はユーザー申告: Validation最高45.39%（epoch 77）、epoch 100で45.28%。前回のcheckpoint・ログ・実験時設定はこのPCでは未確認です。

## 変更

- `dataset.py`: `copy_paste=False`を既定値として設定し、Parserから`dataset.copy_paste`を渡します。現在は省略時も無効です。無効時は`apply_copy_paste`を呼びません。
- `config_nocopypaste.yaml`: 既存の`config.yaml`を複製し、`dataset.copy_paste: false`だけ追加しました。
- `check_nocopypaste.py`: 実データの1バッチのみで読み込み、入力形状、float32、forward、loss、backward、両分岐の有限勾配を確認します。学習ループ・optimizer.step・checkpoint保存は実行しません。
- 本書と`nocopypaste_baseline.json`（作業開始時の記録）、`nocopypaste_dataset.diff`（差分）、`nocopypaste_checks.json`（静的確認結果）を追加しました。

現在は`config.yaml`にも`dataset.copy_paste: false`を設定しています。`model.py`、`projection.py`、`trainer.py`、`train.py`、`labels.yaml`は変更していません。特に`fused = bev_features + sphere_on_bev`はそのままです。

## 維持した学習条件

batch_size=2、max_epochs=100、lr=0.001、workers=4、BEV=7×512×512、球面=6×64×512、float32、AdamW、weight_decay=0.01、CosineAnnealingLR、min_lr=0.00002、重み付きCE＋球面補助損失0.3。回転・反転、投影、ラベル分割、正規化はそのままです。乱数シードは元の学習コードで固定されていないため、新たな固定は追加していません。

Copy-Pasteの確率判定用`torch.rand`は無効時にも以前と同じ場所で1回消費します。ただしCopy-Paste内の乱数消費がなくなるため、拡張の具体的な抽選結果まで以前と一致する保証はありません。

前回の実行時`arch_cfg.yaml`、`data_cfg.yaml`、環境・シード記録があれば、実行環境で今回の設定と照合してください。今回は現在の`config.yaml`との比較でCopy-Paste以外に差分がないことを確認しています。

## 実行環境で最初に行う1バッチ確認

リポジトリ直下から、データパスだけ実際の場所に置き換えて実行します。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m spherical_bev_fusion.check_nocopypaste \
  --dataset /実際のパス/SemanticKitti
```

既存のuv環境を使用する場合は`python -B`を`uv run --no-sync python -B`へ置換できます。`--no-sync`はフォルダ外の仮想環境やlockファイルの更新を避けるためです。依存ライブラリが未導入の場合は勝手に同期せず、環境管理者に確認してください。

確認結果は固有名の`spherical_bev_fusion/preflight_runs/nocopypaste_*/result.json`へ保存します。TensorBoardの初期化ファイルもその診断用フォルダ内です。本学習用の出力フォルダは作りません。Copy-Pasteメソッドを例外を発生する関数に置換して、確率判定を強制的に満たした場合と通常の1バッチ読み込みで呼ばれないことを確認します。

## 許可後に実行する100 epoch本学習

**まだ開始していません。1バッチ確認の成功とユーザーの許可後にのみ実行してください。**

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -B -m spherical_bev_fusion.train \
  --dataset /実際のパス/SemanticKitti \
  --arch_cfg spherical_bev_fusion/config_nocopypaste.yaml \
  --data_cfg spherical_bev_fusion/labels.yaml \
  --log spherical_bev_fusion/runs/SphericalBEVFusion_NoCopyPaste_bs2
```

`--resume`、`--pretrained`は付けません。初期化状態（内部epoch 0）から100 epoch学習します。画面・ログ上の最初のepochは1です。

本学習の保存先は`spherical_bev_fusion/runs/SphericalBEVFusion_NoCopyPaste_bs2/`。`log.txt`、`metrics.jsonl`、`results.json`、`best_val.path`、`latest.path`、`tfrecord/`を保存します。現時点でこのパスは存在しません。既存のtrain.pyは新規学習時、保存先が空でなければエラーにし、削除しません。上記パスと引数を使用し、他の実行と同時に同じ保存先を使わないでください。

## このPCでの確認結果

Python構文、Copy-Paste条件式（判定値を強制した検査）、その他のdatasetコード不変、モデル等6ファイルのSHA-256一致を確認済みです。実データとPyTorchがないため、実データ1バッチの確認は未完了です。確認コマンドはデータ未配置のエラーで終了しました。これをforward/backward成功として扱っていません。本学習は開始していません。

2026-10-23更新：通常の学習でも無効になるよう`config.yaml`とdatasetの既定値を変更しました。過去の`nocopypaste_*.json`・diffは作成時点の記録です。本学習は開始していません。
