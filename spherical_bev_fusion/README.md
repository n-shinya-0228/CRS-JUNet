# 球面投影＋Polar BEV セグメンテーション

既存ファイルを変更せず、このディレクトリ内でデータ読み込み・投影・CNN・学習・評価を統合しています。SWin・CSWin・timmには依存しません。

`SemanticKitti_Polar3.py`と`laserscan_Polar3.py`に合わせ、BEVは512×512、半径51.2m、7チャンネル（最大高さ・平均高さ・最大反射強度・密度・高さ差・X差・Y差）、代表点は各セルで最も高い点です。球面投影は64×512、距離・XYZ・反射強度・マスクの6チャンネルです。

同じ点群から両投影を作成し、CNNの各分岐で128次元まで復元します。BEV代表点と球面投影の最近点が一致する位置で球面特徴をサンプリングし、有効マスクを掛けてから`bev_features + sphere_on_bev`で加算します。融合直後の1×1畳み込みでクラス分類します。対応のないBEVセルではBEV特徴を使用します。球面分岐には補助損失を設けています。

Polar3のcopy-pasteを共有点群に適用し、回転・反転も投影前に行うことで両投影の整合を維持します。Polar3の投影後のroll・dropoutと厳密に同じ拡張ではありません。データは既存と同じSemanticKITTIで、ラベル対応・学習/検証分割は既存の`semantic-kitti.yaml`をコピーしています。検証は08、未ラベルのテスト分割を学習や精度算出に使用しません。

リポジトリルートから実行します。

```bash
python -m pip install torch numpy pyyaml tensorboardX
python -m spherical_bev_fusion.train --dataset SemanticKitti --log spherical_bev_fusion/runs/fusion128
```

データ配置は`SemanticKitti/sequences/00/velodyne/*.bin`と対応する`labels/*.label`です。事前キャッシュ生成は不要です。既定は100エポックで、設定は`config.yaml`にあります。

```bash
python -m spherical_bev_fusion.train --dataset SemanticKitti --resume spherical_bev_fusion/runs/fusion128/latest.path
python -m spherical_bev_fusion.train --dataset SemanticKitti --evaluate --pretrained spherical_bev_fusion/runs/fusion128/best_val.path
```

出力は`log.txt`、各エポックの`metrics.jsonl`、`results.json`、`best_val.path`、`latest.path`、TensorBoardです。単独評価時は`evaluation.json`にmIoU・accuracy・19クラスのIoU（学習ID1～19順）を保存します。指標は既存Polar学習と同じ**BEV投影ピクセル単位**です。ignore=0を混同行列の両軸で除外します。全点群に逆投影した公式3D評価とは異なり、既存の点群単位評価と直接比較することはできません。

学習済み重みがない状態で数値を記載しません。実行状況は`execution_status.json`を参照してください。

## この環境での実行状況（2026-10-08）

専用環境`/tmp/fusion128-env`で内部確認を実行済みです。実データ学習コマンドも実行しましたが、`SemanticKitti/sequences`が存在しないため、データ確認で終了しています。学習済み重み・実データmIoU・accuracyは生成されていません。

```bash
/tmp/fusion128-env/bin/python -m spherical_bev_fusion.train --dataset /path/to/SemanticKitti
```
