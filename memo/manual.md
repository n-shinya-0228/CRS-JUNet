# 研究の進め方
実行コマンド
--------------------------------------------------------------
- 学習

  `uv run python train.py --dataset SemanticKitti/ --arch_cfg config/arch/SJunNet.yaml --log logs/BEV_`

- グラフの確認

  `uv run python -m tensorboard.main --logdir logs --port 6006`

- 推論

  `uv run python infer_bev_512v3.py --dataset SemanticKitti/ --arch_cfg config/arch/SJunNet.yaml --data_cfg config/labels/semantic-kitti.yaml --pretrained logs/BEV_512_/best_val.path --save_path output/BEV_`

- 評価　

  `uv run python evaluate_iou.py --dataset SemanticKitti/ --predictions output/ --split valid --data_cfg config/labels/semantic-kitti.yaml`

- 結果の可視化

  `WAYLAND_DISPLAY="" uv run python vis_open3d.py --bin_path SemanticKitti/sequences/08/velodyne/000000.bin --label_path output/BEV_/sequences/08/predictions/000000.label`

- データの作成　

  `uv run python prepare_bev.py --dataset SemanticKitti/ --data_cfg config/labels/semantic-kitti.yaml`

--------------------------------------------------------------



バージョンの切り替え
--------------------------------------------------------------
- dataset 

  Parser.pyの中のfrom .SemanticKittiをかえる

- model 
  
  yamlファイルのmodel"name"のところをかえる（新しく作ったら__init__.pyに追加する）

- laserscan 
  
  prepare_bev.pyのところを変更して新しいデータを作成する（必要によってdatasetの中も変更）

- trainer 
  
  train.pyのところをかえる

**それぞれのバージョンによってyamlファイルの画像サイズやバッチサイズなどの数値をかえる**

--------------------------------------------------------------

それぞれのコードの役割
--------------------------------------------------------------
★dataset
- SemanticKitti.py
  
  このファイルは、生のデータをネットワークが理解できるテンソルに加工を担当している。

  PyTorchの Dataset クラスを継承しており、主に「1フレーム（1スキャン）分のデータをどう処理するか」というルールが書かれている。
  
  - データの読み込み: ハードディスクから生の3D点群データ（.bin）と正解ラベル（.label）を探して読み込みこむ。事前計算された .pt ファイルを読み込むこともある。
  - 2Dへの投影: 3Dの点群を、ネットワークが処理しやすい2Dの画像形式（球面投影のRange Image、あるいは上から見下ろしたBEV画像）に変換する。
  - 前処理とノイズ除去: レーザーが当たらなかった「黒抜け（小穴）」を周囲の値で埋めたり（インペインティング）、データのスケールを0〜1に揃えたり（正規化）する。
  - データ拡張: 学習時のみ、画像を反転させたり一部を隠したりして、モデルが景色を丸暗記（過学習）するのを防ぐ。

- Parser.py
  
  このファイルは、SemanticKitti.py が1つずつ作ったデータを、ネットワークの元へ効率よく運ぶのを担当している。

  PyTorchの DataLoader という機能を使って、学習ループへのデータの渡し方を管理している。

  - データセットの分割: 膨大なデータを「学習用（Train）」「検証用（Valid）」「テスト用（Test）」の3つのグループに正しく振り分ける。
  - バッチ化: ネットワークは1枚ずつではなく、複数枚（例：4フレームや8フレーム）を一度にまとめて処理します。Parserは、SemanticKittiが作った1フレーム分のデータを指定されたバッチサイズ分だけ集め、1つの巨大なテンソル（例：[4, 4, 512, 512]）に結合する。
  - シャッフル: 学習用データを取り出す順番を毎回ランダムにかき混ぜて、学習の偏りを防ぐ。
  - 特殊な結合ルール（Collate関数）: 画像テンソルは単純に結合し、文字列（ファイルパス）などはリストにまとめる、といった複雑な仕分け作業を collate_fn を使って安全に行う。

★model
- __ init __.py


- JunNet.py

★utils
- laserscan.py

--------------------------------------------------------------

データセット
--------------------------------------------------------------
https://semantic-kitti.org/dataset.html

ここからダウンロードしてCRS-JUNet/のところに置く

--------------------------------------------------------------

Zone.Identifierの消し方
--------------------------------------------------------------
カレントディレクトリ以下にある `:Zone.Identifier` で終わるファイルを検索して表示 

`find . -type f -name '*:Zone.Identifier' -print`

確認後、問題なければ実際に削除 (-delete オプション)

`find . -type f -name '*:Zone.Identifier' -delete`

もしくは -exec rm {} \; を使う場合 (少し遅いが確実)

`find . -type f -name '*:Zone.Identifier' -exec rm {} \;`

--------------------------------------------------------------
