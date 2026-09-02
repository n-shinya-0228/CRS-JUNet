import argparse
import gzip
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.utils.laserscan_Pointnet import (
    build_polar_point_features,
    PolarPointPretrainModel,
)


CLASS_NAMES = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

# 今の研究で特に見たい小物体クラス
SMALL_CLASSES = [2, 3, 6, 7, 8, 18, 19]
CORE_SMALL_CLASSES = [2, 3, 6, 7, 8]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return None


def extract_label_tensor(obj, H=512, W=512):
    """
    polar_512_prlb/*.pt から [H,W] のBEVラベルを取り出す。
    保存形式が dict / tuple / list のどれでもある程度対応する。
    """

    # 1. dictなら名前が明確なキーを優先
    if isinstance(obj, dict):
        preferred_keys = [
            "labels_t",
            "label",
            "labels",
            "proj_label",
            "proj_sem_label",
            "semantic_label",
        ]

        for key in preferred_keys:
            if key in obj:
                t = _to_tensor(obj[key])
                if t is not None:
                    t = t.squeeze()
                    if t.ndim == 2 and tuple(t.shape) == (H, W):
                        return t.long()

        # 名前が違う場合のフォールバック
        candidates = []
        for key, value in obj.items():
            t = _to_tensor(value)
            if t is None:
                continue
            t = t.squeeze()
            if t.ndim == 2 and tuple(t.shape) == (H, W):
                candidates.append((key, t))

        # int系を優先
        for _, t in candidates:
            if not torch.is_floating_point(t) and t.dtype != torch.bool:
                return t.long()

        # 浮動小数でも整数値ラベルなら候補
        for _, t in candidates:
            if torch.is_floating_point(t):
                if torch.allclose(t, t.round()):
                    return t.long()

    # 2. tuple/list
    if isinstance(obj, (tuple, list)):
        candidates = []
        for i, value in enumerate(obj):
            t = _to_tensor(value)
            if t is None:
                continue
            t = t.squeeze()
            if t.ndim == 2 and tuple(t.shape) == (H, W):
                candidates.append((i, t))

        # maskよりlabelを優先するため、整数型かつ2値でないものを先に探す
        for _, t in candidates:
            if not torch.is_floating_point(t) and t.dtype != torch.bool:
                if t.numel() > 0 and int(t.max().item()) > 1:
                    return t.long()

        for _, t in candidates:
            if not torch.is_floating_point(t) and t.dtype != torch.bool:
                return t.long()

    # 3. Tensorそのもの
    t = _to_tensor(obj)
    if t is not None:
        t = t.squeeze()
        if t.ndim == 2 and tuple(t.shape) == (H, W):
            return t.long()

    raise RuntimeError(
        "BEV label [512,512] を polar_512_prlb の .pt から見つけられませんでした。"
        " 保存形式を確認して extract_label_tensor() を調整してください。"
    )


def build_learning_lut(learning_map):
    """
    SemanticKITTI raw label -> learning label の LUT。
    既存BEVラベルがすでに 0..19 なら使用されない。
    """
    if learning_map is None:
        return None

    parsed = {int(k): int(v) for k, v in learning_map.items()}
    max_key = max(parsed.keys())

    lut = torch.zeros(max_key + 1, dtype=torch.long)
    for k, v in parsed.items():
        lut[k] = v

    return lut


def maybe_remap_labels(labels, num_classes, learning_lut):
    """
    labels がすでに learning ID (0..num_classes-1) ならそのまま。
    raw SemanticKITTI IDらしければ learning_map で変換。
    """
    labels = labels.long()

    if labels.numel() == 0:
        return labels

    max_label = int(labels.max().item())
    min_label = int(labels.min().item())

    if min_label >= 0 and max_label < num_classes:
        return labels

    if learning_lut is None:
        raise RuntimeError(
            f"ラベル範囲が [{min_label}, {max_label}] で num_classes={num_classes} を超えています。"
            " data_cfg の learning_map を使ったremapが必要です。"
        )

    if max_label >= learning_lut.numel():
        raise RuntimeError(
            f"BEV label max={max_label} が learning_map LUT size={learning_lut.numel()} を超えています。"
        )

    return learning_lut[labels]


class PointEncoderDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        data_cfg,
        split="train",
        label_folder="polar_512_prlb",
        H=512,
        W=512,
    ):
        self.dataset_root = Path(dataset_root)
        self.data_cfg = data_cfg
        self.split = split
        self.label_folder = label_folder
        self.H = H
        self.W = W

        sequences = data_cfg["split"][split]
        sequences = [f"{int(s):02d}" for s in sequences]

        samples = []
        missing = 0

        for seq in sequences:
            velodyne_dir = self.dataset_root / "sequences" / seq / "velodyne"
            label_dir = self.dataset_root / "sequences" / seq / label_folder

            if not velodyne_dir.exists():
                print(f"[WARN] velodyne directory not found: {velodyne_dir}")
                continue

            scan_files = sorted(velodyne_dir.glob("*.bin"))

            for bin_path in scan_files:
                label_path = label_dir / f"{bin_path.stem}.pt"

                if not label_path.exists():
                    missing += 1
                    continue

                samples.append((bin_path, label_path, seq, bin_path.stem))

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found for split={split}. "
                f"dataset_root={dataset_root}, label_folder={label_folder}"
            )

        self.samples = samples

        print(
            f"[{split}] samples={len(self.samples)}"
            + (f", missing labels={missing}" if missing > 0 else "")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        bin_path, label_path, seq, frame = self.samples[index]

        scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        points = torch.from_numpy(scan).float()

        with gzip.open(label_path, "rb") as f:
            saved = torch.load(
                f,
                map_location="cpu",
                weights_only=False,
            )

        labels = extract_label_tensor(
            saved,
            H=self.H,
            W=self.W,
        )

        return {
            "points": points,       # [N,4]
            "labels": labels,       # [512,512]
            "seq": seq,
            "frame": frame,
        }


def single_item_collate(batch):
    # 点数がscanごとに異なるため、Step Aでは1 scanずつ処理する
    return batch[0]


def select_fixed_subset(dataset, sample_count, seed):
    if sample_count <= 0 or sample_count >= len(dataset.samples):
        return

    rng = random.Random(seed)
    selected_indices = sorted(
        rng.sample(range(len(dataset.samples)), sample_count)
    )
    dataset.samples = [dataset.samples[i] for i in selected_indices]


def write_subset_manifest(dataset, path):
    with open(path, "w") as f:
        f.write("sequence\tframe\tbin_path\tlabel_path\n")
        for bin_path, label_path, sequence, frame in dataset.samples:
            f.write(
                f"{sequence}\t{frame}\t{bin_path}\t{label_path}\n"
            )


def update_confusion(confusion, pred, target, num_classes):
    valid = target != 0

    pred = pred[valid]
    target = target[valid]

    if target.numel() == 0:
        return confusion

    indices = target * num_classes + pred

    hist = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    confusion += hist.cpu()
    return confusion


def metrics_from_confusion(confusion):
    confusion = confusion.float()

    tp = torch.diag(confusion)
    gt = confusion.sum(dim=1)
    pred_count = confusion.sum(dim=0)
    union = gt + pred_count - tp

    iou = torch.full(
        (confusion.shape[0],),
        float("nan"),
        dtype=torch.float32,
    )

    valid_union = union > 0
    iou[valid_union] = tp[valid_union] / union[valid_union]

    # class 0 はignore
    class_iou = iou[1:]

    valid_classes = torch.isfinite(class_iou)
    if valid_classes.any():
        miou = class_iou[valid_classes].mean().item()
    else:
        miou = 0.0

    total = confusion[1:, :].sum()
    correct = tp[1:].sum()
    accuracy = (
        (correct / total).item()
        if total > 0
        else 0.0
    )

    small_values = []
    for c in SMALL_CLASSES:
        if c < len(iou) and torch.isfinite(iou[c]):
            small_values.append(iou[c])

    small_miou = (
        torch.stack(small_values).mean().item()
        if len(small_values) > 0
        else 0.0
    )

    core_small_values = []
    core_small_positive = 0
    for c in CORE_SMALL_CLASSES:
        if c < len(iou) and torch.isfinite(iou[c]):
            core_small_values.append(iou[c])
            if iou[c] > 0:
                core_small_positive += 1

    core_small_miou = (
        torch.stack(core_small_values).mean().item()
        if len(core_small_values) > 0
        else 0.0
    )

    return {
        "iou": iou,
        "miou": miou,
        "accuracy": accuracy,
        "small_miou": small_miou,
        "core_small_miou": core_small_miou,
        "core_small_positive_classes": core_small_positive,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_classes,
    learning_lut,
    H,
    W,
    max_radius,
):
    model.train()

    total_loss = 0.0
    valid_steps = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        points = batch["points"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"]
        labels = maybe_remap_labels(
            labels,
            num_classes=num_classes,
            learning_lut=learning_lut,
        ).to(device)

        point_features, unique_cells, inverse_indices = \
            build_polar_point_features(
                points,
                H=H,
                W=W,
                max_radius=max_radius,
            )

        if point_features.shape[0] == 0:
            continue

        outputs = model(
            point_features,
            inverse_indices,
        )

        logits = outputs["logits"]

        # [H,W] -> [H*W] -> occupied cellだけ取得
        target = labels.reshape(-1)[unique_cells]

        if logits.shape[0] != target.shape[0]:
            raise RuntimeError(
                f"logits/target size mismatch: "
                f"logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )

        valid = target != 0
        if not torch.any(valid):
            continue

        loss = criterion(
            logits,
            target,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 異常な勾配対策
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        total_loss += loss.item()
        valid_steps += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cells=int(unique_cells.numel()),
        )

    return total_loss / max(valid_steps, 1)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    num_classes,
    learning_lut,
    H,
    W,
    max_radius,
):
    model.eval()

    total_loss = 0.0
    valid_steps = 0

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    pbar = tqdm(loader, desc="Valid", leave=False)

    for batch in pbar:
        points = batch["points"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"]
        labels = maybe_remap_labels(
            labels,
            num_classes=num_classes,
            learning_lut=learning_lut,
        ).to(device)

        point_features, unique_cells, inverse_indices = \
            build_polar_point_features(
                points,
                H=H,
                W=W,
                max_radius=max_radius,
            )

        if point_features.shape[0] == 0:
            continue

        outputs = model(
            point_features,
            inverse_indices,
        )

        logits = outputs["logits"]
        target = labels.reshape(-1)[unique_cells]

        valid = target != 0
        if not torch.any(valid):
            continue

        loss = criterion(
            logits,
            target,
        )

        pred = logits.argmax(dim=1)

        confusion = update_confusion(
            confusion,
            pred,
            target,
            num_classes,
        )

        total_loss += loss.item()
        valid_steps += 1

    metrics = metrics_from_confusion(confusion)

    return (
        total_loss / max(valid_steps, 1),
        metrics,
    )


def print_class_iou(iou):
    print("\nPer-class IoU:")
    for class_id in range(1, min(len(iou), len(CLASS_NAMES))):
        value = iou[class_id]
        if torch.isfinite(value):
            print(
                f"  {class_id:2d} "
                f"{CLASS_NAMES[class_id]:15s}: "
                f"{100.0 * value.item():6.2f}%"
            )
        else:
            print(
                f"  {class_id:2d} "
                f"{CLASS_NAMES[class_id]:15s}: "
                f"   N/A"
            )

def build_class_weights(
    data_cfg,
    num_classes=20,
    epsilon=1.2,
    small_boost=None,
):
    """
    SemanticKITTI yaml の content と learning_map から
    learning class 0..19 の重みを作る。

    log inverse weighting:
        weight = 1 / log(epsilon + frequency)

    class 0 は ignore なので weight=0
    """

    if epsilon <= 1.0:
        raise ValueError("class_weight_epsilon must be greater than 1.0")

    content = data_cfg["content"]
    learning_map = data_cfg["learning_map"]

    class_freq = np.zeros(
        num_classes,
        dtype=np.float64
    )

    for raw_id, freq in content.items():

        raw_id = int(raw_id)
        freq = float(freq)

        learning_id = int(
            learning_map[raw_id]
        )

        if learning_id < num_classes:
            class_freq[learning_id] += freq

    # 念のため正規化
    total = class_freq.sum()

    if total > 0:
        class_freq /= total

    weights = np.zeros(
        num_classes,
        dtype=np.float32
    )

    for c in range(1, num_classes):

        if class_freq[c] > 0:
            weights[c] = (
                1.0 /
                np.log(
                    epsilon
                    + class_freq[c]
                )
            )

    # class 0 は ignore
    weights[0] = 0.0

    # class 1～19の平均weightを1にする
    valid = weights[1:] > 0

    weights[1:][valid] /= (
        weights[1:][valid].mean()
    )

    # -------------------------
    # Small-class boost
    # -------------------------
    if small_boost is None:
        small_boost = {
            2: 2.0,   # bicycle
            3: 2.0,   # motorcycle
            6: 2.0,   # person
            7: 2.0,   # bicyclist
            8: 2.0,   # motorcyclist
            18: 1.5,  # pole
            19: 1.5,  # traffic-sign
        }

    for c, boost in small_boost.items():
        weights[c] *= boost

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def serializable_iou(iou):
    result = {}
    for class_id in range(1, min(len(iou), len(CLASS_NAMES))):
        value = iou[class_id]
        result[CLASS_NAMES[class_id]] = (
            value.item() if torch.isfinite(value) else None
        )
    return result


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(path, data):
    with open(path, "a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")

def main():
    parser = argparse.ArgumentParser(
        "Pretrain Polar PointNet encoder"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="SemanticKITTI root directory",
    )

    parser.add_argument(
        "--data_cfg",
        type=str,
        required=True,
        help="SemanticKITTI yaml config",
    )

    parser.add_argument(
        "--label_folder",
        type=str,
        default="polar_512_prlb",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="point_encoder_runs",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Result filesに記録する実験名。省略時はsave_dir名を使う",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="train/valid固定ランダムsubsetのseed",
    )

    parser.add_argument(
        "--class_weight_epsilon",
        type=float,
        default=1.2,
    )

    parser.add_argument("--boost_bicycle", type=float, default=2.0)
    parser.add_argument("--boost_motorcycle", type=float, default=2.0)
    parser.add_argument("--boost_person", type=float, default=2.0)
    parser.add_argument("--boost_bicyclist", type=float, default=2.0)
    parser.add_argument("--boost_motorcyclist", type=float, default=2.0)
    parser.add_argument("--boost_pole", type=float, default=1.5)
    parser.add_argument("--boost_traffic_sign", type=float, default=1.5)

    parser.add_argument(
        "--H",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--W",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max_radius",
        type=float,
        default=51.2,
    )

    parser.add_argument(
        "--debug_samples",
        type=int,
        default=0,
        help=">0ならtrain/validから固定seedでN件選ぶ（後方互換用）",
    )

    parser.add_argument(
        "--debug_train_samples",
        type=int,
        default=0,
        help=">0ならtrain split全体から固定seedでN件選ぶ",
    )

    parser.add_argument(
        "--debug_valid_samples",
        type=int,
        default=0,
        help=">0ならvalid split全体から固定seedでN件選ぶ",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "このStep A実装はGPU使用を推奨します。CUDAが見つかりません。"
        )

    device = torch.device("cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.data_cfg, "r") as f:
        data_cfg = yaml.safe_load(f)

    num_classes = len(data_cfg["learning_map_inv"])

    if num_classes != 20:
        print(
            f"[WARN] num_classes={num_classes}. "
            "標準SemanticKITTI learning classesなら20です。"
        )

    learning_lut = build_learning_lut(
        data_cfg.get("learning_map")
    )

    train_dataset = PointEncoderDataset(
        dataset_root=args.dataset,
        data_cfg=data_cfg,
        split="train",
        label_folder=args.label_folder,
        H=args.H,
        W=args.W,
    )

    valid_dataset = PointEncoderDataset(
        dataset_root=args.dataset,
        data_cfg=data_cfg,
        split="valid",
        label_folder=args.label_folder,
        H=args.H,
        W=args.W,
    )

    train_sample_count = (
        args.debug_train_samples
        if args.debug_train_samples > 0
        else args.debug_samples
    )
    valid_sample_count = (
        args.debug_valid_samples
        if args.debug_valid_samples > 0
        else args.debug_samples
    )

    select_fixed_subset(
        train_dataset,
        train_sample_count,
        args.subset_seed,
    )
    select_fixed_subset(
        valid_dataset,
        valid_sample_count,
        args.subset_seed + 1,
    )

    write_subset_manifest(
        train_dataset,
        save_dir / "train_subset.tsv",
    )
    write_subset_manifest(
        valid_dataset,
        save_dir / "valid_subset.tsv",
    )

    if train_sample_count > 0 or valid_sample_count > 0:
        print(
            f"[DEBUG] fixed random subset seed={args.subset_seed}: "
            f"train={len(train_dataset)}, valid={len(valid_dataset)}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=single_item_collate,
        persistent_workers=(args.num_workers > 0),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=single_item_collate,
        persistent_workers=(args.num_workers > 0),
    )

    model = PolarPointPretrainModel(
        num_classes=num_classes,
        in_channels=7,
        feature_dim=args.feature_dim,
    ).to(device)

    class_weights = build_class_weights(
        data_cfg,
        num_classes=num_classes,
        epsilon=args.class_weight_epsilon,
        small_boost={
            2: args.boost_bicycle,
            3: args.boost_motorcycle,
            6: args.boost_person,
            7: args.boost_bicyclist,
            8: args.boost_motorcyclist,
            18: args.boost_pole,
            19: args.boost_traffic_sign,
        },
    ).to(device)

    print("\nClass weights:")

    for c in range(1, num_classes):
        print(
            f"{c:2d} "
            f"{CLASS_NAMES[c]:15s}: "
            f"{class_weights[c].item():.3f}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_miou = -1.0
    best_small_miou = -1.0
    best_core_small_miou = -1.0
    best_overall_record = None
    best_small_record = None
    best_core_small_record = None

    experiment_name = args.experiment_name or save_dir.name
    boost_config = {
        "bicycle": args.boost_bicycle,
        "motorcycle": args.boost_motorcycle,
        "person": args.boost_person,
        "bicyclist": args.boost_bicyclist,
        "motorcyclist": args.boost_motorcyclist,
        "pole": args.boost_pole,
        "traffic-sign": args.boost_traffic_sign,
    }
    run_config = {
        "experiment_name": experiment_name,
        "dataset": args.dataset,
        "data_cfg": args.data_cfg,
        "label_folder": args.label_folder,
        "epsilon": args.class_weight_epsilon,
        "small_boost": boost_config,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "subset_seed": args.subset_seed,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "feature_dim": args.feature_dim,
        "H": args.H,
        "W": args.W,
        "max_radius": args.max_radius,
        "class_weights": class_weights.detach().cpu().tolist(),
    }
    save_json(save_dir / "run_config.json", run_config)

    print("\n================================")
    print("Polar Point Encoder Pretraining")
    print("================================")
    print(f"device       : {device}")
    print(f"classes      : {num_classes}")
    print(f"feature dim  : {args.feature_dim}")
    print(f"epochs       : {args.epochs}")
    print(f"lr           : {args.lr}")
    print(f"epsilon      : {args.class_weight_epsilon}")
    print(f"boosts       : {boost_config}")
    print(f"train scans  : {len(train_dataset)}")
    print(f"valid scans  : {len(valid_dataset)}")
    print("================================\n")

    # 最初の1サンプルでラベル対応を確認
    sample = train_dataset[0]

    sample_points = sample["points"].to(device)
    sample_labels = maybe_remap_labels(
        sample["labels"],
        num_classes=num_classes,
        learning_lut=learning_lut,
    ).to(device)

    with torch.no_grad():
        pf, uc, inv = build_polar_point_features(
            sample_points,
            H=args.H,
            W=args.W,
            max_radius=args.max_radius,
        )

        sample_target = sample_labels.reshape(-1)[uc]

    unique_target, counts = torch.unique(
        sample_target.cpu(),
        return_counts=True,
    )

    print(
        "[Sanity check] first scan:",
        sample["seq"],
        sample["frame"],
    )
    print("  raw points      :", tuple(sample_points.shape))
    print("  point features  :", tuple(pf.shape))
    print("  occupied cells  :", int(uc.numel()))
    print(
        "  target classes  :",
        list(
            zip(
                unique_target.tolist(),
                counts.tolist(),
            )
        ),
    )
    print(
        "  target min/max  :",
        int(sample_target.min().item()),
        int(sample_target.max().item()),
    )
    print()

    for epoch in range(args.epochs):
        print(
            f"========== Epoch {epoch + 1}/{args.epochs} =========="
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            learning_lut=learning_lut,
            H=args.H,
            W=args.W,
            max_radius=args.max_radius,
        )

        val_loss, metrics = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            learning_lut=learning_lut,
            H=args.H,
            W=args.W,
            max_radius=args.max_radius,
        )

        val_miou = metrics["miou"]
        val_acc = metrics["accuracy"]
        small_miou = metrics["small_miou"]
        core_small_miou = metrics["core_small_miou"]
        core_small_positive = metrics["core_small_positive_classes"]

        print(
            f"train loss : {train_loss:.4f}\n"
            f"val loss   : {val_loss:.4f}\n"
            f"val mIoU   : {100.0 * val_miou:.2f}%\n"
            f"val Acc    : {100.0 * val_acc:.2f}%\n"
            f"small mIoU : {100.0 * small_miou:.2f}%\n"
            f"core small : {100.0 * core_small_miou:.2f}%\n"
            f"core IoU>0 : {core_small_positive}/{len(CORE_SMALL_CLASSES)}"
        )

        class_iou = serializable_iou(metrics["iou"])
        epoch_record = {
            "experiment_name": experiment_name,
            "epsilon": args.class_weight_epsilon,
            "small_boost": boost_config,
            "learning_rate": args.lr,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_accuracy": val_acc,
            "small_miou": small_miou,
            "core_small_miou": core_small_miou,
            "core_small_positive_classes": core_small_positive,
            "class_iou": class_iou,
            "checkpoint_path": str(save_dir / "last_point_pretrain.pth"),
            "best_overall_checkpoint": str(
                save_dir / "best_point_pretrain_full.pth"
            ),
            "best_small_checkpoint": str(
                save_dir / "best_small_point_pretrain_full.pth"
            ),
        }

        # 毎epochの再開用checkpoint
        last_ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_miou": val_miou,
            "val_accuracy": val_acc,
            "small_miou": small_miou,
            "core_small_miou": core_small_miou,
            "core_small_positive_classes": core_small_positive,
            "class_iou": class_iou,
            "feature_dim": args.feature_dim,
            "num_classes": num_classes,
            "run_config": run_config,
        }

        torch.save(
            last_ckpt,
            save_dir / "last_point_pretrain.pth",
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_overall_record = dict(epoch_record)

            # Step Bでそのまま読むencoder単体
            torch.save(
                model.encoder.state_dict(),
                save_dir / "best_point_encoder.pth",
            )

            # 情報付きのcheckpointも保存
            torch.save(
                last_ckpt,
                save_dir / "best_point_pretrain_full.pth",
            )

            print(
                f"[BEST] val mIoU = "
                f"{100.0 * best_miou:.2f}%"
            )

        if small_miou > best_small_miou:
            best_small_miou = small_miou
            best_small_record = dict(epoch_record)

            torch.save(
                model.encoder.state_dict(),
                save_dir / "best_small_point_encoder.pth",
            )

            torch.save(
                last_ckpt,
                save_dir / "best_small_point_pretrain_full.pth",
            )

            print(
                f"[BEST SMALL] small mIoU = "
                f"{100.0 * best_small_miou:.2f}%"
            )

        if core_small_miou > best_core_small_miou:
            best_core_small_miou = core_small_miou
            best_core_small_record = dict(epoch_record)

        print_class_iou(metrics["iou"])
        append_jsonl(save_dir / "epoch_results.jsonl", epoch_record)

        print()

    summary = {
        "run_config": run_config,
        "best_overall": best_overall_record,
        "best_small": best_small_record,
        "best_core_small": best_core_small_record,
    }
    save_json(save_dir / "summary.json", summary)

    print("================================")
    print("Training finished")
    print(
        "Best val mIoU:",
        f"{100.0 * best_miou:.2f}%"
    )
    print(
        "Best overall epoch:",
        best_overall_record["epoch"] if best_overall_record else "N/A",
    )
    print(
        "Best small mIoU:",
        f"{100.0 * best_small_miou:.2f}%",
        "at epoch",
        best_small_record["epoch"] if best_small_record else "N/A",
    )
    print(
        "Best core-small mIoU:",
        f"{100.0 * best_core_small_miou:.2f}%",
        "at epoch",
        best_core_small_record["epoch"] if best_core_small_record else "N/A",
    )
    print(
        "Encoder:",
        save_dir / "best_point_encoder.pth"
    )
    print("================================")


if __name__ == "__main__":
    main()
