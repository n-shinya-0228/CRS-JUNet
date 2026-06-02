# prepare_multiframe_cache.py
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm


def load_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3].astype(np.float32)
    remissions = scan[:, 3:4].astype(np.float32)
    return points, remissions


def load_label(label_path):
    label = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
    return label


def load_poses(pose_file):
    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            T = np.fromstring(line, dtype=np.float32, sep=" ")
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1])).astype(np.float32)
            poses.append(T)
    return poses


def build_multiframe_cache_for_sequence(dataset_root, seq, num_past_frames=2):
    seq = f"{int(seq):02d}"

    seq_root = os.path.join(dataset_root, "sequences", seq)
    velodyne_dir = os.path.join(seq_root, "velodyne")
    label_dir = os.path.join(seq_root, "labels")
    pose_file = os.path.join(seq_root, "poses.txt")

    save_dir = os.path.join(seq_root, f"multiframe_{num_past_frames}past")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(velodyne_dir):
        print(f"Skip sequence {seq}: velodyne not found")
        return

    scan_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith(".bin")])

    if os.path.exists(pose_file):
        poses = load_poses(pose_file)
    else:
        poses = [np.eye(4, dtype=np.float32) for _ in scan_files]

    print(f"Processing sequence {seq}, scans={len(scan_files)}")

    for idx, scan_name in enumerate(tqdm(scan_files)):
        save_path = os.path.join(save_dir, scan_name.replace(".bin", ".npz"))

        if os.path.exists(save_path):
            continue

        bin_path = os.path.join(velodyne_dir, scan_name)
        label_path = os.path.join(label_dir, scan_name.replace(".bin", ".label"))

        if not os.path.exists(label_path):
            continue

        points, remissions = load_bin(bin_path)
        labels = load_label(label_path)

        cur_pose = poses[idx]
        cur_pose_inv = np.linalg.inv(cur_pose).astype(np.float32)

        all_points = [points]
        all_rems = [remissions]
        all_labels = [labels]

        for k in range(1, num_past_frames + 1):
            past_idx = idx - k

            if past_idx < 0:
                break

            past_scan_name = scan_files[past_idx]
            past_bin_path = os.path.join(velodyne_dir, past_scan_name)
            past_label_path = os.path.join(label_dir, past_scan_name.replace(".bin", ".label"))

            if not os.path.exists(past_label_path):
                break

            past_points, past_rems = load_bin(past_bin_path)
            past_labels = load_label(past_label_path)

            past_keep_ratio = 0.1 

            if past_keep_ratio < 1.0:
                n_past = past_points.shape[0]
                keep_n = int(n_past * past_keep_ratio)
                keep_idx = np.random.choice(n_past, keep_n, replace=False)

                past_points = past_points[keep_idx]
                past_rems = past_rems[keep_idx]
                past_labels = past_labels[keep_idx]


            past_pose = poses[past_idx]
            transform = cur_pose_inv @ past_pose

            n = past_points.shape[0]
            past_points_h = np.hstack(
                [past_points, np.ones((n, 1), dtype=np.float32)]
            )
            past_points_transformed = (transform @ past_points_h.T).T[:, :3].astype(np.float32)

            all_points.append(past_points_transformed)
            all_rems.append(past_rems)
            all_labels.append(past_labels)

        points_cat = np.concatenate(all_points, axis=0).astype(np.float32)
        remissions_cat = np.concatenate(all_rems, axis=0).astype(np.float32)
        labels_cat = np.concatenate(all_labels, axis=0).astype(np.int32)

        np.savez(
            save_path,
            points=points_cat,
            remissions=remissions_cat,
            labels=labels_cat,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SemanticKitti/")
    parser.add_argument("--data_cfg", type=str, default="config/labels/semantic-kitti.yaml")
    parser.add_argument("--num_past_frames", type=int, default=2)
    args = parser.parse_args()

    with open(args.data_cfg, "r") as f:
        data_cfg = yaml.safe_load(f)

    sequences = (
        data_cfg["split"]["train"]
        + data_cfg["split"]["valid"]
        + data_cfg["split"]["test"]
    )

    print("=== Multi-frame point cache creation ===")
    print(f"dataset: {args.dataset}")
    print(f"num_past_frames: {args.num_past_frames}")

    for seq in sequences:
        build_multiframe_cache_for_sequence(
            dataset_root=args.dataset,
            seq=seq,
            num_past_frames=args.num_past_frames,
        )

    print("Done.")


if __name__ == "__main__":
    main()