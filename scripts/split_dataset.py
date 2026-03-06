from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import re
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Split frames+labels into YOLO Pose dataset structure (train/val/test).")
    p.add_argument("--frames-dir", type=str, required=True, help="Directory of extracted frames (jpg).")
    p.add_argument("--labels-dir", type=str, required=True, help="Directory of YOLO pose labels (txt) matching images.")
    p.add_argument("--out-dataset-dir", type=str, required=True, help="Output dataset directory, e.g. datasets/pushup_pose.")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def video_id_from_name(name: str) -> str:
    # Expected: <video>_frame000123.jpg
    m = re.match(r"^(.*)_frame\d+\.(jpg|jpeg|png)$", name, re.IGNORECASE)
    return m.group(1) if m else Path(name).stem


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dataset_dir)

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("train+val+test must sum to 1.0")

    imgs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not imgs:
        raise SystemExit(f"No images found in {frames_dir}")

    # Group by original video name to avoid leakage
    groups = defaultdict(list)
    for ip in imgs:
        vid = video_id_from_name(ip.name)
        groups[vid].append(ip)

    vids = list(groups.keys())
    random.Random(args.seed).shuffle(vids)

    n = len(vids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_vids = set(vids[:n_train])
    val_vids = set(vids[n_train:n_train+n_val])
    test_vids = set(vids[n_train+n_val:])

    def split_of(vid: str) -> str:
        if vid in train_vids: return "train"
        if vid in val_vids: return "val"
        return "test"

    # Create dirs
    for split in ["train","val","test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing_labels = 0
    copied = 0
    for vid, frames in groups.items():
        split = split_of(vid)
        for ip in frames:
            lp = labels_dir / (ip.stem + ".txt")
            if not lp.exists():
                missing_labels += 1
                continue
            shutil.copy2(ip, out_dir / "images" / split / ip.name)
            shutil.copy2(lp, out_dir / "labels" / split / lp.name)
            copied += 1

    print(f"[OK] videos: {n} -> train {len(train_vids)}, val {len(val_vids)}, test {len(test_vids)}")
    print(f"[OK] copied pairs: {copied}")
    if missing_labels:
        print(f"[WARN] missing labels for {missing_labels} images (skipped)")


if __name__ == "__main__":
    main()
