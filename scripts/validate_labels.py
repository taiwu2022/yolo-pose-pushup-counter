from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Validate YOLO Pose labels (txt) for basic format sanity.")
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--labels-dir", type=str, required=True)
    p.add_argument("--k", type=int, default=17, help="Number of keypoints.")
    p.add_argument("--dims", type=int, default=3, help="Dims per keypoint (2 or 3). COCO pose uses 3.")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not imgs:
        raise SystemExit(f"No images in {images_dir}")

    # Expected tokens per line: class + 4 bbox + k*dims
    exp = 1 + 4 + args.k * args.dims

    bad = 0
    missing = 0
    for ip in imgs:
        lp = labels_dir / (ip.stem + ".txt")
        if not lp.exists():
            missing += 1
            continue
        lines = [ln.strip() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) != exp:
                bad += 1
                print(f"[BAD] {lp.name}: expected {exp} tokens, got {len(parts)}")
                break

            # Basic range checks on normalized coords
            try:
                vals = list(map(float, parts[1:]))
            except ValueError:
                bad += 1
                print(f"[BAD] {lp.name}: non-float tokens")
                break

            # bbox x,y,w,h must be 0..1
            bx, by, bw, bh = vals[0], vals[1], vals[2], vals[3]
            if not (0 <= bx <= 1 and 0 <= by <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                bad += 1
                print(f"[BAD] {lp.name}: bbox out of range")
                break

    print(f"[DONE] images={len(imgs)}, missing_labels={missing}, bad_labels={bad}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
