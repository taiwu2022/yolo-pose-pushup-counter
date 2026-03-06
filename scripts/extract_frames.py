from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from videos for labeling/training.")
    p.add_argument("--video-dir", type=str, required=True, help="Directory containing input videos (*.mp4, *.mov, ...).")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for frames.")
    p.add_argument("--every-n", type=int, default=3, help="Save one frame every N frames (e.g. 3 for ~10fps at 30fps).")
    p.add_argument("--max-per-video", type=int, default=0, help="Max frames per video (0 = no limit).")
    p.add_argument("--resize", type=int, default=0, help="Optional resize long side to this value (0 = keep).")
    return p.parse_args()


def iter_videos(video_dir: Path):
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    for p in sorted(video_dir.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(video_dir))
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[WARN] cannot open: {vp}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total if total > 0 else None, desc=f"Extract {vp.name}")
        saved = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % int(args.every_n) == 0:
                if args.resize and args.resize > 0:
                    h, w = frame.shape[:2]
                    scale = args.resize / float(max(h, w))
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                out_name = f"{vp.stem}_frame{idx:06d}.jpg"
                cv2.imwrite(str(out_dir / out_name), frame)
                saved += 1
                if args.max_per_video and saved >= int(args.max_per_video):
                    break
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        print(f"[OK] {vp.name}: saved {saved} frames -> {out_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
