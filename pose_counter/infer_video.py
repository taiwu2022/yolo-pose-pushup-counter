from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .pose_utils import extract_pose
from .visualize import draw_skeleton, draw_hud
from .utils.video_io import probe_video, make_writer
from .counters.pushup import PushUpCounter, PushUpConfig


def parse_args():
    p = argparse.ArgumentParser(description="YOLO Pose push-up counter (video -> counted video + reps csv).")
    p.add_argument("--source", type=str, required=True, help="Input video path.")
    p.add_argument("--output", type=str, default="", help="Output video path (mp4).")
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="Pose weights, e.g. yolo11n-pose.pt or weights/best.pt.")
    p.add_argument("--device", type=str, default="", help="Device, e.g. '0' or 'cpu'. Empty for auto.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--max-det", type=int, default=5, help="Max detections per frame.")
    p.add_argument("--kpt-conf", type=float, default=0.25, help="Keypoint confidence gate for counting/plotting.")
    p.add_argument("--down-th", type=float, default=90.0, help="Elbow angle threshold for DOWN.")
    p.add_argument("--up-th", type=float, default=160.0, help="Elbow angle threshold for UP / count.")
    p.add_argument("--min-interval", type=float, default=0.35, help="Min interval between reps (seconds).")
    p.add_argument("--ema-alpha", type=float, default=0.25, help="EMA smoothing alpha for elbow angle.")
    p.add_argument("--hold-frames", type=int, default=2, help="Require consecutive frames to confirm a transition.")
    p.add_argument("--show", action="store_true", help="Show realtime window (can be slow).")
    p.add_argument("--no-skeleton", action="store_true", help="Do not draw skeleton.")
    p.add_argument("--no-hud", action="store_true", help="Do not draw HUD.")
    p.add_argument("--person", type=str, default="largest", choices=["largest"], help="Person selection strategy (single-person default).")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(src)

    meta = probe_video(src)
    out_path = Path(args.output) if args.output else Path("outputs") / f"{src.stem}_counted.mp4"
    csv_path = out_path.with_suffix("").with_name(out_path.stem + "_reps.csv")

    cfg = PushUpConfig(
        down_th=float(args.down_th),
        up_th=float(args.up_th),
        kpt_conf_th=float(args.kpt_conf),
        min_interval_s=float(args.min_interval),
        ema_alpha=float(args.ema_alpha),
        hold_frames=int(args.hold_frames),
    )
    counter = PushUpCounter(fps=meta.fps, cfg=cfg)

    model = YOLO(args.weights)

    writer = make_writer(out_path, meta.fps, meta.width, meta.height)

    # Stream results frame-by-frame for efficiency
    stream = model.predict(
        source=str(src),
        stream=True,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=args.device if args.device else None,
        verbose=False,
    )

    pbar = tqdm(total=meta.frame_count if meta.frame_count > 0 else None, desc="Processing")
    frame_idx = 0
    for result in stream:
        frame = result.orig_img  # numpy (BGR)
        if frame is None:
            break

        det = extract_pose(result)
        if det is not None:
            counter.update(det.xy, det.conf, frame_idx=frame_idx)
            if not args.no_skeleton:
                draw_skeleton(frame, det.xy, det.conf, conf_th=float(args.kpt_conf))

        if not args.no_hud:
            draw_hud(
                frame,
                count=counter.count,
                phase=counter.phase,
                angle=counter.last_angle,
                fps=meta.fps,
                frame_idx=frame_idx,
            )

        writer.write(frame)

        if args.show:
            cv2.imshow("pushup counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        frame_idx += 1
        pbar.update(1)
    pbar.close()

    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    counter.save_csv(csv_path)
    print(f"Saved video: {out_path}")
    print(f"Saved reps : {csv_path}")
    print(f"Total reps: {counter.count}")


if __name__ == "__main__":
    main()
