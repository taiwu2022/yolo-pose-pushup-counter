from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .pose_utils import extract_pose
from .visualize import (
    draw_skeleton,
    draw_hud,
    compose_with_simulated_views,
    draw_ground_mask_from_binary,
)
from .utils.video_io import probe_video, make_writer
from .counters.pushup import PushUpCounter, PushUpConfig
from .ground_seg import GroundSegEstimator, GroundSegConfig


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
    p.add_argument("--min-rep-duration", type=float, default=1.0, help="Minimum duration of one push-up cycle (down->up), in seconds.")
    p.add_argument("--ema-alpha", type=float, default=0.25, help="EMA smoothing alpha for elbow angle.")
    p.add_argument("--hold-frames", type=int, default=2, help="Require consecutive frames to confirm a transition.")
    p.add_argument("--min-hand-contacts", type=int, default=1, help="How many wrists must be near ground to enter ready state.")
    p.add_argument("--contact-hold-frames", type=int, default=1, help="Consecutive near-ground frames required before ready latch.")
    p.add_argument("--ready-hold-s", type=float, default=2.0, help="Keep ready state latched for N seconds after activation.")
    p.add_argument("--hand-drop-fraction-start", type=float, default=0.90, help="Relative drop fraction of wrist-ground distance to start ready.")
    p.add_argument("--ground-contact-tol", type=float, default=24.0, help="Absolute wrist-ground distance tolerance (pixels).")
    p.add_argument("--ground-model-alpha", type=float, default=0.18, help="EMA alpha for temporal ground-line model.")
    p.add_argument("--ground-model-min-points", type=int, default=14, help="Min sampled floor points to update ground-line model.")
    p.add_argument("--head-drop-fraction-or", type=float, default=0.45, help="Reserved compatibility param for head-drop OR down trigger.")
    p.add_argument("--head-recover-ratio-or", type=float, default=0.85, help="Reserved compatibility param for head-drop OR up trigger.")
    p.add_argument("--show", action="store_true", help="Show realtime window (can be slow).")
    p.add_argument("--no-skeleton", action="store_true", help="Do not draw skeleton.")
    p.add_argument("--no-hud", action="store_true", help="Do not draw HUD.")
    p.add_argument("--no-sim-views", action="store_true", help="Do not append simulated viewpoint panels on the right.")
    p.add_argument("--no-ground-mask", action="store_true", help="Do not draw estimated ground mask.")
    p.add_argument("--ground-alpha", type=float, default=0.28, help="Ground mask alpha blending.")
    p.add_argument("--ground-seg-model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="HF model for semantic ground segmentation.")
    p.add_argument("--ground-seg-stride", type=int, default=2, help="Run segmentation every N frames and reuse last mask.")
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
        min_rep_duration_s=float(args.min_rep_duration),
        ema_alpha=float(args.ema_alpha),
        hold_frames=int(args.hold_frames),
        min_hand_contacts=int(args.min_hand_contacts),
        contact_hold_frames=int(args.contact_hold_frames),
        ready_hold_s=float(args.ready_hold_s),
        hand_drop_fraction_start=float(args.hand_drop_fraction_start),
        ground_contact_tol_px=float(args.ground_contact_tol),
        ground_model_alpha=float(args.ground_model_alpha),
        ground_model_min_points=int(args.ground_model_min_points),
        head_drop_fraction_or=float(args.head_drop_fraction_or),
        head_recover_ratio_or=float(args.head_recover_ratio_or),
    )
    counter = PushUpCounter(fps=meta.fps, cfg=cfg)

    model = YOLO(args.weights)
    ground_seg = GroundSegEstimator(
        cfg=GroundSegConfig(
            model_name=str(args.ground_seg_model),
            stride=max(1, int(args.ground_seg_stride)),
        )
    )

    panel_w = max(360, int(meta.width * 0.42)) if not args.no_sim_views else 0
    out_w = meta.width + panel_w
    writer = make_writer(out_path, meta.fps, out_w, meta.height)

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

        seg_mask = ground_seg.update(frame) if ground_seg is not None else None

        det = extract_pose(result)
        if det is not None:
            counter.update(det.xy, det.conf, frame_idx=frame_idx, ground_mask=seg_mask)
            if not args.no_skeleton:
                draw_skeleton(frame, det.xy, det.conf, conf_th=float(args.kpt_conf))

        if not args.no_ground_mask:
            draw_ground_mask_from_binary(frame, seg_mask, alpha=float(args.ground_alpha))

        if not args.no_hud:
            draw_hud(
                frame,
                count=counter.count,
                phase=counter.phase,
                angle=counter.last_angle,
                head_ground_dist=counter.last_head_ground_ratio,
                ready_to_count=counter.ready_to_count,
                fps=meta.fps,
                frame_idx=frame_idx,
            )

        out_frame = frame
        if not args.no_sim_views:
            det_xy = det.xy if det is not None else None
            det_conf = det.conf if det is not None else None
            out_frame = compose_with_simulated_views(
                frame=frame,
                kpt_xy=det_xy,
                kpt_conf=det_conf,
                conf_th=float(args.kpt_conf),
            )

        writer.write(out_frame)

        if args.show:
            cv2.imshow("pushup counter", out_frame)
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
