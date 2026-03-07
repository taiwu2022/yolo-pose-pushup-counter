from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_counter.pose_utils import extract_pose
from pose_counter.visualize import draw_skeleton
from pose_counter.constants import KPT


def parse_args():
    p = argparse.ArgumentParser(description="Run MMAction2 skeleton push-up model on a video and render per-frame probs.")
    p.add_argument("--video", type=str, required=True, help="Input video path.")
    p.add_argument("--output", type=str, required=True, help="Output rendered video path.")
    p.add_argument("--csv", type=str, default=None, help="Optional output CSV for per-frame probability.")
    p.add_argument("--mmaction2-dir", type=str, default="/Users/taiwu/Documents/GitHub/mmaction2")
    p.add_argument("--config", type=str, default="/Users/taiwu/Documents/GitHub/mmaction2/configs/pushup_stgcn_pseudo.py")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/taiwu/Documents/GitHub/mmaction2/work_dirs/pushup_stgcn_pseudo/best_acc_top1_epoch_17.pth",
    )
    p.add_argument("--yolo-weights", type=str, default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--stride", type=int, default=12, help="Sliding window stride for action inference.")
    p.add_argument("--device", type=str, default="cpu", help="MMAction2 device, e.g. cpu or mps.")
    p.add_argument("--kpt-conf", type=float, default=0.25, help="Keypoint conf threshold for drawing.")
    p.add_argument("--push-th", type=float, default=0.72, help="Upper threshold to enter push_up state.")
    p.add_argument("--not-push-th", type=float, default=0.45, help="Lower threshold to return to not_pushup.")
    p.add_argument("--smooth-alpha", type=float, default=0.15, help="EMA alpha for smoothed probability.")
    p.add_argument("--min-push-s", type=float, default=0.8, help="Minimum push_up-state duration to count 1 event.")
    p.add_argument("--elbow-down-th", type=float, default=120.0, help="Elbow angle threshold to enter down.")
    p.add_argument("--elbow-up-th", type=float, default=157.0, help="Elbow angle threshold to finish rep on up.")
    p.add_argument("--min-rep-s", type=float, default=1.0, help="Minimum down->up duration for one valid rep.")
    return p.parse_args()


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    window_prob: float,
    smooth_prob: float,
    state: str,
    count: int,
    elbow_angle: float,
):
    if np.isnan(window_prob):
        txt_w = "window prob: N/A"
        color = (180, 180, 180)
    else:
        txt_w = f"window prob: {window_prob:.3f}"
        color = (0, 220, 0) if state == "push_up" else (0, 180, 255)

    txt_s = "smooth prob: N/A" if np.isnan(smooth_prob) else f"smooth prob: {smooth_prob:.3f}"
    txt_e = "elbow angle: N/A" if np.isnan(elbow_angle) else f"elbow angle: {elbow_angle:.1f}"
    cv2.rectangle(frame, (18, 18), (520, 218), (20, 20, 20), thickness=-1)
    cv2.putText(frame, f"frame: {frame_idx}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, txt_w, (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, txt_s, (30, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"state: {state}", (30, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, txt_e, (30, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"push_up count: {count}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba) + 1e-9)
    nbc = float(np.linalg.norm(bc) + 1e-9)
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _elbow_angle(xy: np.ndarray | None, conf: np.ndarray | None, conf_th: float) -> float:
    if xy is None or conf is None:
        return float("nan")
    vals = []
    for s, e, w in [
        (KPT["left_shoulder"], KPT["left_elbow"], KPT["left_wrist"]),
        (KPT["right_shoulder"], KPT["right_elbow"], KPT["right_wrist"]),
    ]:
        if conf[s] >= conf_th and conf[e] >= conf_th and conf[w] >= conf_th:
            vals.append(_angle_deg(xy[s], xy[e], xy[w]))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Needed for loading checkpoints saved by MMEngine under torch>=2.6 defaults.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    mmaction2_dir = Path(args.mmaction2_dir).resolve()
    if str(mmaction2_dir) not in sys.path:
        sys.path.insert(0, str(mmaction2_dir))

    from mmengine.dataset import Compose
    from mmaction.apis import inference_skeleton, init_recognizer

    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    clip_len = 48
    test_pipeline = None
    try:
        # Keep in sync with your config.
        clip_len = int(model.cfg.val_pipeline[2]["clip_len"])
        test_pipeline = Compose(model.cfg.val_pipeline)
    except Exception:
        # Fallback for configs without val_pipeline symbol.
        try:
            pipeline_cfg = model.cfg.test_dataloader["dataset"]["pipeline"]
            clip_len = int(pipeline_cfg[2]["clip_len"])
            test_pipeline = Compose(pipeline_cfg)
        except Exception:
            test_pipeline = None

    yolo = YOLO(args.yolo_weights)
    stream = yolo.predict(
        source=str(video_path),
        stream=True,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=5,
        verbose=False,
    )

    pose_results = []
    draw_xy = []
    draw_conf = []
    draw_bbox = []
    H = W = None
    for r in stream:
        if r.orig_img is None:
            continue
        if H is None or W is None:
            H, W = r.orig_img.shape[:2]
        det = extract_pose(r)
        if det is None:
            pose_results.append(
                {
                    "keypoints": np.zeros((0, 17, 2), dtype=np.float32),
                    "keypoint_scores": np.zeros((0, 17), dtype=np.float32),
                }
            )
            draw_xy.append(None)
            draw_conf.append(None)
            draw_bbox.append(None)
        else:
            pose_results.append(
                {
                    "keypoints": np.expand_dims(det.xy.astype(np.float32), axis=0),
                    "keypoint_scores": np.expand_dims(det.conf.astype(np.float32), axis=0),
                }
            )
            draw_xy.append(det.xy.astype(np.float32))
            draw_conf.append(det.conf.astype(np.float32))
            draw_bbox.append(det.bbox_xyxy.astype(np.float32) if det.bbox_xyxy is not None else None)

    if H is None or W is None or len(pose_results) == 0:
        raise SystemExit(f"Failed to decode frames from {video_path}")

    n = len(pose_results)
    stride = max(1, int(args.stride))
    sum_prob = np.zeros((n,), dtype=np.float32)
    cnt = np.zeros((n,), dtype=np.float32)

    if n < clip_len:
        print(f"[WARN] video is shorter than clip_len={clip_len}, no action windows inferred.")
    else:
        for st in range(0, n - clip_len + 1, stride):
            ed = st + clip_len
            res = inference_skeleton(model, pose_results[st:ed], (H, W), test_pipeline=test_pipeline)
            prob_push = float(res.pred_score[1].item())
            sum_prob[st:ed] += prob_push
            cnt[st:ed] += 1.0

    probs = np.full((n,), np.nan, dtype=np.float32)
    valid = cnt > 0
    probs[valid] = sum_prob[valid] / cnt[valid]
    # EMA smoothing to reduce jitter.
    smooth = np.full((n,), np.nan, dtype=np.float32)
    last = np.nan
    alpha = float(np.clip(args.smooth_alpha, 0.01, 1.0))
    for i in range(n):
        p = probs[i]
        if np.isnan(p):
            smooth[i] = last
            continue
        if np.isnan(last):
            last = p
        else:
            last = alpha * p + (1.0 - alpha) * last
        smooth[i] = last

    # Hysteresis state from classifier probabilities.
    state = "not_pushup"
    state_start = 0
    prob_count = 0
    # Rep counter from elbow angle trajectory (more suitable for reps).
    elbow_series = np.array([_elbow_angle(draw_xy[i], draw_conf[i], float(args.kpt_conf)) for i in range(n)], dtype=np.float32)
    rep_count = 0
    rep_phase = "up"
    rep_start_idx = -1
    fps_eff = 30.0

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    fps_eff = float(fps)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(W), int(H)),
    )

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = float(probs[i]) if i < len(probs) else float("nan")
        ps = float(smooth[i]) if i < len(smooth) else float("nan")

        if not np.isnan(ps):
            if state == "not_pushup" and ps >= float(args.push_th):
                state = "push_up"
                state_start = i
            elif state == "push_up" and ps <= float(args.not_push_th):
                dur_s = (i - state_start) / max(1e-6, fps_eff)
                if dur_s >= float(args.min_push_s):
                    prob_count += 1
                state = "not_pushup"
                state_start = i

        # Elbow-based repetition counting.
        ea = float(elbow_series[i]) if i < len(elbow_series) else float("nan")
        if not np.isnan(ea):
            if rep_phase == "up" and ea <= float(args.elbow_down_th):
                rep_phase = "down"
                rep_start_idx = i
            elif rep_phase == "down" and ea >= float(args.elbow_up_th):
                dur_s = (i - rep_start_idx) / max(1e-6, fps_eff) if rep_start_idx >= 0 else 0.0
                if dur_s >= float(args.min_rep_s):
                    rep_count += 1
                rep_phase = "up"
                rep_start_idx = -1

        # Draw bbox + skeleton.
        if i < len(draw_bbox) and draw_bbox[i] is not None:
            x1, y1, x2, y2 = [int(v) for v in draw_bbox[i]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)
        if i < len(draw_xy) and draw_xy[i] is not None and draw_conf[i] is not None:
            draw_skeleton(frame, draw_xy[i], draw_conf[i], conf_th=float(args.kpt_conf))

        frame = draw_hud(frame, i, p, ps, state, rep_count, ea)
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("frame,window_prob,smoothed_prob,state,prob_count,elbow_angle,rep_count\n")
            running_state = "not_pushup"
            running_start = 0
            running_prob_count = 0
            running_rep_count = 0
            running_rep_phase = "up"
            running_rep_start = -1
            for j, p in enumerate(probs.tolist()):
                ps = float(smooth[j]) if j < len(smooth) else float("nan")
                if not np.isnan(ps):
                    if running_state == "not_pushup" and ps >= float(args.push_th):
                        running_state = "push_up"
                        running_start = j
                    elif running_state == "push_up" and ps <= float(args.not_push_th):
                        dur_s = (j - running_start) / max(1e-6, fps_eff)
                        if dur_s >= float(args.min_push_s):
                            running_prob_count += 1
                        running_state = "not_pushup"
                        running_start = j
                ea = float(elbow_series[j]) if j < len(elbow_series) else float("nan")
                if not np.isnan(ea):
                    if running_rep_phase == "up" and ea <= float(args.elbow_down_th):
                        running_rep_phase = "down"
                        running_rep_start = j
                    elif running_rep_phase == "down" and ea >= float(args.elbow_up_th):
                        dur_s = (j - running_rep_start) / max(1e-6, fps_eff) if running_rep_start >= 0 else 0.0
                        if dur_s >= float(args.min_rep_s):
                            running_rep_count += 1
                        running_rep_phase = "up"
                        running_rep_start = -1
                p_txt = "" if np.isnan(p) else f"{p:.6f}"
                ps_txt = "" if np.isnan(ps) else f"{ps:.6f}"
                ea_txt = "" if np.isnan(ea) else f"{ea:.3f}"
                f.write(f"{j},{p_txt},{ps_txt},{running_state},{running_prob_count},{ea_txt},{running_rep_count}\n")

    print(f"[DONE] output video: {out_path}")
    print(f"[DONE] final push_up count (elbow): {rep_count}")
    print(f"[DONE] classifier-state event count: {prob_count}")
    if args.csv:
        print(f"[DONE] csv: {args.csv}")


if __name__ == "__main__":
    main()
