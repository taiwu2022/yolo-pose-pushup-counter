from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import numpy as np

# Optional dependency only used for writing pkl in a format MMAction2 loads well.
try:
    import mmengine  # type: ignore
except Exception:
    mmengine = None

from ultralytics import YOLO

# Ensure project root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_counter.pose_utils import extract_pose
from pose_counter.constants import KPT


@dataclass
class VideoSeq:
    video: Path
    fps: float
    keypoint: np.ndarray  # (T, 17, 2) normalized [0,1]
    score: np.ndarray     # (T, 17)
    elbow: np.ndarray     # (T,) nan when unavailable


def parse_args():
    p = argparse.ArgumentParser(description="Build pseudo-labeled MMAction2 skeleton dataset for push-up.")
    p.add_argument("--video-dir", type=str, required=True, help="Directory with raw videos.")
    p.add_argument("--out-dir", type=str, default="datasets/mmaction2_pushup", help="Output dataset directory.")
    p.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO pose weights.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--clip-len", type=int, default=48, help="Frames per clip for GCN model.")
    p.add_argument("--stride", type=int, default=12, help="Sliding window stride.")
    p.add_argument("--pos-angle-range", type=float, default=28.0, help="Pseudo-positive threshold: elbow angle range in clip.")
    p.add_argument("--neg-angle-range", type=float, default=10.0, help="Pseudo-negative threshold: elbow angle range in clip.")
    p.add_argument("--neg-ratio", type=float, default=1.0, help="Max negatives per positive.")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument(
        "--split-mode",
        type=str,
        choices=["random", "video"],
        default="video",
        help="Split mode: random clip split or strict video-level split.",
    )
    p.add_argument(
        "--val-videos",
        type=str,
        nargs="*",
        default=None,
        help="Optional video basenames for val split, e.g. test.MOV 6.MOV. Only used in video split mode.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def iter_videos(video_dir: Path):
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".MOV", ".MP4"}
    for p in sorted(video_dir.rglob("*")):
        if p.suffix in exts:
            yield p


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba) + 1e-9)
    nbc = float(np.linalg.norm(bc) + 1e-9)
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def elbow_angle_frame(xy: np.ndarray, conf: np.ndarray, conf_th: float = 0.25) -> float:
    vals = []
    for s, e, w in [
        (KPT["left_shoulder"], KPT["left_elbow"], KPT["left_wrist"]),
        (KPT["right_shoulder"], KPT["right_elbow"], KPT["right_wrist"]),
    ]:
        if conf[s] >= conf_th and conf[e] >= conf_th and conf[w] >= conf_th:
            vals.append(angle_deg(xy[s], xy[e], xy[w]))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def extract_video_sequence(model: YOLO, video: Path, args) -> VideoSeq:
    stream = model.predict(
        source=str(video),
        stream=True,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=5,
        verbose=False,
    )

    k_list = []
    s_list = []
    e_list = []
    fps = 30.0

    for result in stream:
        frame = result.orig_img
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if getattr(result, "speed", None) is not None:
            pass
        det = extract_pose(result)
        if det is None:
            k = np.zeros((17, 2), dtype=np.float32)
            s = np.zeros((17,), dtype=np.float32)
            e = float("nan")
        else:
            k = det.xy.astype(np.float32)
            k[:, 0] = np.clip(k[:, 0] / max(1.0, float(w)), 0.0, 1.0)
            k[:, 1] = np.clip(k[:, 1] / max(1.0, float(h)), 0.0, 1.0)
            s = det.conf.astype(np.float32)
            e = elbow_angle_frame(det.xy.astype(np.float32), det.conf.astype(np.float32), conf_th=0.25)

        k_list.append(k)
        s_list.append(s)
        e_list.append(e)

    keypoint = np.asarray(k_list, dtype=np.float32)
    score = np.asarray(s_list, dtype=np.float32)
    elbow = np.asarray(e_list, dtype=np.float32)
    return VideoSeq(video=video, fps=fps, keypoint=keypoint, score=score, elbow=elbow)


def make_samples(seq: VideoSeq, clip_len: int, stride: int, pos_thr: float, neg_thr: float):
    T = int(seq.keypoint.shape[0])
    out_pos = []
    out_neg = []
    if T < clip_len:
        return out_pos, out_neg

    for st in range(0, T - clip_len + 1, stride):
        ed = st + clip_len
        e = seq.elbow[st:ed]
        e = e[~np.isnan(e)]
        if len(e) < max(8, clip_len // 6):
            continue
        rng = float(np.max(e) - np.min(e))

        sample = {
            "video": seq.video,
            "start": st,
            "end": ed,
            "keypoint": seq.keypoint[st:ed],
            "score": seq.score[st:ed],
            "angle_range": rng,
        }
        if rng >= pos_thr:
            out_pos.append(sample)
        elif rng <= neg_thr:
            out_neg.append(sample)

    return out_pos, out_neg


def cap_negatives(neg_samples: list[dict], pos_count: int, neg_ratio: float) -> list[dict]:
    max_neg = int(max(0, pos_count) * float(neg_ratio))
    out = list(neg_samples)
    random.shuffle(out)
    return out[:max_neg]


def to_mmaction_ann(sample: dict, label: int, clip_id: str):
    kp = sample["keypoint"]  # (T,17,2)
    sc = sample["score"]     # (T,17)
    T = int(kp.shape[0])
    ann = {
        "frame_dir": clip_id,
        "total_frames": T,
        "label": int(label),
        "keypoint": np.expand_dims(kp, axis=0),       # (1,T,17,2)
        "keypoint_score": np.expand_dims(sc, axis=0), # (1,T,17)
        "img_shape": (256, 256),
        "original_shape": (256, 256),
    }
    return ann


def dump_pkl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mmengine is not None:
        mmengine.dump(obj, str(path))
    else:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(obj, f)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(video_dir))
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    model = YOLO(args.weights)

    per_video_samples: dict[str, dict[str, list[dict]]] = {}

    for v in videos:
        print(f"[INFO] extracting keypoints: {v.name}")
        seq = extract_video_sequence(model, v, args)
        p, n = make_samples(
            seq,
            clip_len=int(args.clip_len),
            stride=int(args.stride),
            pos_thr=float(args.pos_angle_range),
            neg_thr=float(args.neg_angle_range),
        )
        print(f"[INFO] {v.name}: pos={len(p)} neg={len(n)}")
        per_video_samples[v.name] = {"pos": p, "neg": n}

    pos_all = [s for one in per_video_samples.values() for s in one["pos"]]
    if not pos_all:
        raise SystemExit("No pseudo-positive clips found. Lower --pos-angle-range or use more push-up videos.")

    train_data: list[tuple[dict, int]]
    val_data: list[tuple[dict, int]]
    train_video_names: list[str]
    val_video_names: list[str]

    if args.split_mode == "video":
        video_names = sorted(per_video_samples.keys())
        if args.val_videos:
            val_name_set = set(args.val_videos)
            unknown = sorted([n for n in val_name_set if n not in per_video_samples])
            if unknown:
                raise SystemExit(f"--val-videos contains unknown names: {unknown}")
            val_video_names = [n for n in video_names if n in val_name_set]
        else:
            rng = random.Random(args.seed)
            shuffled = list(video_names)
            rng.shuffle(shuffled)
            n_val_videos = max(1, int(round(len(shuffled) * float(args.val_ratio))))
            n_val_videos = min(n_val_videos, max(1, len(shuffled) - 1))
            val_video_names = sorted(shuffled[:n_val_videos])

        train_video_names = [n for n in video_names if n not in set(val_video_names)]
        if not train_video_names or not val_video_names:
            raise SystemExit("Video split produced empty train or val set. Adjust --val-ratio/--val-videos.")

        train_pos = [s for n in train_video_names for s in per_video_samples[n]["pos"]]
        train_neg = [s for n in train_video_names for s in per_video_samples[n]["neg"]]
        val_pos = [s for n in val_video_names for s in per_video_samples[n]["pos"]]
        val_neg = [s for n in val_video_names for s in per_video_samples[n]["neg"]]

        train_neg = cap_negatives(train_neg, len(train_pos), float(args.neg_ratio))
        val_neg = cap_negatives(val_neg, len(val_pos), float(args.neg_ratio))

        train_data = [(s, 1) for s in train_pos] + [(s, 0) for s in train_neg]
        val_data = [(s, 1) for s in val_pos] + [(s, 0) for s in val_neg]
        random.shuffle(train_data)
        random.shuffle(val_data)
    else:
        neg_all = [s for one in per_video_samples.values() for s in one["neg"]]
        neg_all = cap_negatives(neg_all, len(pos_all), float(args.neg_ratio))
        data = [(s, 1) for s in pos_all] + [(s, 0) for s in neg_all]  # 1=push_up, 0=other
        random.shuffle(data)
        n_val = max(1, int(len(data) * float(args.val_ratio))) if len(data) > 5 else 1
        val_data = data[:n_val]
        train_data = data[n_val:]
        train_video_names = sorted({Path(s["video"]).name for s, _ in train_data})
        val_video_names = sorted({Path(s["video"]).name for s, _ in val_data})

    if not train_data or not val_data:
        raise SystemExit("Empty train or val clips after split. Adjust thresholds/ratio.")

    def build_split(split_data, split_name: str):
        anns = []
        for i, (s, label) in enumerate(split_data):
            clip_id = f"{split_name}_{i:06d}"
            anns.append(to_mmaction_ann(s, label=label, clip_id=clip_id))
        return anns

    train_anns = build_split(train_data, "train")
    val_anns = build_split(val_data, "val")

    dump_pkl(out_dir / "train.pkl", train_anns)
    dump_pkl(out_dir / "val.pkl", val_anns)

    classes = ["other", "push_up"]
    (out_dir / "label_map.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    summary = {
        "videos": [str(v) for v in videos],
        "split_mode": args.split_mode,
        "train_videos": train_video_names,
        "val_videos": val_video_names,
        "num_train": len(train_anns),
        "num_val": len(val_anns),
        "num_pos": int(sum(1 for _, y in train_data + val_data if y == 1)),
        "num_neg": int(sum(1 for _, y in train_data + val_data if y == 0)),
        "clip_len": int(args.clip_len),
        "stride": int(args.stride),
        "classes": classes,
    }
    import json

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] dataset -> {out_dir}")
    print(f"[DONE] train={len(train_anns)} val={len(val_anns)} classes={classes}")


if __name__ == "__main__":
    main()
