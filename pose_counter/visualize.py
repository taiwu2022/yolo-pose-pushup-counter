from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .constants import KPT, SKELETON_EDGES


HEAD_KPTS = {
    KPT["nose"],
    KPT["left_eye"],
    KPT["right_eye"],
    KPT["left_ear"],
    KPT["right_ear"],
}


@lru_cache(maxsize=2)
def _load_head_asset(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3:
        return None
    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    if img.shape[2] != 4:
        return None
    return img


def _paste_rgba_center(dst_bgr: np.ndarray, src_rgba: np.ndarray, center_xy: tuple[float, float], out_size: int, angle_deg: float) -> None:
    out_size = int(max(8, out_size))
    src = cv2.resize(src_rgba, (out_size, out_size), interpolation=cv2.INTER_AREA)

    c = (out_size * 0.5, out_size * 0.5)
    m = cv2.getRotationMatrix2D(c, float(angle_deg), 1.0)
    rot = cv2.warpAffine(
        src,
        m,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    x0 = int(round(center_xy[0] - out_size * 0.5))
    y0 = int(round(center_xy[1] - out_size * 0.5))
    x1 = x0 + out_size
    y1 = y0 + out_size

    h, w = dst_bgr.shape[:2]
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(w, x1)
    cy1 = min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    patch = rot[sy0:sy1, sx0:sx1]
    rgb = patch[:, :, :3].astype(np.float32)
    a = (patch[:, :, 3:4].astype(np.float32) / 255.0)
    dst = dst_bgr[cy0:cy1, cx0:cx1].astype(np.float32)
    out = rgb * a + dst * (1.0 - a)
    dst_bgr[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)


def draw_skeleton(
    frame: np.ndarray,
    kpt_xy: np.ndarray,
    kpt_conf: np.ndarray,
    conf_th: float = 0.25,
):
    """Draw COCO skeleton on frame."""
    # lines
    for i, j in SKELETON_EDGES:
        if kpt_conf[i] >= conf_th and kpt_conf[j] >= conf_th:
            p1 = tuple(int(v) for v in kpt_xy[i])
            p2 = tuple(int(v) for v in kpt_xy[j])
            cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    # points
    for idx in range(kpt_xy.shape[0]):
        if kpt_conf[idx] >= conf_th:
            p = tuple(int(v) for v in kpt_xy[idx])
            cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    count: int,
    phase: str,
    angle: float | None = None,
    head_ground_dist: float | None = None,
    ready_to_count: bool | None = None,
    fps: float | None = None,
    frame_idx: int | None = None,
):
    lines = [f"Count: {count}", f"Phase: {phase}"]
    if ready_to_count is not None:
        lines.append("Ready to count push up" if ready_to_count else "Not Ready to count push up")
    if angle is not None:
        lines.append(f"Elbow angle: {angle:.1f} deg")
    if head_ground_dist is not None:
        lines.append(f"Head-ground dist: {head_ground_dist:.3f}")
    if fps is not None and frame_idx is not None:
        t = frame_idx / fps
        lines.append(f"Time: {t:.2f}s")

    h, w = frame.shape[:2]
    x = int(w * 0.18)
    y = int(h * 0.14)
    line_h = 34
    box_w = 340
    box_h = line_h * len(lines) + 18

    # Semi-transparent backdrop so text remains readable on busy frames.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 16, y - 28), (x - 16 + box_w, y - 28 + box_h), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

    for s in lines:
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (20, 20, 20), 1, cv2.LINE_AA)
        y += line_h


def _normalize_pose_to_body_frame(
    kpt_xy: np.ndarray,
    kpt_conf: np.ndarray,
    conf_th: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a pseudo-3D body-centric skeleton from 2D keypoints.

    Returns:
      points_3d: (K, 3) pseudo-3D points in normalized body coordinates
      valid: (K,) boolean mask for points above confidence threshold
    """
    valid = kpt_conf >= conf_th
    k = int(kpt_xy.shape[0])
    points_3d = np.zeros((k, 3), dtype=np.float32)

    # Prefer torso anchors for robust normalization.
    torso_ids = [KPT["left_hip"], KPT["right_hip"], KPT["left_shoulder"], KPT["right_shoulder"]]
    torso_valid = [idx for idx in torso_ids if valid[idx]]
    if torso_valid:
        origin = np.mean(kpt_xy[torso_valid], axis=0)
    else:
        fallback = np.where(valid)[0]
        if len(fallback) == 0:
            return points_3d, valid
        origin = np.mean(kpt_xy[fallback], axis=0)

    hip_center = None
    if valid[KPT["left_hip"]] and valid[KPT["right_hip"]]:
        hip_center = 0.5 * (kpt_xy[KPT["left_hip"]] + kpt_xy[KPT["right_hip"]])
    shoulder_center = None
    if valid[KPT["left_shoulder"]] and valid[KPT["right_shoulder"]]:
        shoulder_center = 0.5 * (kpt_xy[KPT["left_shoulder"]] + kpt_xy[KPT["right_shoulder"]])

    scale = 0.0
    if hip_center is not None and shoulder_center is not None:
        scale = float(np.linalg.norm(shoulder_center - hip_center))
    if scale < 1e-3:
        idx = np.where(valid)[0]
        if len(idx) >= 2:
            mins = np.min(kpt_xy[idx], axis=0)
            maxs = np.max(kpt_xy[idx], axis=0)
            scale = float(np.linalg.norm(maxs - mins) * 0.35)
    if scale < 1e-3:
        scale = 100.0

    # Add a lightweight depth heuristic: left/right keypoints get opposite z signs.
    left_ids = {KPT["left_eye"], KPT["left_ear"], KPT["left_shoulder"], KPT["left_elbow"], KPT["left_wrist"], KPT["left_hip"], KPT["left_knee"], KPT["left_ankle"]}
    right_ids = {KPT["right_eye"], KPT["right_ear"], KPT["right_shoulder"], KPT["right_elbow"], KPT["right_wrist"], KPT["right_hip"], KPT["right_knee"], KPT["right_ankle"]}

    for idx in range(k):
        if not valid[idx]:
            continue
        rel = (kpt_xy[idx] - origin) / scale
        x_b = float(rel[0])
        y_b = float(-rel[1])  # image y-down -> body y-up
        z_b = 0.0
        if idx in left_ids:
            z_b = -0.25
        elif idx in right_ids:
            z_b = 0.25
        points_3d[idx] = np.array([x_b, y_b, z_b], dtype=np.float32)

    return points_3d, valid


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))

    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return rx @ ry


def _draw_view_tile(
    tile: np.ndarray,
    points_3d: np.ndarray,
    valid: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    title: str,
) -> None:
    h, w = tile.shape[:2]
    tile[:] = (22, 22, 22)
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (70, 70, 70), 1, cv2.LINE_AA)

    if not np.any(valid):
        cv2.putText(tile, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(tile, "No pose", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
        return

    r = _rotation_matrix(yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    pr = points_3d @ r.T
    xy = pr[:, :2].copy()

    idx = np.where(valid)[0]
    mins = np.min(xy[idx], axis=0)
    maxs = np.max(xy[idx], axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    sx = (w * 0.75) / span[0]
    sy = (h * 0.7) / span[1]
    s = float(min(sx, sy))

    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    xy[:, 0] = (xy[:, 0] - cx) * s + w * 0.5
    # Convert body y-up to image y-down to avoid upside-down rendering.
    xy[:, 1] = -(xy[:, 1] - cy) * s + h * 0.56

    # Elliptical contact surface (instead of fixed parallelogram).
    support_ids = [KPT["left_wrist"], KPT["right_wrist"], KPT["left_ankle"], KPT["right_ankle"]]
    support = [sid for sid in support_ids if valid[sid]]
    if len(support) >= 2:
        pts = xy[np.array(support, dtype=np.int32)]
        cx_e = float(np.mean(pts[:, 0]))
        cy_e = float(np.max(pts[:, 1])) + 8.0
        span_x = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        ax_x = int(max(28.0, min(w * 0.40, span_x * 0.65 + 36.0)))
        ax_y = int(max(12.0, ax_x * 0.32))
        overlay = tile.copy()
        cv2.ellipse(overlay, (int(cx_e), int(cy_e)), (ax_x, ax_y), 0.0, 0, 360, (88, 150, 96), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.50, tile, 0.50, 0.0, tile)
        cv2.ellipse(tile, (int(cx_e), int(cy_e)), (ax_x, ax_y), 0.0, 0, 360, (150, 235, 168), 1, cv2.LINE_AA)

    for i, j in SKELETON_EDGES:
        if i in HEAD_KPTS or j in HEAD_KPTS:
            continue
        if valid[i] and valid[j]:
            p1 = (int(xy[i, 0]), int(xy[i, 1]))
            p2 = (int(xy[j, 0]), int(xy[j, 1]))
            cv2.line(tile, p1, p2, (0, 210, 255), 2, cv2.LINE_AA)
    for i in idx:
        if i in HEAD_KPTS:
            continue
        p = (int(xy[i, 0]), int(xy[i, 1]))
        cv2.circle(tile, p, 3, (80, 255, 120), -1, cv2.LINE_AA)

    # Replace head keypoints with portrait sprite.
    repo_root = Path(__file__).resolve().parents[1]
    head_asset = _load_head_asset(str(repo_root / "data" / "head_wuyanzu.png"))
    if head_asset is not None:
        shoulder_ids = [KPT["left_shoulder"], KPT["right_shoulder"]]
        shoulder_valid = [i for i in shoulder_ids if valid[i]]
        if len(shoulder_valid) == 2:
            sh_l = xy[KPT["left_shoulder"]]
            sh_r = xy[KPT["right_shoulder"]]
            shoulder_c = 0.5 * (sh_l + sh_r)
            shoulder_w = float(np.linalg.norm(sh_r - sh_l))
            roll = float(np.degrees(np.arctan2(sh_r[1] - sh_l[1], sh_r[0] - sh_l[0])))
        else:
            shoulder_c = np.array([float(w) * 0.5, float(h) * 0.45], dtype=np.float32)
            shoulder_w = float(min(w, h) * 0.18)
            roll = 0.0

        head_ids = [KPT["nose"], KPT["left_eye"], KPT["right_eye"], KPT["left_ear"], KPT["right_ear"]]
        head_valid = [i for i in head_ids if valid[i]]
        if head_valid:
            head_c = np.mean(xy[np.array(head_valid, dtype=np.int32)], axis=0)
        else:
            head_c = np.array([shoulder_c[0], shoulder_c[1] - shoulder_w * 0.65], dtype=np.float32)

        size = int(max(36.0, shoulder_w * 1.05))
        center = (float(head_c[0]), float(head_c[1] - size * 0.08))
        _paste_rgba_center(tile, head_asset, center_xy=center, out_size=size, angle_deg=float(np.clip(roll * 0.8, -30.0, 30.0)))

    cv2.putText(tile, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)


def compose_with_simulated_views(
    frame: np.ndarray,
    kpt_xy: np.ndarray | None,
    kpt_conf: np.ndarray | None,
    conf_th: float = 0.25,
    panel_ratio: float = 0.42,
) -> np.ndarray:
    """Compose output frame with right-side pseudo-3D simulated viewpoint panels."""
    h, w = frame.shape[:2]
    panel_w = max(360, int(w * panel_ratio))
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (14, 14, 14)

    # Strong border around the original video area for clearer separation.
    cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(canvas, (8, 8), (w - 9, h - 9), (0, 170, 255), 1, cv2.LINE_AA)

    margin = 10
    inner_x0 = w + margin
    inner_w = panel_w - margin * 2
    inner_h = h - margin * 2

    if kpt_xy is None or kpt_conf is None:
        points_3d = np.zeros((17, 3), dtype=np.float32)
        valid = np.zeros((17,), dtype=bool)
    else:
        points_3d, valid = _normalize_pose_to_body_frame(kpt_xy, kpt_conf, conf_th=conf_th)

    tile = canvas[margin : margin + inner_h, inner_x0 : inner_x0 + inner_w]
    _draw_view_tile(tile, points_3d, valid, yaw_deg=0.0, pitch_deg=8.0, title="Front view")

    return canvas


def draw_ground_mask_from_binary(
    frame: np.ndarray,
    mask01: np.ndarray | None,
    alpha: float = 0.28,
    fill_color: tuple[int, int, int] = (40, 170, 40),
) -> None:
    """Draw ground mask from a binary array (H, W), values {0,1}."""
    if mask01 is None:
        return
    if mask01.ndim != 2:
        return
    h, w = frame.shape[:2]
    if mask01.shape[0] != h or mask01.shape[1] != w:
        return

    mask = (mask01 > 0).astype(np.uint8)
    if mask.sum() == 0:
        return

    overlay = frame.copy()
    overlay[mask > 0] = fill_color
    a_clamped = max(0.0, min(1.0, float(alpha)))
    cv2.addWeighted(overlay, a_clamped, frame, 1.0 - a_clamped, 0.0, frame)
