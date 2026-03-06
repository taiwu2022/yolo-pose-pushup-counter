from __future__ import annotations

import cv2
import numpy as np

from .constants import KPT, SKELETON_EDGES


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
    fps: float | None = None,
    frame_idx: int | None = None,
):
    lines = [f"Count: {count}", f"Phase: {phase}"]
    if angle is not None:
        lines.append(f"Elbow angle: {angle:.1f} deg")
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

    # Simulated ground as a tilted parallelogram for a stable "virtual world" cue.
    base_y = int(h * 0.70)
    near_y = int(h * 0.94)
    skew = int(w * 0.16)
    ground_poly = np.array(
        [
            [max(0, 8 + skew), base_y],
            [min(w - 1, w - 9), base_y],
            [max(0, w - 9 - skew), near_y],
            [8, near_y],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    overlay = tile.copy()
    cv2.fillPoly(overlay, [ground_poly], (58, 90, 62), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, tile, 0.45, 0.0, tile)
    cv2.polylines(tile, [ground_poly], isClosed=True, color=(110, 170, 116), thickness=1, lineType=cv2.LINE_AA)

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

    for i, j in SKELETON_EDGES:
        if valid[i] and valid[j]:
            p1 = (int(xy[i, 0]), int(xy[i, 1]))
            p2 = (int(xy[j, 0]), int(xy[j, 1]))
            cv2.line(tile, p1, p2, (0, 210, 255), 2, cv2.LINE_AA)
    for i in idx:
        p = (int(xy[i, 0]), int(xy[i, 1]))
        cv2.circle(tile, p, 3, (80, 255, 120), -1, cv2.LINE_AA)

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
