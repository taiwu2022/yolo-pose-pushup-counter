from __future__ import annotations

import cv2
import numpy as np

from .constants import SKELETON_EDGES


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

    x, y = 20, 30
    for s in lines:
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        y += 28
