from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ..constants import KPT
from ..geometry import angle_deg
from ..utils.smoothing import EMA
from .base import BaseCounter, RepEvent


@dataclass
class PushUpConfig:
    # hysteresis thresholds (degrees)
    down_th: float = 118.0  # when elbow angle <= down_th => "down"
    up_th: float = 157.0    # when elbow angle >= up_th => count rep and go "up"

    # keypoint confidence gate
    kpt_conf_th: float = 0.25

    # debounce
    min_interval_s: float = 0.35
    min_rep_duration_s: float = 1.0

    # smoothing
    ema_alpha: float = 0.25

    # require consecutive frames to confirm a transition
    hold_frames: int = 2

    # contact-ready gate
    min_hand_contacts: int = 1
    contact_hold_frames: int = 1
    ready_hold_s: float = 2.0
    ground_contact_tol_px: float = 24.0
    hand_drop_fraction_start: float = 0.90
    ground_model_alpha: float = 0.18
    ground_model_min_points: int = 14
    head_drop_fraction_or: float = 0.45
    head_recover_ratio_or: float = 0.85
    require_head_cycle: bool = True


class PushUpCounter(BaseCounter):
    """Push-up rep counter based on elbow angle hysteresis."""

    def __init__(self, fps: float, cfg: Optional[PushUpConfig] = None):
        super().__init__(fps=fps)
        self.cfg = cfg or PushUpConfig()
        self._ema = EMA(alpha=self.cfg.ema_alpha)
        self._last_count_frame: int = -10**9
        self._down_start_frame: Optional[int] = None
        self._hold_down: int = 0
        self._hold_up: int = 0
        self._hold_contact: int = 0
        self._ready_until_frame: int = -1
        self._wrist_baseline_dist: dict[int, float] = {}
        self._ground_m: Optional[float] = None
        self._ground_b: Optional[float] = None
        self.last_angle: Optional[float] = None
        self.last_head_ground_ratio: Optional[float] = None
        self.ready_to_count: bool = False
        self._contact_head_baseline: Optional[float] = None
        self._down_reached_by_head: bool = False

    def _arm_angle(self, xy: np.ndarray, conf: np.ndarray, side: str) -> Optional[Tuple[float, float]]:
        """Return (angle, quality_conf) for left/right arm if keypoints are reliable."""
        if side == "left":
            s, e, w = KPT["left_shoulder"], KPT["left_elbow"], KPT["left_wrist"]
        else:
            s, e, w = KPT["right_shoulder"], KPT["right_elbow"], KPT["right_wrist"]

        if conf[s] < self.cfg.kpt_conf_th or conf[e] < self.cfg.kpt_conf_th or conf[w] < self.cfg.kpt_conf_th:
            return None
        ang = angle_deg(xy[s], xy[e], xy[w])
        q = float(min(conf[s], conf[e], conf[w]))
        return float(ang), q

    def _best_elbow_angle(self, xy: np.ndarray, conf: np.ndarray) -> Optional[float]:
        left = self._arm_angle(xy, conf, "left")
        right = self._arm_angle(xy, conf, "right")

        if left is None and right is None:
            return None
        if left is None:
            return right[0]
        if right is None:
            return left[0]

        # pick better-quality arm; if similar, average
        if abs(left[1] - right[1]) < 0.05:
            return float((left[0] + right[0]) / 2.0)
        return left[0] if left[1] > right[1] else right[0]

    def _update_ground_model(self, ground_mask: Optional[np.ndarray]) -> None:
        if ground_mask is None or ground_mask.ndim != 2:
            return
        h, w = ground_mask.shape
        step = max(4, w // 120)
        xs: list[float] = []
        ys: list[float] = []
        for xi in range(0, w, step):
            col = ground_mask[:, xi]
            idx = np.where(col > 0)[0]
            if idx.size == 0:
                continue
            y = int(np.min(idx))
            if y < int(h * 0.08):
                continue
            xs.append(float(xi))
            ys.append(float(y))

        if len(xs) < int(self.cfg.ground_model_min_points):
            return
        xv = np.asarray(xs, dtype=np.float32)
        yv = np.asarray(ys, dtype=np.float32)
        m, b = np.polyfit(xv, yv, deg=1)
        m = float(m)
        b = float(b)
        if self._ground_m is None or self._ground_b is None:
            self._ground_m = m
            self._ground_b = b
        else:
            a = float(self.cfg.ground_model_alpha)
            self._ground_m = (1.0 - a) * self._ground_m + a * m
            self._ground_b = (1.0 - a) * self._ground_b + a * b

    def _ground_y_at_x(self, ground_mask: Optional[np.ndarray], x: float) -> Optional[float]:
        if self._ground_m is not None and self._ground_b is not None:
            return float(self._ground_m * float(x) + self._ground_b)
        if ground_mask is None or ground_mask.ndim != 2:
            return None
        h, w = ground_mask.shape
        xi = int(np.clip(round(float(x)), 0, w - 1))
        for r in (0, 2, 4, 7, 10, 14):
            x0 = max(0, xi - r)
            x1 = min(w - 1, xi + r)
            col = ground_mask[:, x0 : x1 + 1]
            ys = np.where(col > 0)[0]
            if ys.size > 0:
                return float(np.min(ys))
        return None

    def _hand_contact_ok(self, xy: np.ndarray, conf: np.ndarray, ground_mask: Optional[np.ndarray]) -> bool:
        hits = 0
        for wid in (KPT["left_wrist"], KPT["right_wrist"]):
            if conf[wid] < self.cfg.kpt_conf_th:
                continue
            gy = self._ground_y_at_x(ground_mask, float(xy[wid, 0]))
            if gy is None:
                continue
            dist = max(0.0, float(gy - float(xy[wid, 1])))

            prev = self._wrist_baseline_dist.get(wid)
            if prev is None:
                base = dist
            else:
                target = max(dist, prev * 0.995)
                base = 0.12 * target + 0.88 * prev
            self._wrist_baseline_dist[wid] = base

            abs_near = dist <= float(self.cfg.ground_contact_tol_px)
            rel_near = False
            if base > 1e-3:
                rel_near = dist <= base * max(0.0, 1.0 - float(self.cfg.hand_drop_fraction_start))
            if abs_near or rel_near:
                hits += 1
        return hits >= int(self.cfg.min_hand_contacts)

    def _head_ground_ratio(self, xy: np.ndarray, conf: np.ndarray, ground_mask: Optional[np.ndarray]) -> Optional[float]:
        # Head reference: nose first, fallback to shoulder center.
        if conf[KPT["nose"]] >= self.cfg.kpt_conf_th:
            head_xy = xy[KPT["nose"]]
        elif conf[KPT["left_shoulder"]] >= self.cfg.kpt_conf_th and conf[KPT["right_shoulder"]] >= self.cfg.kpt_conf_th:
            head_xy = 0.5 * (xy[KPT["left_shoulder"]] + xy[KPT["right_shoulder"]])
        else:
            return None

        # Normalize by torso length for perspective robustness.
        if conf[KPT["left_shoulder"]] < self.cfg.kpt_conf_th or conf[KPT["right_shoulder"]] < self.cfg.kpt_conf_th:
            return None
        if conf[KPT["left_hip"]] < self.cfg.kpt_conf_th or conf[KPT["right_hip"]] < self.cfg.kpt_conf_th:
            return None
        sh = 0.5 * (xy[KPT["left_shoulder"]] + xy[KPT["right_shoulder"]])
        hip = 0.5 * (xy[KPT["left_hip"]] + xy[KPT["right_hip"]])
        torso = float(np.linalg.norm(sh - hip))
        if torso < 1e-3:
            return None

        gy = self._ground_y_at_x(ground_mask, float(head_xy[0]))
        if gy is None:
            return None
        d = max(0.0, float(gy - head_xy[1]))
        return float(d / torso)

    def update(self, kpt_xy, kpt_conf, frame_idx: int, **kwargs) -> None:
        xy = np.asarray(kpt_xy, dtype=np.float32)
        conf = np.asarray(kpt_conf, dtype=np.float32)
        ground_mask = kwargs.get("ground_mask")
        self._update_ground_model(ground_mask)

        contact_now = self._hand_contact_ok(xy, conf, ground_mask=ground_mask)
        if contact_now:
            self._hold_contact += 1
            if self._hold_contact >= int(self.cfg.contact_hold_frames):
                hold_frames = int(max(1.0, float(self.cfg.ready_hold_s) * self.fps))
                self._ready_until_frame = max(self._ready_until_frame, int(frame_idx) + hold_frames)
        else:
            self._hold_contact = 0
        contact_ok = int(frame_idx) <= self._ready_until_frame
        self.ready_to_count = bool(contact_ok)

        ang = self._best_elbow_angle(xy, conf)
        head_ratio = self._head_ground_ratio(xy, conf, ground_mask=ground_mask)
        if head_ratio is not None:
            self.last_head_ground_ratio = head_ratio
            if contact_ok:
                if self._contact_head_baseline is None:
                    self._contact_head_baseline = head_ratio
                else:
                    # Slow baseline update while latched in ready.
                    self._contact_head_baseline = max(self._contact_head_baseline * 0.98 + head_ratio * 0.02, head_ratio)
        elif not contact_ok:
            self._contact_head_baseline = None

        if ang is None:
            # No reliable signal this frame
            self._hold_down = 0
            self._hold_up = 0
            return

        ang_s = self._ema.update(float(ang))
        self.last_angle = ang_s

        head_down = False
        head_up = False
        if (
            contact_ok
            and self._contact_head_baseline is not None
            and self.last_head_ground_ratio is not None
            and self._contact_head_baseline > 1e-6
        ):
            down_target = self._contact_head_baseline * (1.0 - float(self.cfg.head_drop_fraction_or))
            up_target = self._contact_head_baseline * float(self.cfg.head_recover_ratio_or)
            head_down = self.last_head_ground_ratio <= down_target
            head_up = self.last_head_ground_ratio >= up_target

        # Bootstrap phase if unknown
        if self.phase == "unknown":
            if ang_s >= self.cfg.up_th:
                self.phase = "up"
            elif head_down or (ang_s <= self.cfg.down_th):
                self.phase = "down"
                self._down_start_frame = int(frame_idx)
                self._down_reached_by_head = bool(head_down)

        # Hysteresis + hold frames
        if self.phase == "up":
            if (head_down or (ang_s <= self.cfg.down_th)) and contact_ok:
                self._hold_down += 1
                if self._hold_down >= self.cfg.hold_frames:
                    self.phase = "down"
                    self._hold_down = 0
                    self._down_start_frame = int(frame_idx)
                    self._down_reached_by_head = bool(head_down)
            else:
                self._hold_down = 0

        elif self.phase == "down":
            if head_down:
                self._down_reached_by_head = True
            if head_up or (ang_s >= self.cfg.up_th):
                self._hold_up += 1
                if self._hold_up >= self.cfg.hold_frames:
                    # Debounce by time
                    min_frames = int(self.cfg.min_interval_s * self.fps)
                    min_rep_frames = int(self.cfg.min_rep_duration_s * self.fps)
                    rep_duration_ok = (
                        self._down_start_frame is not None
                        and (frame_idx - self._down_start_frame) >= min_rep_frames
                    )
                    head_cycle_ok = (not self.cfg.require_head_cycle) or (self._down_reached_by_head and head_up)
                    if frame_idx - self._last_count_frame >= min_frames and rep_duration_ok and contact_ok and head_cycle_ok:
                        self.count += 1
                        self._last_count_frame = frame_idx
                        self.events.append(
                            RepEvent(
                                rep_index=self.count,
                                frame_idx=frame_idx,
                                time_s=frame_idx / self.fps,
                                meta={
                                    "elbow_angle": float(ang_s),
                                    "head_ground_ratio": self.last_head_ground_ratio,
                                    "rep_duration_s": float((frame_idx - self._down_start_frame) / self.fps) if self._down_start_frame is not None else None,
                                },
                            )
                        )
                    self.phase = "up"
                    self._hold_up = 0
                    self._down_start_frame = None
                    self._down_reached_by_head = False
            else:
                self._hold_up = 0
