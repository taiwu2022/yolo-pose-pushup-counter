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
    down_th: float = 90.0   # when elbow angle <= down_th => "down"
    up_th: float = 160.0    # when elbow angle >= up_th => count rep and go "up"

    # keypoint confidence gate
    kpt_conf_th: float = 0.25

    # debounce
    min_interval_s: float = 0.35

    # smoothing
    ema_alpha: float = 0.25

    # require consecutive frames to confirm a transition
    hold_frames: int = 2


class PushUpCounter(BaseCounter):
    """Push-up rep counter based on elbow angle hysteresis."""

    def __init__(self, fps: float, cfg: Optional[PushUpConfig] = None):
        super().__init__(fps=fps)
        self.cfg = cfg or PushUpConfig()
        self._ema = EMA(alpha=self.cfg.ema_alpha)
        self._last_count_frame: int = -10**9
        self._hold_down: int = 0
        self._hold_up: int = 0
        self.last_angle: Optional[float] = None

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

    def update(self, kpt_xy, kpt_conf, frame_idx: int) -> None:
        xy = np.asarray(kpt_xy, dtype=np.float32)
        conf = np.asarray(kpt_conf, dtype=np.float32)

        ang = self._best_elbow_angle(xy, conf)
        if ang is None:
            # No reliable signal this frame
            self._hold_down = 0
            self._hold_up = 0
            return

        ang_s = self._ema.update(float(ang))
        self.last_angle = ang_s

        # Bootstrap phase if unknown
        if self.phase == "unknown":
            if ang_s >= self.cfg.up_th:
                self.phase = "up"
            elif ang_s <= self.cfg.down_th:
                self.phase = "down"

        # Hysteresis + hold frames
        if self.phase == "up":
            if ang_s <= self.cfg.down_th:
                self._hold_down += 1
                if self._hold_down >= self.cfg.hold_frames:
                    self.phase = "down"
                    self._hold_down = 0
            else:
                self._hold_down = 0

        elif self.phase == "down":
            if ang_s >= self.cfg.up_th:
                self._hold_up += 1
                if self._hold_up >= self.cfg.hold_frames:
                    # Debounce by time
                    min_frames = int(self.cfg.min_interval_s * self.fps)
                    if frame_idx - self._last_count_frame >= min_frames:
                        self.count += 1
                        self._last_count_frame = frame_idx
                        self.events.append(
                            RepEvent(
                                rep_index=self.count,
                                frame_idx=frame_idx,
                                time_s=frame_idx / self.fps,
                                meta={"elbow_angle": float(ang_s)},
                            )
                        )
                    self.phase = "up"
                    self._hold_up = 0
            else:
                self._hold_up = 0
