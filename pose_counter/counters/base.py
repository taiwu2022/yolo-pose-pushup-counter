from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import csv
from pathlib import Path


@dataclass
class RepEvent:
    rep_index: int
    frame_idx: int
    time_s: float
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseCounter:
    """Base class for repetition counters."""

    def __init__(self, fps: float):
        self.fps = float(fps)
        self.count: int = 0
        self.phase: str = "unknown"
        self.events: List[RepEvent] = []

    def update(self, kpt_xy, kpt_conf, frame_idx: int, **kwargs) -> None:
        raise NotImplementedError

    def save_csv(self, csv_path: str | Path) -> None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rep_index", "frame_idx", "time_s", "meta_json"])
            import json
            for ev in self.events:
                w.writerow([ev.rep_index, ev.frame_idx, f"{ev.time_s:.6f}", json.dumps(ev.meta, ensure_ascii=False)])
