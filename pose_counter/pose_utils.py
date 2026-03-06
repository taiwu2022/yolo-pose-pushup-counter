from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .utils.np_utils import to_numpy


@dataclass
class PoseDet:
    xy: np.ndarray       # (K, 2) pixel coords
    conf: np.ndarray     # (K,) confidence/visibility
    bbox_xyxy: Optional[np.ndarray] = None  # (4,) pixel coords


def select_largest_person(result) -> Optional[int]:
    """Return index of the person with largest bbox area."""
    if result is None or result.boxes is None:
        return None
    boxes_xyxy = to_numpy(result.boxes.xyxy)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return int(np.argmax(areas))


def extract_pose(result, person_idx: Optional[int] = None) -> Optional[PoseDet]:
    """Extract keypoints of one person from an Ultralytics pose Result."""
    if result is None or getattr(result, "keypoints", None) is None:
        return None
    kpts = result.keypoints

    # Choose which person
    if person_idx is None:
        person_idx = select_largest_person(result)
    if person_idx is None:
        return None

    # keypoints.data is typically (N, K, 3): x, y, conf/visibility (see Ultralytics examples)
    data = to_numpy(getattr(kpts, "data", None))
    if data is not None and data.ndim == 3 and data.shape[-1] >= 2:
        xy = data[person_idx, :, :2].astype(np.float32)
        if data.shape[-1] >= 3:
            conf = data[person_idx, :, 2].astype(np.float32)
        else:
            conf = np.ones((xy.shape[0],), dtype=np.float32)
    else:
        # Fallback: xy + conf (if present)
        xy = to_numpy(getattr(kpts, "xy", None))
        if xy is None:
            return None
        xy = xy[person_idx].astype(np.float32)
        conf_attr = getattr(kpts, "conf", None)
        conf = to_numpy(conf_attr)[person_idx].astype(np.float32) if conf_attr is not None else np.ones((xy.shape[0],), dtype=np.float32)

    bbox = None
    if result.boxes is not None:
        boxes_xyxy = to_numpy(result.boxes.xyxy)
        if boxes_xyxy is not None and len(boxes_xyxy) > person_idx:
            bbox = boxes_xyxy[person_idx].astype(np.float32)

    return PoseDet(xy=xy, conf=conf, bbox_xyxy=bbox)
