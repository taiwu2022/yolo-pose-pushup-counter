from __future__ import annotations

import math
import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees given 2D points a,b,c (shape (2,))."""
    ba = a - b
    bc = c - b
    # Prevent division by zero
    nba = float(np.linalg.norm(ba) + 1e-9)
    nbc = float(np.linalg.norm(bc) + 1e-9)
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))
