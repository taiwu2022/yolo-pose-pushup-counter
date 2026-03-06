from __future__ import annotations

import numpy as np


def to_numpy(x):
    """Convert torch.Tensor / list / numpy array to numpy array."""
    if x is None:
        return None
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)
