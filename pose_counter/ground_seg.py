from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class GroundSegConfig:
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    stride: int = 2
    min_conf_th: float = 0.0


class GroundSegEstimator:
    """Ground mask estimator using semantic segmentation (ADE20K-like labels)."""

    def __init__(self, cfg: Optional[GroundSegConfig] = None):
        self.cfg = cfg or GroundSegConfig()
        self._frame_idx = 0
        self._last_mask: Optional[np.ndarray] = None

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except Exception as e:
            raise RuntimeError(
                "Ground segmentation requires 'transformers' and 'torch'. "
                "Install with: pip install transformers"
            ) from e

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self.cfg.model_name)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(self.cfg.model_name)
        self._model.eval()

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._model.to(self._device)

        id2label = getattr(self._model.config, "id2label", {})
        keywords = ("floor", "ground", "earth", "road", "grass", "field", "sand", "pavement")
        self._ground_ids = [
            int(i)
            for i, name in id2label.items()
            if any(k in str(name).lower() for k in keywords)
        ]
        if not self._ground_ids:
            # Fallback for ADE20K where class 3 is typically "floor".
            self._ground_ids = [3]

    def _keep_bottom_connected(self, mask: np.ndarray) -> np.ndarray:
        """Keep only components connected to the bottom edge to suppress false positives."""
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        n, labels, _stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask

        h, w = mask.shape
        bottom_labels = set(int(x) for x in labels[h - 1, :] if x > 0)
        if not bottom_labels:
            return np.zeros((h, w), dtype=np.uint8)
        out = np.zeros((h, w), dtype=np.uint8)
        for lbl in bottom_labels:
            out[labels == lbl] = 1
        return out

    def _predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model(**inputs)
            logits = out.logits
            up = self._torch.nn.functional.interpolate(
                logits,
                size=frame_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            pred = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        mask = np.isin(pred, self._ground_ids).astype(np.uint8)
        # Smooth ragged edges.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = self._keep_bottom_connected(mask)
        return mask

    def update(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        self._frame_idx += 1
        stride = max(1, int(self.cfg.stride))
        if self._last_mask is not None and (self._frame_idx % stride) != 0:
            return self._last_mask

        try:
            self._last_mask = self._predict(frame_bgr)
            return self._last_mask
        except Exception:
            return self._last_mask
