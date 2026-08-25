"""Safe handling of Tensor/NumPy values in runtime fusion and context building."""
from __future__ import annotations

from typing import Any, List, Optional


def coerce_landmarks(value: Any) -> Optional[List[List[float]]]:
    """Convert pose keypoints to a plain Python list, or None if empty/missing."""
    if value is None:
        return None

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.detach().cpu().tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return list(value)

    return None


def resolve_landmarks(primary: Any, fallback: Any) -> Optional[List[List[float]]]:
    """Pick primary keypoints when present, otherwise fallback — never uses truthiness on tensors."""
    coerced = coerce_landmarks(primary)
    if coerced is not None:
        return coerced
    return coerce_landmarks(fallback)
