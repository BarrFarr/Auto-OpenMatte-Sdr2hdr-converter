"""Temporal metrics: parameter variance, flicker risk.

Evaluates temporal stability of reshaping parameters across frames/scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class TemporalStability:
    """Temporal stability measurements for a parameter sequence."""

    mean: float
    std: float
    coefficient_of_variation: float
    max_frame_delta: float
    flicker_risk: float  # 0-1, higher = more flicker likely


def parameter_variance(
    parameters: Sequence[float],
) -> float:
    """Coefficient of variation for a sequence of reshaping parameters.

    Given multiple parameter values (e.g., from the same scene with noise,
    or across consecutive frames), compute the coefficient of variation
    (std / mean) as a stability metric.

    Args:
        parameters: Sequence of scalar parameter values.

    Returns:
        Coefficient of variation (0 = perfectly stable).
    """
    arr = np.asarray(parameters, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    mean = np.mean(arr)
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


def flicker_risk(
    parameters: Sequence[float],
    threshold: float = 0.05,
) -> TemporalStability:
    """Assess flicker risk from parameter variation.

    Computes temporal stability metrics and a flicker risk score based
    on frame-to-frame parameter changes.

    Args:
        parameters: Sequence of parameter values (one per frame/scene).
        threshold: Maximum acceptable frame-to-frame change as fraction
            of mean. Changes above this contribute to flicker risk.

    Returns:
        TemporalStability dataclass with all stability measurements.
    """
    arr = np.asarray(parameters, dtype=np.float64)

    if len(arr) < 2:
        return TemporalStability(
            mean=float(arr[0]) if len(arr) == 1 else 0.0,
            std=0.0,
            coefficient_of_variation=0.0,
            max_frame_delta=0.0,
            flicker_risk=0.0,
        )

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else 0.0

    # Frame-to-frame deltas
    deltas = np.abs(np.diff(arr))
    max_delta = float(np.max(deltas))

    # Flicker risk: fraction of frame transitions exceeding threshold
    if abs(mean_val) > 1e-12:
        relative_deltas = deltas / abs(mean_val)
        risk = float(np.mean(relative_deltas > threshold))
    else:
        risk = 0.0

    return TemporalStability(
        mean=mean_val,
        std=std_val,
        coefficient_of_variation=cv,
        max_frame_delta=max_delta,
        flicker_risk=risk,
    )
