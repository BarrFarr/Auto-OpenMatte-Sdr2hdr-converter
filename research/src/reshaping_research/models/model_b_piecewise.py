"""Model B: Piecewise Linear - percentile-matched control points.

Luminance: piecewise linear function with N control points at fixed percentiles.
At each percentile P of SDR, the mapping target = median of corresponding HDR values.
Interpolate linearly between control points.
Chroma: simple linear gain (1 param).

Parameters: ~9-17 (N y-values at fixed x-positions + 1 chroma gain)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import BT2020_LUMA

# Default percentile positions for control points
DEFAULT_PERCENTILES = (1, 5, 15, 30, 50, 70, 85, 95, 99)


class ModelBPiecewise:
    """Piecewise linear model with percentile-matched control points.

    Uses fixed x-positions (SDR percentile values) with fitted y-positions
    (median HDR values at each percentile bin).
    """

    def __init__(self, percentiles: tuple = DEFAULT_PERCENTILES) -> None:
        self.percentiles = percentiles
        self.params: Dict[str, object] = {
            "x_knots": np.linspace(0.0, 1.0, len(percentiles)),
            "y_knots": np.linspace(0.0, 1.0, len(percentiles)),
            "chroma_gain": 1.0,
        }

    def name(self) -> str:
        """Return model name."""
        return "Model B: Piecewise Linear"

    def param_count(self) -> int:
        """Return number of parameters."""
        return len(self.percentiles) + 1  # N y-values + 1 chroma gain

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, object]:
        """Estimate piecewise linear mapping from overlap region.

        Args:
            sdr_linear: Flattened SDR luminance (linear BT.2020 Y), shape (N,).
            hdr_linear: Corresponding HDR luminance (linear, normalized [0,1]), shape (N,).
            sdr_chroma: Optional SDR RGB pixels (N, 3) in linear BT.2020.
            hdr_chroma: Optional HDR RGB pixels (N, 3) in linear BT.2020.

        Returns:
            Dictionary of fitted parameters.
        """
        if len(sdr_linear) < 10:
            return self.params

        # Compute x-knots as percentile values of SDR
        x_knots = np.percentile(sdr_linear, list(self.percentiles))

        # For each percentile bin, compute median of corresponding HDR values
        y_knots = np.zeros_like(x_knots)
        for i in range(len(x_knots)):
            if i == 0:
                low = 0.0
            else:
                low = (x_knots[i - 1] + x_knots[i]) / 2.0
            if i == len(x_knots) - 1:
                high = np.max(sdr_linear) + 1e-10
            else:
                high = (x_knots[i] + x_knots[i + 1]) / 2.0

            mask = (sdr_linear >= low) & (sdr_linear < high)
            if np.sum(mask) > 0:
                y_knots[i] = float(np.median(hdr_linear[mask]))
            else:
                # Fallback: linear interpolation
                y_knots[i] = x_knots[i]

        # Ensure monotonicity: enforce non-decreasing y-knots
        for i in range(1, len(y_knots)):
            if y_knots[i] < y_knots[i - 1]:
                y_knots[i] = y_knots[i - 1]

        # Chroma gain
        chroma_gain = 1.0
        if sdr_chroma is not None and hdr_chroma is not None:
            sdr_y = np.einsum("...c,c->...", sdr_chroma, BT2020_LUMA)
            hdr_y = np.einsum("...c,c->...", hdr_chroma, BT2020_LUMA)
            sdr_chroma_mag = np.sqrt(
                np.sum((sdr_chroma - sdr_y[..., np.newaxis]) ** 2, axis=-1)
            )
            hdr_chroma_mag = np.sqrt(
                np.sum((hdr_chroma - hdr_y[..., np.newaxis]) ** 2, axis=-1)
            )
            valid = sdr_chroma_mag > 1e-6
            if np.sum(valid) > 10:
                chroma_gain = float(
                    np.median(hdr_chroma_mag[valid] / sdr_chroma_mag[valid])
                )

        self.params = {
            "x_knots": x_knots,
            "y_knots": y_knots,
            "chroma_gain": chroma_gain,
        }
        return self.params

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, object]] = None,
    ) -> NDArray[np.floating]:
        """Apply piecewise linear transform to full SDR image.

        Args:
            sdr_image_linear: (H, W, 3) linear BT.2020 RGB.
            params: Optional override parameters.

        Returns:
            (H, W, 3) HDR linear BT.2020 RGB.
        """
        p = params if params is not None else self.params
        x_knots = np.asarray(p["x_knots"])
        y_knots = np.asarray(p["y_knots"])
        chroma_gain = float(p["chroma_gain"])

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)
        lum_3d = lum[..., np.newaxis]

        # Apply piecewise linear mapping to luminance
        hdr_lum = np.interp(lum, x_knots, y_knots)
        hdr_lum_3d = hdr_lum[..., np.newaxis]

        # Apply chroma gain
        chroma = sdr_image_linear - lum_3d
        hdr_chroma = chroma_gain * chroma

        # Reconstruct
        result = hdr_lum_3d + hdr_chroma

        return np.maximum(result, 0.0)
