"""Model A: Linear Gain - simplest possible reshaping model.

Y' = a * Y  (luminance gain)
C' = b * C  (chroma gain, applied as ratio scaling of RGB around Y)

Parameters: 2 (a = luminance gain, b = chroma gain)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import BT2020_LUMA


class ModelALinear:
    """Linear gain model for SDR-to-HDR reshaping.

    Applies a constant luminance gain and a constant chroma gain.
    Simplest possible model - serves as baseline.
    """

    def __init__(self) -> None:
        self.params: Dict[str, float] = {"a": 1.0, "b": 1.0}

    def name(self) -> str:
        """Return model name."""
        return "Model A: Linear Gain"

    def param_count(self) -> int:
        """Return number of parameters."""
        return 2

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, float]:
        """Estimate gain parameters from overlap region.

        Args:
            sdr_linear: Flattened SDR luminance (linear BT.2020 Y), shape (N,).
            hdr_linear: Corresponding HDR luminance (linear, normalized [0,1]), shape (N,).
            sdr_chroma: Optional SDR RGB pixels (N, 3) in linear BT.2020.
            hdr_chroma: Optional HDR RGB pixels (N, 3) in linear BT.2020.

        Returns:
            Dictionary of fitted parameters.
        """
        # Filter valid pixels (avoid division by zero)
        valid = sdr_linear > 1e-6
        if np.sum(valid) < 10:
            self.params = {"a": 1.0, "b": 1.0}
            return self.params

        # Luminance gain: median of ratio
        ratios = hdr_linear[valid] / sdr_linear[valid]
        a = float(np.median(ratios))

        # Chroma gain
        b = 1.0
        if sdr_chroma is not None and hdr_chroma is not None:
            sdr_y = np.einsum("...c,c->...", sdr_chroma, BT2020_LUMA)
            hdr_y = np.einsum("...c,c->...", hdr_chroma, BT2020_LUMA)

            # Chroma magnitude = distance from achromatic axis
            sdr_y_3d = sdr_y[..., np.newaxis]
            hdr_y_3d = hdr_y[..., np.newaxis]

            sdr_chroma_vec = sdr_chroma - sdr_y_3d
            hdr_chroma_vec = hdr_chroma - hdr_y_3d

            sdr_chroma_mag = np.sqrt(np.sum(sdr_chroma_vec**2, axis=-1))
            hdr_chroma_mag = np.sqrt(np.sum(hdr_chroma_vec**2, axis=-1))

            chroma_valid = sdr_chroma_mag > 1e-6
            if np.sum(chroma_valid) > 10:
                chroma_ratios = hdr_chroma_mag[chroma_valid] / sdr_chroma_mag[chroma_valid]
                b = float(np.median(chroma_ratios))

        self.params = {"a": a, "b": b}
        return self.params

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, float]] = None,
    ) -> NDArray[np.floating]:
        """Apply linear gain transform to full SDR image.

        Args:
            sdr_image_linear: (H, W, 3) linear BT.2020 RGB.
            params: Optional override parameters. Uses fitted params if None.

        Returns:
            (H, W, 3) HDR linear BT.2020 RGB.
        """
        p = params if params is not None else self.params
        a = p["a"]
        b = p["b"]

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)
        lum_3d = lum[..., np.newaxis]

        # Apply luminance gain
        hdr_lum = a * lum_3d

        # Apply chroma gain: scale deviation from luminance
        chroma = sdr_image_linear - lum_3d
        hdr_chroma = b * chroma

        # Reconstruct
        result = hdr_lum + hdr_chroma

        return np.maximum(result, 0.0)
