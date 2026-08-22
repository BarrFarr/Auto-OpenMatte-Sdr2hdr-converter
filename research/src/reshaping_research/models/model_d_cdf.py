"""Model D: CDF Matching - empirical cumulative distribution function mapping.

Compute empirical CDFs of SDR and HDR luminance.
Mapping: for each SDR value x, find SDR_CDF(x), then HDR_CDF_inv(SDR_CDF(x)).
Store as 64-point discretized lookup table.
Chroma: simple ratio (1 param).

Effective storage: 64 + 1 = 65 params (nonparametric approach).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import BT2020_LUMA

DEFAULT_LUT_SIZE = 64


class ModelDCdf:
    """CDF matching model for SDR-to-HDR reshaping.

    Maps SDR luminance to HDR using histogram equalization principles:
    HDR_value = HDR_CDF_inv(SDR_CDF(SDR_value))
    """

    def __init__(self, lut_size: int = DEFAULT_LUT_SIZE) -> None:
        self.lut_size = lut_size
        # Initialize identity LUT
        self.params: Dict[str, object] = {
            "lut_x": np.linspace(0.0, 1.0, lut_size),
            "lut_y": np.linspace(0.0, 1.0, lut_size),
            "chroma_gain": 1.0,
        }

    def name(self) -> str:
        """Return model name."""
        return "Model D: CDF Matching"

    def param_count(self) -> int:
        """Return number of parameters."""
        return self.lut_size + 1  # LUT values + chroma gain

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, object]:
        """Estimate CDF-based mapping from overlap region.

        Args:
            sdr_linear: Flattened SDR luminance (linear BT.2020 Y), shape (N,).
            hdr_linear: Corresponding HDR luminance (linear, normalized [0,1]), shape (N,).
            sdr_chroma: Optional SDR RGB pixels (N, 3) in linear BT.2020.
            hdr_chroma: Optional HDR RGB pixels (N, 3) in linear BT.2020.

        Returns:
            Dictionary of fitted parameters.
        """
        if len(sdr_linear) < 20:
            return self.params

        # Build SDR CDF
        sdr_sorted = np.sort(sdr_linear)
        sdr_cdf = np.linspace(0.0, 1.0, len(sdr_sorted))

        # Build HDR CDF
        hdr_sorted = np.sort(hdr_linear)
        hdr_cdf = np.linspace(0.0, 1.0, len(hdr_sorted))

        # Create LUT: for uniform x-grid, find the mapping
        sdr_max = float(np.max(sdr_linear))
        if sdr_max < 1e-10:
            sdr_max = 1.0
        lut_x = np.linspace(0.0, sdr_max, self.lut_size)

        # For each x value, find SDR_CDF(x)
        sdr_cdf_at_x = np.interp(lut_x, sdr_sorted, sdr_cdf)

        # Then find HDR_CDF_inv(SDR_CDF(x))
        lut_y = np.interp(sdr_cdf_at_x, hdr_cdf, hdr_sorted)

        # Ensure monotonicity
        for i in range(1, len(lut_y)):
            if lut_y[i] < lut_y[i - 1]:
                lut_y[i] = lut_y[i - 1]

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
            "lut_x": lut_x,
            "lut_y": lut_y,
            "chroma_gain": chroma_gain,
        }
        return self.params

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, object]] = None,
    ) -> NDArray[np.floating]:
        """Apply CDF-matched LUT to full SDR image.

        Args:
            sdr_image_linear: (H, W, 3) linear BT.2020 RGB.
            params: Optional override parameters.

        Returns:
            (H, W, 3) HDR linear BT.2020 RGB.
        """
        p = params if params is not None else self.params
        lut_x = np.asarray(p["lut_x"])
        lut_y = np.asarray(p["lut_y"])
        chroma_gain = float(p["chroma_gain"])

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)
        lum_3d = lum[..., np.newaxis]

        # Apply LUT mapping to luminance
        hdr_lum = np.interp(lum, lut_x, lut_y)
        hdr_lum_3d = hdr_lum[..., np.newaxis]

        # Apply chroma gain
        chroma = sdr_image_linear - lum_3d
        hdr_chroma = chroma_gain * chroma

        # Reconstruct
        result = hdr_lum_3d + hdr_chroma

        return np.maximum(result, 0.0)
