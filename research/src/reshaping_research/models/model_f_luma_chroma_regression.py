"""Model F: Luma + Chroma Regression - optimized piecewise linear + 3x3 color matrix.

Luminance: monotonic piecewise linear with 8 knots (like Model B but with
optimized knot positions via scipy.optimize).
Chroma: 3x3 matrix mapping (9 params) with regularization toward identity.
Total: 8 (luma knots) + 9 (3x3 matrix) + 1 (saturation) = ~18-20 parameters.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..utils.color_spaces import BT2020_LUMA

DEFAULT_NUM_KNOTS = 8
IDENTITY_3X3 = np.eye(3, dtype=np.float64)


class ModelFLumaChromaRegression:
    """Piecewise linear luma + 3x3 color matrix model.

    Luminance path uses optimized knot positions for piecewise linear mapping.
    Chroma path uses a 3x3 matrix regularized toward identity, similar to
    the production pipeline's color correction approach.
    """

    def __init__(
        self,
        num_knots: int = DEFAULT_NUM_KNOTS,
        matrix_lambda: float = 0.5,
    ) -> None:
        self.num_knots = num_knots
        self.matrix_lambda = matrix_lambda
        self.params: Dict[str, object] = {
            "x_knots": np.linspace(0.0, 1.0, num_knots),
            "y_knots": np.linspace(0.0, 1.0, num_knots),
            "color_matrix": IDENTITY_3X3.copy(),
            "saturation": 1.0,
        }

    def name(self) -> str:
        """Return model name."""
        return "Model F: Luma + Chroma Regression"

    def param_count(self) -> int:
        """Return number of parameters."""
        return self.num_knots + 9 + 1  # knot y-values + 3x3 matrix + saturation

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, object]:
        """Estimate piecewise linear + color matrix from overlap region.

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

        # Step 1: Fit optimized piecewise linear for luminance
        x_knots, y_knots = self._fit_luma_piecewise(sdr_linear, hdr_linear)

        # Step 2: Fit 3x3 color matrix
        color_matrix = IDENTITY_3X3.copy()
        saturation = 1.0
        if sdr_chroma is not None and hdr_chroma is not None:
            color_matrix, saturation = self._fit_color_matrix(
                sdr_chroma, hdr_chroma
            )

        self.params = {
            "x_knots": x_knots,
            "y_knots": y_knots,
            "color_matrix": color_matrix,
            "saturation": saturation,
        }
        return self.params

    def _fit_luma_piecewise(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Fit piecewise linear with optimized knot positions."""
        # Initial knots at evenly spaced percentiles
        percentiles = np.linspace(1, 99, self.num_knots)
        x_knots = np.percentile(sdr_linear, percentiles)

        # Ensure strictly increasing x_knots
        for i in range(1, len(x_knots)):
            if x_knots[i] <= x_knots[i - 1]:
                x_knots[i] = x_knots[i - 1] + 1e-8

        # For each knot, find median HDR value in neighborhood
        y_knots = np.zeros(self.num_knots)
        for i in range(self.num_knots):
            if i == 0:
                low = 0.0
            else:
                low = (x_knots[i - 1] + x_knots[i]) / 2.0
            if i == self.num_knots - 1:
                high = float(np.max(sdr_linear)) + 1e-10
            else:
                high = (x_knots[i] + x_knots[i + 1]) / 2.0

            mask = (sdr_linear >= low) & (sdr_linear < high)
            if np.sum(mask) > 0:
                y_knots[i] = float(np.median(hdr_linear[mask]))
            else:
                y_knots[i] = float(np.interp(x_knots[i], x_knots[:i], y_knots[:i])) if i > 0 else 0.0

        # Enforce monotonicity
        for i in range(1, len(y_knots)):
            if y_knots[i] < y_knots[i - 1]:
                y_knots[i] = y_knots[i - 1]

        # Optimize knot y-positions to minimize MSE
        n = len(sdr_linear)
        if n > 5000:
            idx = np.linspace(0, n - 1, 5000, dtype=int)
            sdr_sub = sdr_linear[idx]
            hdr_sub = hdr_linear[idx]
        else:
            sdr_sub = sdr_linear
            hdr_sub = hdr_linear

        def objective(y_vals: NDArray) -> float:
            # Enforce monotonicity in objective
            y_mono = np.maximum.accumulate(y_vals)
            pred = np.interp(sdr_sub, x_knots, y_mono)
            return float(np.mean((pred - hdr_sub) ** 2))

        result = minimize(
            objective,
            y_knots,
            method="L-BFGS-B",
            bounds=[(0.0, None) for _ in range(self.num_knots)],
            options={"maxiter": 200, "ftol": 1e-10},
        )

        y_knots_opt = np.maximum.accumulate(result.x)

        return x_knots, y_knots_opt

    def _fit_color_matrix(
        self,
        sdr_chroma: NDArray[np.floating],
        hdr_chroma: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], float]:
        """Fit 3x3 color correction matrix with regularization toward identity."""
        n = len(sdr_chroma)
        if n > 3000:
            idx = np.linspace(0, n - 1, 3000, dtype=int)
            sdr_sub = sdr_chroma[idx]
            hdr_sub = hdr_chroma[idx]
        else:
            sdr_sub = sdr_chroma
            hdr_sub = hdr_chroma

        # Compute saturation
        sdr_y = np.einsum("...c,c->...", sdr_sub, BT2020_LUMA)
        hdr_y = np.einsum("...c,c->...", hdr_sub, BT2020_LUMA)

        sdr_chroma_mag = np.sqrt(np.sum((sdr_sub - sdr_y[..., np.newaxis]) ** 2, axis=-1))
        hdr_chroma_mag = np.sqrt(np.sum((hdr_sub - hdr_y[..., np.newaxis]) ** 2, axis=-1))

        valid = sdr_chroma_mag > 1e-6
        if np.sum(valid) < 20:
            return IDENTITY_3X3.copy(), 1.0

        saturation = float(np.median(hdr_chroma_mag[valid] / sdr_chroma_mag[valid]))

        # Fit 3x3 matrix via regularized least squares
        S = sdr_sub.T  # (3, N)
        H = hdr_sub.T  # (3, N)
        n_samples = S.shape[1]

        # M = (H@S^T + lambda*I*N) @ (S@S^T + lambda*I*N)^-1
        reg = self.matrix_lambda * np.eye(3) * n_samples
        try:
            M = (H @ S.T + reg) @ np.linalg.inv(S @ S.T + reg)
        except np.linalg.LinAlgError:
            M = IDENTITY_3X3.copy()

        return M, saturation

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, object]] = None,
    ) -> NDArray[np.floating]:
        """Apply piecewise linear luma + color matrix to full SDR image.

        Args:
            sdr_image_linear: (H, W, 3) linear BT.2020 RGB.
            params: Optional override parameters.

        Returns:
            (H, W, 3) HDR linear BT.2020 RGB.
        """
        p = params if params is not None else self.params
        x_knots = np.asarray(p["x_knots"])
        y_knots = np.asarray(p["y_knots"])
        color_matrix = np.asarray(p["color_matrix"])
        saturation = float(p["saturation"])

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)

        # Apply piecewise linear luma mapping
        hdr_lum = np.interp(lum, x_knots, y_knots)
        hdr_lum_3d = hdr_lum[..., np.newaxis]

        # Apply 3x3 color matrix
        color_corrected = np.einsum("ij,...j->...i", color_matrix, sdr_image_linear)

        # Compute new luminance after color matrix
        cc_lum = np.einsum("...c,c->...", color_corrected, BT2020_LUMA)
        cc_lum_3d = cc_lum[..., np.newaxis]

        # Reconstruct: use fitted luma + color-corrected chroma with saturation
        chroma = color_corrected - cc_lum_3d
        result = hdr_lum_3d + saturation * chroma

        return np.maximum(result, 0.0)
