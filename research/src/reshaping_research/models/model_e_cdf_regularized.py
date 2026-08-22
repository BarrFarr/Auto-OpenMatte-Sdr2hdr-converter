"""Model E: CDF + Regularization - CDF matching with parametric correction.

Start with CDF mapping from Model D as initialization.
Parameterize as: CDF_base + correction_function.
Correction is a low-degree polynomial (degree 3-4) or a few sigmoid bumps.
Optimize via scipy.optimize to minimize: reconstruction_error + lambda * smoothness_penalty.

Parameters: 10-15 (4-5 for correction polynomial + 4-5 sigmoid params + 1-2 chroma)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..utils.color_spaces import BT2020_LUMA

DEFAULT_LUT_SIZE = 64
DEFAULT_CORRECTION_DEGREE = 3
DEFAULT_NUM_SIGMOIDS = 2


class ModelECdfRegularized:
    """CDF matching with parametric regularization.

    Combines the flexibility of CDF matching with a smooth parametric
    correction function, optimized to balance accuracy and smoothness.
    """

    def __init__(
        self,
        lut_size: int = DEFAULT_LUT_SIZE,
        correction_degree: int = DEFAULT_CORRECTION_DEGREE,
        num_sigmoids: int = DEFAULT_NUM_SIGMOIDS,
        smoothness_lambda: float = 0.1,
    ) -> None:
        self.lut_size = lut_size
        self.correction_degree = correction_degree
        self.num_sigmoids = num_sigmoids
        self.smoothness_lambda = smoothness_lambda
        # Parameters: correction poly + sigmoid params + chroma
        n_correction = correction_degree + 1
        n_sigmoid = num_sigmoids * 3  # amplitude, center, width per sigmoid
        self.params: Dict[str, object] = {
            "lut_x": np.linspace(0.0, 1.0, lut_size),
            "lut_y": np.linspace(0.0, 1.0, lut_size),
            "correction_coeffs": np.zeros(n_correction),
            "sigmoid_params": np.zeros(n_sigmoid),
            "chroma_gain": 1.0,
            "chroma_slope": 0.0,
        }

    def name(self) -> str:
        """Return model name."""
        return "Model E: CDF + Regularization"

    def param_count(self) -> int:
        """Return number of parameters."""
        # correction_poly + sigmoids + chroma (gain + slope)
        return (self.correction_degree + 1) + (self.num_sigmoids * 3) + 2

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, object]:
        """Estimate CDF + correction from overlap region.

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

        # Step 1: Build base CDF mapping (same as Model D)
        sdr_sorted = np.sort(sdr_linear)
        sdr_cdf = np.linspace(0.0, 1.0, len(sdr_sorted))
        hdr_sorted = np.sort(hdr_linear)
        hdr_cdf = np.linspace(0.0, 1.0, len(hdr_sorted))

        sdr_max = float(np.max(sdr_linear))
        if sdr_max < 1e-10:
            sdr_max = 1.0
        lut_x = np.linspace(0.0, sdr_max, self.lut_size)
        sdr_cdf_at_x = np.interp(lut_x, sdr_sorted, sdr_cdf)
        lut_y_base = np.interp(sdr_cdf_at_x, hdr_cdf, hdr_sorted)

        # Ensure monotonicity in base
        for i in range(1, len(lut_y_base)):
            if lut_y_base[i] < lut_y_base[i - 1]:
                lut_y_base[i] = lut_y_base[i - 1]

        # Step 2: Optimize correction function
        # Subsample for efficiency
        n = len(sdr_linear)
        if n > 3000:
            idx = np.linspace(0, n - 1, 3000, dtype=int)
            sdr_sub = sdr_linear[idx]
            hdr_sub = hdr_linear[idx]
        else:
            sdr_sub = sdr_linear
            hdr_sub = hdr_linear

        # Get base prediction for subsample
        base_pred = np.interp(sdr_sub, lut_x, lut_y_base)
        residual = hdr_sub - base_pred

        # Normalize SDR to [0,1] for correction fitting
        sdr_norm = sdr_sub / sdr_max if sdr_max > 1e-10 else sdr_sub

        # Fit correction to residual
        correction_coeffs, sigmoid_params = self._fit_correction(
            sdr_norm, residual
        )

        # Step 3: Chroma parameters
        chroma_gain = 1.0
        chroma_slope = 0.0
        if sdr_chroma is not None and hdr_chroma is not None:
            chroma_gain, chroma_slope = self._fit_chroma(
                sdr_chroma, hdr_chroma
            )

        # Build final LUT (base + correction)
        lut_x_norm = lut_x / sdr_max if sdr_max > 1e-10 else lut_x
        correction = self._eval_correction(lut_x_norm, correction_coeffs, sigmoid_params)
        lut_y = lut_y_base + correction

        # Ensure monotonicity in final LUT
        for i in range(1, len(lut_y)):
            if lut_y[i] < lut_y[i - 1]:
                lut_y[i] = lut_y[i - 1]

        self.params = {
            "lut_x": lut_x,
            "lut_y": lut_y,
            "correction_coeffs": correction_coeffs,
            "sigmoid_params": sigmoid_params,
            "chroma_gain": chroma_gain,
            "chroma_slope": chroma_slope,
        }
        return self.params

    def _fit_correction(
        self,
        x_norm: NDArray[np.floating],
        residual: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Fit correction function (polynomial + sigmoids) to residual."""
        n_poly = self.correction_degree + 1
        n_sig = self.num_sigmoids * 3

        # Initial: zero correction
        init = np.zeros(n_poly + n_sig)
        # Initialize sigmoid centers spread across [0,1]
        for i in range(self.num_sigmoids):
            init[n_poly + i * 3 + 1] = (i + 1) / (self.num_sigmoids + 1)  # center
            init[n_poly + i * 3 + 2] = 0.1  # width

        num_sigmoids = self.num_sigmoids
        smoothness_lambda = self.smoothness_lambda

        def objective(params: NDArray) -> float:
            poly_coeffs = params[:n_poly]
            sig_params = params[n_poly:]
            pred = self._eval_correction(x_norm, poly_coeffs, sig_params)
            mse = float(np.mean((pred - residual) ** 2))
            # Smoothness penalty
            smoothness = float(np.sum(poly_coeffs**2))
            sig_amps = sig_params[:num_sigmoids * 3:3] if len(sig_params) > 0 else np.array([])
            smoothness += float(np.sum(sig_amps**2))
            return mse + smoothness_lambda * smoothness

        result = minimize(
            objective,
            init,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-10},
        )

        best = result.x
        correction_coeffs = best[:n_poly]
        sigmoid_params = best[n_poly:]

        return correction_coeffs, sigmoid_params

    def _fit_chroma(
        self,
        sdr_chroma: NDArray[np.floating],
        hdr_chroma: NDArray[np.floating],
    ) -> Tuple[float, float]:
        """Fit luminance-dependent chroma scaling: gain + slope * Y."""
        sdr_y = np.einsum("...c,c->...", sdr_chroma, BT2020_LUMA)
        hdr_y = np.einsum("...c,c->...", hdr_chroma, BT2020_LUMA)

        sdr_chroma_mag = np.sqrt(
            np.sum((sdr_chroma - sdr_y[..., np.newaxis]) ** 2, axis=-1)
        )
        hdr_chroma_mag = np.sqrt(
            np.sum((hdr_chroma - hdr_y[..., np.newaxis]) ** 2, axis=-1)
        )

        valid = sdr_chroma_mag > 1e-6
        if np.sum(valid) < 20:
            return 1.0, 0.0

        ratio = hdr_chroma_mag[valid] / sdr_chroma_mag[valid]
        lum_vals = sdr_y[valid]

        # Clip outliers
        ratio = np.clip(ratio, 0.1, 10.0)

        # Fit linear: ratio = gain + slope * lum
        try:
            coeffs = np.polyfit(lum_vals, ratio, 1)
            return float(coeffs[1]), float(coeffs[0])  # intercept, slope
        except (np.linalg.LinAlgError, ValueError):
            return float(np.median(ratio)), 0.0

    def _eval_correction(
        self,
        x: NDArray[np.floating],
        poly_coeffs: NDArray[np.floating],
        sigmoid_params: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate correction function = polynomial + sigmoids."""
        # Polynomial part
        result = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(poly_coeffs):
            result = result + c * np.power(x, i)

        # Sigmoid bumps (Gaussian bumps)
        for i in range(self.num_sigmoids):
            amp = sigmoid_params[i * 3]
            center = sigmoid_params[i * 3 + 1]
            width = max(abs(sigmoid_params[i * 3 + 2]), 0.01)
            result = result + amp * np.exp(-((x - center) ** 2) / (2.0 * width**2))

        return result

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, object]] = None,
    ) -> NDArray[np.floating]:
        """Apply CDF + correction transform to full SDR image.

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
        chroma_slope = float(p["chroma_slope"])

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)
        lum_3d = lum[..., np.newaxis]

        # Apply LUT mapping to luminance
        hdr_lum = np.interp(lum, lut_x, lut_y)
        hdr_lum_3d = hdr_lum[..., np.newaxis]

        # Luminance-dependent chroma scaling
        chroma_scale = chroma_gain + chroma_slope * lum
        chroma_scale = np.clip(chroma_scale, 0.1, 10.0)
        chroma_scale_3d = chroma_scale[..., np.newaxis]

        # Apply chroma transform
        chroma = sdr_image_linear - lum_3d
        hdr_chroma = chroma_scale_3d * chroma

        # Reconstruct
        result = hdr_lum_3d + hdr_chroma

        return np.maximum(result, 0.0)
