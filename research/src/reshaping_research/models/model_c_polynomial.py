"""Model C: Monotonic Polynomial - polynomial with monotonicity constraint.

Luminance: polynomial of degree 3-5 with monotonicity constraint.
Fit via scipy.optimize.minimize with constraints (derivative >= 0).
Chroma: degree-2 polynomial for saturation scaling as function of luminance.

Parameters: 8-12 (degree+1 for luma polynomial + 3 for chroma polynomial)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..utils.color_spaces import BT2020_LUMA

DEFAULT_LUMA_DEGREE = 4
DEFAULT_CHROMA_DEGREE = 2


class ModelCPolynomial:
    """Monotonic polynomial model for SDR-to-HDR reshaping.

    Fits a polynomial of configurable degree with the constraint
    that the derivative is non-negative (monotonically non-decreasing).
    """

    def __init__(
        self, luma_degree: int = DEFAULT_LUMA_DEGREE, chroma_degree: int = DEFAULT_CHROMA_DEGREE
    ) -> None:
        self.luma_degree = luma_degree
        self.chroma_degree = chroma_degree
        # Initialize with identity-like coefficients
        luma_init = np.zeros(luma_degree + 1)
        luma_init[1] = 1.0
        self.params: Dict[str, object] = {
            "luma_coeffs": luma_init,
            "chroma_coeffs": np.array([1.0] + [0.0] * chroma_degree),
        }

    def name(self) -> str:
        """Return model name."""
        return "Model C: Monotonic Polynomial"

    def param_count(self) -> int:
        """Return number of parameters."""
        return (self.luma_degree + 1) + (self.chroma_degree + 1)

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, object]:
        """Estimate monotonic polynomial from overlap region.

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

        # Subsample for efficiency
        n = len(sdr_linear)
        if n > 5000:
            idx = np.linspace(0, n - 1, 5000, dtype=int)
            sdr_sub = sdr_linear[idx]
            hdr_sub = hdr_linear[idx]
        else:
            sdr_sub = sdr_linear
            hdr_sub = hdr_linear

        # Fit monotonic polynomial for luminance
        luma_coeffs = self._fit_monotonic_poly(sdr_sub, hdr_sub, self.luma_degree)

        # Fit chroma polynomial
        chroma_coeffs = self._fit_chroma_poly(
            sdr_linear, hdr_linear, sdr_chroma, hdr_chroma
        )

        self.params = {
            "luma_coeffs": luma_coeffs,
            "chroma_coeffs": chroma_coeffs,
        }
        return self.params

    def _fit_monotonic_poly(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        degree: int,
    ) -> NDArray[np.floating]:
        """Fit a polynomial with monotonicity constraint."""
        # Initial guess: least squares polynomial (unconstrained)
        try:
            init_coeffs = np.polyfit(x, y, degree)[::-1]  # low to high degree
        except (np.linalg.LinAlgError, ValueError):
            init_coeffs = np.zeros(degree + 1)
            init_coeffs[1] = 1.0

        # Evaluation points for monotonicity constraint
        check_points = np.linspace(
            max(0.0, float(np.min(x))),
            min(1.0, float(np.max(x))),
            50,
        )

        def objective(coeffs: NDArray) -> float:
            """MSE between polynomial and target."""
            pred = self._eval_poly(x, coeffs)
            return float(np.mean((pred - y) ** 2))

        def monotonicity_penalty(coeffs: NDArray) -> float:
            """Penalty for non-monotonic derivatives."""
            deriv = self._eval_poly_derivative(check_points, coeffs)
            violations = np.minimum(deriv, 0.0)
            return float(np.sum(violations**2))

        def combined_objective(coeffs: NDArray) -> float:
            """Combined objective with monotonicity regularization."""
            mse = objective(coeffs)
            penalty = monotonicity_penalty(coeffs)
            return mse + 100.0 * penalty

        # Optimize
        result = minimize(
            combined_objective,
            init_coeffs,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )

        coeffs = result.x

        # Enforce monotonicity via projection if needed
        coeffs = self._project_monotonic(coeffs, check_points)

        return coeffs

    def _fit_chroma_poly(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]],
        hdr_chroma: Optional[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Fit chroma scaling polynomial as function of luminance."""
        if sdr_chroma is None or hdr_chroma is None:
            coeffs = np.zeros(self.chroma_degree + 1)
            coeffs[0] = 1.0
            return coeffs

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
            coeffs = np.zeros(self.chroma_degree + 1)
            coeffs[0] = 1.0
            return coeffs

        # Chroma ratio as function of SDR luminance
        chroma_ratio = hdr_chroma_mag[valid] / sdr_chroma_mag[valid]
        lum_vals = sdr_y[valid]

        # Clip extreme ratios
        chroma_ratio = np.clip(chroma_ratio, 0.1, 10.0)

        # Fit polynomial
        try:
            coeffs = np.polyfit(lum_vals, chroma_ratio, self.chroma_degree)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            coeffs = np.zeros(self.chroma_degree + 1)
            coeffs[0] = float(np.median(chroma_ratio))

        return coeffs

    def _eval_poly(
        self, x: NDArray[np.floating], coeffs: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Evaluate polynomial: sum(coeffs[i] * x^i)."""
        result = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(coeffs):
            result = result + c * np.power(x, i)
        return result

    def _eval_poly_derivative(
        self, x: NDArray[np.floating], coeffs: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Evaluate polynomial derivative."""
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(1, len(coeffs)):
            result = result + i * coeffs[i] * np.power(x, i - 1)
        return result

    def _project_monotonic(
        self, coeffs: NDArray[np.floating], check_points: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Ensure polynomial is monotonically non-decreasing.

        If derivative is negative at check points, iteratively adjust coefficients.
        Falls back to isotonic regression on the LUT output when violations are severe.
        """
        deriv = self._eval_poly_derivative(check_points, coeffs)
        if np.all(deriv >= -1e-8):
            return coeffs

        # If violation is small, increase linear coefficient
        min_deriv = float(np.min(deriv))
        if min_deriv > -0.5:
            coeffs_fixed = coeffs.copy()
            coeffs_fixed[1] += abs(min_deriv) * 1.1
            deriv_new = self._eval_poly_derivative(check_points, coeffs_fixed)
            if np.all(deriv_new >= -1e-8):
                return coeffs_fixed

        # Hard fallback: apply isotonic regression on the polynomial output
        # evaluated at the check points, then refit a monotonic polynomial
        lut_values = self._eval_poly(check_points, coeffs)
        # Isotonic regression: enforce non-decreasing via cumulative max
        monotonic_values = np.maximum.accumulate(lut_values)

        # Refit polynomial to the corrected (monotonic) LUT values
        try:
            corrected_coeffs = np.polyfit(check_points, monotonic_values, len(coeffs) - 1)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            corrected_coeffs = coeffs.copy()
            corrected_coeffs[1] += abs(min_deriv) * 1.5

        return corrected_coeffs

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, object]] = None,
    ) -> NDArray[np.floating]:
        """Apply monotonic polynomial transform to full SDR image.

        Args:
            sdr_image_linear: (H, W, 3) linear BT.2020 RGB.
            params: Optional override parameters.

        Returns:
            (H, W, 3) HDR linear BT.2020 RGB.
        """
        p = params if params is not None else self.params
        luma_coeffs = np.asarray(p["luma_coeffs"])
        chroma_coeffs = np.asarray(p["chroma_coeffs"])

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)
        lum_3d = lum[..., np.newaxis]

        # Apply luma polynomial
        hdr_lum = self._eval_poly(lum, luma_coeffs)
        hdr_lum = np.maximum(hdr_lum, 0.0)
        hdr_lum_3d = hdr_lum[..., np.newaxis]

        # Compute luminance-dependent chroma scaling
        chroma_scale = self._eval_poly(lum, chroma_coeffs)
        chroma_scale = np.clip(chroma_scale, 0.1, 10.0)
        chroma_scale_3d = chroma_scale[..., np.newaxis]

        # Apply chroma transform
        chroma = sdr_image_linear - lum_3d
        hdr_chroma = chroma_scale_3d * chroma

        # Reconstruct
        result = hdr_lum_3d + hdr_chroma

        return np.maximum(result, 0.0)
