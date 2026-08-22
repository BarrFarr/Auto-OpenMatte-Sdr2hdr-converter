"""Model G: Hybrid - multi-component model inspired by Dolby backward reshaping.

Luminance base: percentile/CDF mapping (8-10 control points from percentile matching).
Highlight shoulder: parametric rolloff in highlights (sigmoid-like, 3 params).
Shadow lift: parametric shadow region adjustment (2 params: threshold, gain).
Chroma: luminance-dependent saturation (linear: sat = s0 + s1*Y, 2 params) + 3x3 matrix.
Optional hue: small hue rotation (3 params, one per axis, constrained small).

Total: 10 (luma) + 3 (highlight) + 2 (shadow) + 2 (sat) + 9 (matrix) + 3 (hue) = ~29 params.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import BT2020_LUMA

DEFAULT_NUM_LUMA_POINTS = 10
IDENTITY_3X3 = np.eye(3, dtype=np.float64)


class ModelGHybrid:
    """Hybrid multi-component reshaping model.

    Separates luminance and chroma processing paths with specialized
    handling for highlights and shadows, similar to Dolby backward
    reshaping patents. Scene-based fitting with CDF as starting point,
    parametric refinement for extreme regions.
    """

    def __init__(
        self,
        num_luma_points: int = DEFAULT_NUM_LUMA_POINTS,
        matrix_lambda: float = 1.0,
        hue_constraint: float = 0.1,
    ) -> None:
        self.num_luma_points = num_luma_points
        self.matrix_lambda = matrix_lambda
        self.hue_constraint = hue_constraint
        self.params: Dict[str, object] = {
            # Luminance base
            "luma_x": np.linspace(0.0, 1.0, num_luma_points),
            "luma_y": np.linspace(0.0, 1.0, num_luma_points),
            # Highlight shoulder: threshold, slope, max_output
            "highlight_threshold": 0.8,
            "highlight_slope": 1.0,
            "highlight_max": 1.0,
            # Shadow lift: threshold, gain
            "shadow_threshold": 0.05,
            "shadow_gain": 1.0,
            # Chroma: luminance-dependent saturation
            "sat_base": 1.0,
            "sat_slope": 0.0,
            # Color matrix (3x3)
            "color_matrix": IDENTITY_3X3.copy(),
            # Hue rotation (3 small angles)
            "hue_angles": np.zeros(3),
        }

    def name(self) -> str:
        """Return model name."""
        return "Model G: Hybrid"

    def param_count(self) -> int:
        """Return number of parameters."""
        return self.num_luma_points + 3 + 2 + 2 + 9 + 3  # = 29

    def fit(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        sdr_chroma: Optional[NDArray[np.floating]] = None,
        hdr_chroma: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, object]:
        """Estimate all hybrid model parameters from overlap region.

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

        # Step 1: Fit base luminance mapping via percentile matching
        luma_x, luma_y = self._fit_luma_base(sdr_linear, hdr_linear)

        # Step 2: Fit highlight shoulder
        highlight_params = self._fit_highlight(sdr_linear, hdr_linear, luma_x, luma_y)

        # Step 3: Fit shadow lift
        shadow_params = self._fit_shadow(sdr_linear, hdr_linear, luma_x, luma_y)

        # Step 4: Fit chroma components
        sat_base, sat_slope = 1.0, 0.0
        color_matrix = IDENTITY_3X3.copy()
        hue_angles = np.zeros(3)

        if sdr_chroma is not None and hdr_chroma is not None:
            sat_base, sat_slope = self._fit_saturation(sdr_chroma, hdr_chroma)
            color_matrix = self._fit_color_matrix(sdr_chroma, hdr_chroma)
            hue_angles = self._fit_hue_rotation(sdr_chroma, hdr_chroma)

        self.params = {
            "luma_x": luma_x,
            "luma_y": luma_y,
            "highlight_threshold": highlight_params[0],
            "highlight_slope": highlight_params[1],
            "highlight_max": highlight_params[2],
            "shadow_threshold": shadow_params[0],
            "shadow_gain": shadow_params[1],
            "sat_base": sat_base,
            "sat_slope": sat_slope,
            "color_matrix": color_matrix,
            "hue_angles": hue_angles,
        }
        return self.params

    def _fit_luma_base(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Fit base luminance via percentile matching."""
        percentiles = np.linspace(1, 99, self.num_luma_points)
        luma_x = np.percentile(sdr_linear, percentiles)

        # Ensure strictly increasing
        for i in range(1, len(luma_x)):
            if luma_x[i] <= luma_x[i - 1]:
                luma_x[i] = luma_x[i - 1] + 1e-8

        # Median HDR at each percentile bin
        luma_y = np.zeros(self.num_luma_points)
        for i in range(self.num_luma_points):
            if i == 0:
                low = 0.0
            else:
                low = (luma_x[i - 1] + luma_x[i]) / 2.0
            if i == self.num_luma_points - 1:
                high = float(np.max(sdr_linear)) + 1e-10
            else:
                high = (luma_x[i] + luma_x[i + 1]) / 2.0

            mask = (sdr_linear >= low) & (sdr_linear < high)
            if np.sum(mask) > 0:
                luma_y[i] = float(np.median(hdr_linear[mask]))
            else:
                luma_y[i] = luma_x[i]

        # Enforce monotonicity
        for i in range(1, len(luma_y)):
            if luma_y[i] < luma_y[i - 1]:
                luma_y[i] = luma_y[i - 1]

        return luma_x, luma_y

    def _fit_highlight(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        luma_x: NDArray[np.floating],
        luma_y: NDArray[np.floating],
    ) -> Tuple[float, float, float]:
        """Fit highlight shoulder parameters."""
        # Find the 90th percentile of SDR as highlight threshold
        p90 = float(np.percentile(sdr_linear, 90))
        if p90 < 1e-6:
            return (0.8, 1.0, 1.0)

        # Get base prediction at highlight region
        highlight_mask = sdr_linear >= p90
        if np.sum(highlight_mask) < 10:
            return (p90, 1.0, float(np.max(hdr_linear)))

        sdr_hi = sdr_linear[highlight_mask]
        hdr_hi = hdr_linear[highlight_mask]
        base_pred = np.interp(sdr_hi, luma_x, luma_y)

        # Residual in highlights
        residual = hdr_hi - base_pred

        # Estimate slope in highlights
        sdr_range = float(np.max(sdr_hi)) - float(np.min(sdr_hi))
        if sdr_range > 1e-8:
            try:
                slope_fit = np.polyfit(sdr_hi - p90, residual, 1)
                slope = float(slope_fit[0]) + 1.0
            except (np.linalg.LinAlgError, ValueError):
                slope = 1.0
        else:
            slope = 1.0

        highlight_max = float(np.max(hdr_linear))

        return (p90, max(slope, 0.1), highlight_max)

    def _fit_shadow(
        self,
        sdr_linear: NDArray[np.floating],
        hdr_linear: NDArray[np.floating],
        luma_x: NDArray[np.floating],
        luma_y: NDArray[np.floating],
    ) -> Tuple[float, float]:
        """Fit shadow lift parameters."""
        # Find the 10th percentile as shadow threshold
        positive_mask = sdr_linear > 1e-8
        if np.sum(positive_mask) > 10:
            p10 = float(np.percentile(sdr_linear[positive_mask], 10))
        else:
            p10 = 0.01

        shadow_mask = (sdr_linear <= p10) & (sdr_linear > 1e-8)
        if np.sum(shadow_mask) < 5:
            return (p10, 1.0)

        sdr_sh = sdr_linear[shadow_mask]
        hdr_sh = hdr_linear[shadow_mask]

        # Shadow gain: ratio in shadow region
        valid = sdr_sh > 1e-8
        if np.sum(valid) > 3:
            shadow_gain = float(np.median(hdr_sh[valid] / sdr_sh[valid]))
        else:
            shadow_gain = 1.0

        return (p10, max(shadow_gain, 0.0))

    def _fit_saturation(
        self,
        sdr_chroma: NDArray[np.floating],
        hdr_chroma: NDArray[np.floating],
    ) -> Tuple[float, float]:
        """Fit luminance-dependent saturation: sat = s0 + s1 * Y."""
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
        ratio = np.clip(ratio, 0.1, 10.0)

        try:
            coeffs = np.polyfit(lum_vals, ratio, 1)
            return float(coeffs[1]), float(coeffs[0])  # intercept, slope
        except (np.linalg.LinAlgError, ValueError):
            return float(np.median(ratio)), 0.0

    def _fit_color_matrix(
        self,
        sdr_chroma: NDArray[np.floating],
        hdr_chroma: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Fit 3x3 color matrix with regularization toward identity."""
        n = len(sdr_chroma)
        if n > 3000:
            idx = np.linspace(0, n - 1, 3000, dtype=int)
            sdr_sub = sdr_chroma[idx]
            hdr_sub = hdr_chroma[idx]
        else:
            sdr_sub = sdr_chroma
            hdr_sub = hdr_chroma

        S = sdr_sub.T  # (3, N)
        H = hdr_sub.T  # (3, N)
        n_samples = S.shape[1]

        # Regularized least squares
        reg = self.matrix_lambda * np.eye(3) * n_samples
        try:
            M = (H @ S.T + reg) @ np.linalg.inv(S @ S.T + reg)
        except np.linalg.LinAlgError:
            M = IDENTITY_3X3.copy()

        return M

    def _fit_hue_rotation(
        self,
        sdr_chroma: NDArray[np.floating],
        hdr_chroma: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Fit small hue rotation angles (one per RGB axis)."""
        sdr_y = np.einsum("...c,c->...", sdr_chroma, BT2020_LUMA)
        hdr_y = np.einsum("...c,c->...", hdr_chroma, BT2020_LUMA)

        sdr_c = sdr_chroma - sdr_y[..., np.newaxis]
        hdr_c = hdr_chroma - hdr_y[..., np.newaxis]

        # Estimate hue shift per channel as small angle
        angles = np.zeros(3)
        for ch in range(3):
            other = [c for c in range(3) if c != ch]
            sdr_2d = sdr_c[..., other]
            hdr_2d = hdr_c[..., other]

            valid = np.sqrt(np.sum(sdr_2d**2, axis=-1)) > 1e-6
            if np.sum(valid) < 20:
                continue

            sdr_v = sdr_2d[valid]
            hdr_v = hdr_2d[valid]

            cross = sdr_v[:, 0] * hdr_v[:, 1] - sdr_v[:, 1] * hdr_v[:, 0]
            dot = np.sum(sdr_v * hdr_v, axis=-1)

            angle = np.median(np.arctan2(cross, dot))
            angles[ch] = float(np.clip(angle, -self.hue_constraint, self.hue_constraint))

        return angles

    def _apply_highlight_shoulder(
        self,
        lum: NDArray[np.floating],
        hdr_lum: NDArray[np.floating],
        luma_x: NDArray[np.floating],
        luma_y: NDArray[np.floating],
        threshold: float,
        slope: float,
        max_val: float,
    ) -> NDArray[np.floating]:
        """Apply highlight shoulder rolloff."""
        above = lum > threshold
        if not np.any(above):
            return hdr_lum

        result = hdr_lum.copy()
        excess = lum[above] - threshold
        # Base value at threshold
        base = float(np.interp(threshold, luma_x, luma_y))
        # Sigmoid-like soft compression
        compressed = base + max_val * (1.0 - np.exp(-slope * excess / max(max_val, 1e-6)))
        result[above] = compressed

        return result

    def _apply_shadow_lift(
        self,
        lum: NDArray[np.floating],
        hdr_lum: NDArray[np.floating],
        threshold: float,
        gain: float,
    ) -> NDArray[np.floating]:
        """Apply shadow lift adjustment."""
        below = lum < threshold
        if not np.any(below):
            return hdr_lum

        result = hdr_lum.copy()
        # Smooth blend from shadow_gain to base mapping
        t = lum[below] / max(threshold, 1e-8)  # 0 at black, 1 at threshold
        base_val = result[below]
        shadow_val = gain * lum[below]
        result[below] = (1.0 - t) * shadow_val + t * base_val

        return result

    def _build_hue_matrix(self, angles: NDArray[np.floating]) -> NDArray[np.floating]:
        """Build rotation matrix from 3 small hue angles."""
        ax, ay, az = angles

        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)],
        ])
        Ry = np.array([
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)],
        ])

        return Rz @ Ry @ Rx

    def apply(
        self,
        sdr_image_linear: NDArray[np.floating],
        params: Optional[Dict[str, object]] = None,
    ) -> NDArray[np.floating]:
        """Apply hybrid model transform to full SDR image.

        Args:
            sdr_image_linear: (H, W, 3) linear BT.2020 RGB.
            params: Optional override parameters.

        Returns:
            (H, W, 3) HDR linear BT.2020 RGB.
        """
        p = params if params is not None else self.params

        luma_x = np.asarray(p["luma_x"])
        luma_y = np.asarray(p["luma_y"])
        color_matrix = np.asarray(p["color_matrix"])
        hue_angles = np.asarray(p["hue_angles"])
        sat_base = float(p["sat_base"])
        sat_slope = float(p["sat_slope"])
        highlight_threshold = float(p["highlight_threshold"])
        highlight_slope = float(p["highlight_slope"])
        highlight_max = float(p["highlight_max"])
        shadow_threshold = float(p["shadow_threshold"])
        shadow_gain = float(p["shadow_gain"])

        # Compute luminance
        lum = np.einsum("...c,c->...", sdr_image_linear, BT2020_LUMA)

        # Apply base luminance mapping
        hdr_lum = np.interp(lum, luma_x, luma_y)

        # Apply highlight shoulder
        hdr_lum = self._apply_highlight_shoulder(
            lum, hdr_lum, luma_x, luma_y,
            highlight_threshold, highlight_slope, highlight_max
        )

        # Apply shadow lift
        hdr_lum = self._apply_shadow_lift(
            lum, hdr_lum, shadow_threshold, shadow_gain
        )

        hdr_lum_3d = hdr_lum[..., np.newaxis]

        # Apply color matrix to input
        color_corrected = np.einsum("ij,...j->...i", color_matrix, sdr_image_linear)

        # Apply hue rotation
        if np.any(np.abs(hue_angles) > 1e-6):
            hue_matrix = self._build_hue_matrix(hue_angles)
            color_corrected = np.einsum("ij,...j->...i", hue_matrix, color_corrected)

        # Extract chroma from color-corrected
        cc_lum = np.einsum("...c,c->...", color_corrected, BT2020_LUMA)
        cc_lum_3d = cc_lum[..., np.newaxis]
        chroma = color_corrected - cc_lum_3d

        # Luminance-dependent saturation
        sat_scale = sat_base + sat_slope * lum
        sat_scale = np.clip(sat_scale, 0.1, 10.0)
        sat_scale_3d = sat_scale[..., np.newaxis]

        # Reconstruct
        result = hdr_lum_3d + sat_scale_3d * chroma

        return np.maximum(result, 0.0)
