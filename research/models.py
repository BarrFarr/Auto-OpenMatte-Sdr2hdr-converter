"""Seven candidate mathematical models for scene-based SDR-to-HDR reshaping.

Each model implements:
  - fit(sdr_linear, hdr_linear): estimate parameters from overlap region
  - apply(sdr_image): apply fitted transform to full SDR Open Matte image
  - param_count: number of free parameters
  - name: human-readable model identifier

The fitting operates on luminance (BT.2020 Y) extracted from the overlap region,
where both SDR and HDR representations are available. Application transforms the
full SDR image (linear BT.2020 space).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

# BT.2020 luminance weights
_BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


def _luminance(rgb: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute BT.2020 luminance from linear RGB."""
    return np.einsum("...c,c->...", rgb, _BT2020_LUMA)


def _chroma_magnitude(rgb: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute chroma magnitude (distance from achromatic axis)."""
    lum = _luminance(rgb)
    chroma_vec = rgb - lum[..., np.newaxis]
    return np.sqrt(np.sum(chroma_vec**2, axis=-1))


# =============================================================================
# Model A: Linear Gain - 2 params
# =============================================================================

class ModelA:
    """Linear gain: Y' = a*Y, C' = b*C.

    Simplest possible model. Serves as baseline.
    Parameters: a (luminance gain), b (chroma gain).
    """

    name = "A: Linear Gain"
    param_count = 2

    def __init__(self):
        self.params = {"a": 1.0, "b": 1.0}

    def fit(self, sdr_linear, hdr_linear):
        """Fit from flattened luminance arrays of the overlap region."""
        valid = sdr_linear > 1e-6
        if np.sum(valid) < 10:
            self.params = {"a": 1.0, "b": 1.0}
            return
        ratios = hdr_linear[valid] / sdr_linear[valid]
        self.params["a"] = float(np.median(ratios))
        self.params["b"] = self.params["a"]

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        """Fit including chroma gain from RGB overlap pixels."""
        self.fit(sdr_linear, hdr_linear)
        sdr_cm = _chroma_magnitude(sdr_rgb)
        hdr_cm = _chroma_magnitude(hdr_rgb)
        valid = sdr_cm > 1e-6
        if np.sum(valid) > 10:
            self.params["b"] = float(np.median(hdr_cm[valid] / sdr_cm[valid]))

    def apply(self, sdr_image):
        """Apply to (H, W, 3) linear BT.2020 RGB."""
        a = self.params["a"]
        b = self.params["b"]
        lum = _luminance(sdr_image)
        lum3 = lum[..., np.newaxis]
        hdr_lum = a * lum3
        chroma = sdr_image - lum3
        return np.maximum(hdr_lum + b * chroma, 0.0)


# =============================================================================
# Model B: Piecewise Linear - 10 params (8 y-knots + 1 chroma gain + 1 offset)
# =============================================================================

class ModelB:
    """Piecewise linear: 8 control points for luminance curve + chroma.

    Knot x-positions are fixed at SDR percentiles. y-values are fitted.
    Parameters: 8 y-values + 1 chroma gain + 1 chroma offset = 10.
    """

    name = "B: Piecewise Linear"
    param_count = 10

    def __init__(self):
        self.x_knots = np.linspace(0.0, 1.0, 8)
        self.y_knots = np.linspace(0.0, 1.0, 8)
        self.chroma_gain = 1.0
        self.chroma_offset = 0.0

    def fit(self, sdr_linear, hdr_linear):
        """Fit percentile-matched control points."""
        if len(sdr_linear) < 20:
            return
        percentiles = (1, 5, 15, 30, 50, 70, 85, 99)
        self.x_knots = np.percentile(sdr_linear, percentiles)
        for i in range(1, len(self.x_knots)):
            if self.x_knots[i] <= self.x_knots[i - 1]:
                self.x_knots[i] = self.x_knots[i - 1] + 1e-8

        self.y_knots = np.zeros(8)
        for i in range(8):
            low = 0.0 if i == 0 else (self.x_knots[i-1] + self.x_knots[i]) / 2.0
            high = (float(np.max(sdr_linear)) + 1e-10 if i == 7
                    else (self.x_knots[i] + self.x_knots[i+1]) / 2.0)
            mask = (sdr_linear >= low) & (sdr_linear < high)
            if np.sum(mask) > 0:
                self.y_knots[i] = float(np.median(hdr_linear[mask]))
            else:
                self.y_knots[i] = self.x_knots[i]

        for i in range(1, 8):
            if self.y_knots[i] < self.y_knots[i-1]:
                self.y_knots[i] = self.y_knots[i-1]

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        """Fit luma + chroma."""
        self.fit(sdr_linear, hdr_linear)
        sdr_cm = _chroma_magnitude(sdr_rgb)
        hdr_cm = _chroma_magnitude(hdr_rgb)
        valid = sdr_cm > 1e-6
        if np.sum(valid) > 10:
            self.chroma_gain = float(np.median(hdr_cm[valid] / sdr_cm[valid]))

    def apply(self, sdr_image):
        """Apply piecewise linear mapping."""
        lum = _luminance(sdr_image)
        lum3 = lum[..., np.newaxis]
        hdr_lum = np.interp(lum, self.x_knots, self.y_knots)
        hdr_lum3 = hdr_lum[..., np.newaxis]
        chroma = sdr_image - lum3
        return np.maximum(hdr_lum3 + self.chroma_gain * chroma, 0.0)


# =============================================================================
# Model C: Monotonic Polynomial - 8 params (degree 4 luma + 3 chroma poly)
# =============================================================================

class ModelC:
    """Monotonic polynomial: degree 4 for luminance + degree 2 saturation scale.

    Parameters: 5 (luma coeffs) + 3 (chroma coeffs) = 8.
    """

    name = "C: Monotonic Polynomial"
    param_count = 8

    def __init__(self):
        self.luma_coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        self.chroma_coeffs = np.array([1.0, 0.0, 0.0])

    def _eval_poly(self, x, coeffs):
        result = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(coeffs):
            result += c * np.power(x, i)
        return result

    def _eval_deriv(self, x, coeffs):
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(1, len(coeffs)):
            result += i * coeffs[i] * np.power(x, i - 1)
        return result

    def fit(self, sdr_linear, hdr_linear):
        """Fit monotonic polynomial to luminance mapping."""
        if len(sdr_linear) < 20:
            return
        n = len(sdr_linear)
        if n > 5000:
            idx = np.linspace(0, n-1, 5000, dtype=int)
            sx, hx = sdr_linear[idx], hdr_linear[idx]
        else:
            sx, hx = sdr_linear, hdr_linear

        try:
            init = np.polyfit(sx, hx, 4)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            init = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        check_pts = np.linspace(max(0, float(np.min(sx))), min(1, float(np.max(sx))), 50)

        def objective(coeffs):
            pred = self._eval_poly(sx, coeffs)
            mse = float(np.mean((pred - hx)**2))
            deriv = self._eval_deriv(check_pts, coeffs)
            penalty = float(np.sum(np.minimum(deriv, 0.0)**2))
            return mse + 100.0 * penalty

        result = minimize(objective, init, method="L-BFGS-B",
                         options={"maxiter": 500, "ftol": 1e-10})
        self.luma_coeffs = result.x

        deriv = self._eval_deriv(check_pts, self.luma_coeffs)
        if np.any(deriv < -1e-6):
            vals = self._eval_poly(check_pts, self.luma_coeffs)
            vals_mono = np.maximum.accumulate(vals)
            try:
                self.luma_coeffs = np.polyfit(check_pts, vals_mono, 4)[::-1]
            except (np.linalg.LinAlgError, ValueError):
                pass

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        """Fit luma + chroma polynomial."""
        self.fit(sdr_linear, hdr_linear)
        sdr_y = _luminance(sdr_rgb)
        sdr_cm = _chroma_magnitude(sdr_rgb)
        hdr_cm = _chroma_magnitude(hdr_rgb)
        valid = sdr_cm > 1e-6
        if np.sum(valid) < 20:
            self.chroma_coeffs = np.array([1.0, 0.0, 0.0])
            return
        ratio = np.clip(hdr_cm[valid] / sdr_cm[valid], 0.1, 10.0)
        lum_vals = sdr_y[valid]
        try:
            self.chroma_coeffs = np.polyfit(lum_vals, ratio, 2)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            self.chroma_coeffs = np.array([float(np.median(ratio)), 0.0, 0.0])

    def apply(self, sdr_image):
        lum = _luminance(sdr_image)
        lum3 = lum[..., np.newaxis]
        hdr_lum = np.maximum(self._eval_poly(lum, self.luma_coeffs), 0.0)
        hdr_lum3 = hdr_lum[..., np.newaxis]
        chroma_scale = np.clip(self._eval_poly(lum, self.chroma_coeffs), 0.1, 10.0)
        chroma = sdr_image - lum3
        return np.maximum(hdr_lum3 + chroma_scale[..., np.newaxis] * chroma, 0.0)


# =============================================================================
# Model D: CDF Matching - ~65 params (64 LUT + 1 chroma)
# =============================================================================

class ModelD:
    """CDF matching: histogram equalization transfer as 64-point LUT.

    Maps SDR luminance CDF to HDR luminance CDF via inverse transform.
    Parameters: 64 LUT y-values + 1 chroma gain = 65.
    """

    name = "D: CDF Matching"
    param_count = 65

    def __init__(self, lut_size=64):
        self.lut_size = lut_size
        self.lut_x = np.linspace(0.0, 1.0, lut_size)
        self.lut_y = np.linspace(0.0, 1.0, lut_size)
        self.chroma_gain = 1.0

    def fit(self, sdr_linear, hdr_linear):
        """Fit CDF-based luminance mapping."""
        if len(sdr_linear) < 20:
            return
        sdr_sorted = np.sort(sdr_linear)
        sdr_cdf = np.linspace(0.0, 1.0, len(sdr_sorted))
        hdr_sorted = np.sort(hdr_linear)
        hdr_cdf = np.linspace(0.0, 1.0, len(hdr_sorted))

        sdr_max = float(np.max(sdr_linear))
        if sdr_max < 1e-10:
            sdr_max = 1.0
        self.lut_x = np.linspace(0.0, sdr_max, self.lut_size)
        sdr_cdf_at_x = np.interp(self.lut_x, sdr_sorted, sdr_cdf)
        self.lut_y = np.interp(sdr_cdf_at_x, hdr_cdf, hdr_sorted)

        for i in range(1, self.lut_size):
            if self.lut_y[i] < self.lut_y[i-1]:
                self.lut_y[i] = self.lut_y[i-1]

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        self.fit(sdr_linear, hdr_linear)
        sdr_cm = _chroma_magnitude(sdr_rgb)
        hdr_cm = _chroma_magnitude(hdr_rgb)
        valid = sdr_cm > 1e-6
        if np.sum(valid) > 10:
            self.chroma_gain = float(np.median(hdr_cm[valid] / sdr_cm[valid]))

    def apply(self, sdr_image):
        lum = _luminance(sdr_image)
        lum3 = lum[..., np.newaxis]
        hdr_lum = np.interp(lum, self.lut_x, self.lut_y)
        hdr_lum3 = hdr_lum[..., np.newaxis]
        chroma = sdr_image - lum3
        return np.maximum(hdr_lum3 + self.chroma_gain * chroma, 0.0)


# =============================================================================
# Model E: CDF + Regularized Optimization - 12 params
# =============================================================================

class ModelE:
    """CDF + regularized smooth curve with 12 control points.

    Starts from CDF mapping, then fits a smooth 12-point spline
    optimized to minimize reconstruction error + smoothness penalty.
    Parameters: 12 (control point y-values, x fixed at uniform positions).
    """

    name = "E: CDF + Regularized"
    param_count = 12

    def __init__(self, n_ctrl=12, smooth_lambda=0.05):
        self.n_ctrl = n_ctrl
        self.smooth_lambda = smooth_lambda
        self.ctrl_x = np.linspace(0.0, 1.0, n_ctrl)
        self.ctrl_y = np.linspace(0.0, 1.0, n_ctrl)

    def fit(self, sdr_linear, hdr_linear):
        """Fit regularized CDF-based mapping with 12 control points."""
        if len(sdr_linear) < 20:
            return
        sdr_max = float(np.max(sdr_linear))
        if sdr_max < 1e-10:
            sdr_max = 1.0

        sdr_sorted = np.sort(sdr_linear)
        sdr_cdf = np.linspace(0.0, 1.0, len(sdr_sorted))
        hdr_sorted = np.sort(hdr_linear)
        hdr_cdf = np.linspace(0.0, 1.0, len(hdr_sorted))

        self.ctrl_x = np.linspace(0.0, sdr_max, self.n_ctrl)
        sdr_cdf_at_x = np.interp(self.ctrl_x, sdr_sorted, sdr_cdf)
        init_y = np.interp(sdr_cdf_at_x, hdr_cdf, hdr_sorted)

        for i in range(1, self.n_ctrl):
            if init_y[i] < init_y[i-1]:
                init_y[i] = init_y[i-1]

        n = len(sdr_linear)
        if n > 3000:
            idx = np.linspace(0, n-1, 3000, dtype=int)
            sx, hx = sdr_linear[idx], hdr_linear[idx]
        else:
            sx, hx = sdr_linear, hdr_linear

        def objective(y_vals):
            y_mono = np.maximum.accumulate(y_vals)
            pred = np.interp(sx, self.ctrl_x, y_mono)
            mse = float(np.mean((pred - hx)**2))
            d2 = np.diff(y_vals, 2)
            smoothness = float(np.sum(d2**2))
            return mse + self.smooth_lambda * smoothness

        result = minimize(objective, init_y, method="L-BFGS-B",
                         bounds=[(0.0, None) for _ in range(self.n_ctrl)],
                         options={"maxiter": 300, "ftol": 1e-10})
        self.ctrl_y = np.maximum.accumulate(result.x)

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        self.fit(sdr_linear, hdr_linear)

    def apply(self, sdr_image):
        lum = _luminance(sdr_image)
        lum3 = lum[..., np.newaxis]
        hdr_lum = np.interp(lum, self.ctrl_x, self.ctrl_y)
        safe_lum = np.maximum(lum, 1e-10)
        ratio = hdr_lum / safe_lum
        result = sdr_image * ratio[..., np.newaxis]
        return np.maximum(result, 0.0)


# =============================================================================
# Model F: Luma Polynomial + Chroma 3x3 Matrix Regression - 18 params
# =============================================================================

class ModelF:
    """Luma polynomial (degree 4) + chroma 3x3 matrix regression.

    Luminance: degree-4 monotonic polynomial (5 coeffs).
    Chroma: 3x3 matrix (9 params) + saturation scale (1) + 3 offset = 18 total.
    Parameters: 5 + 9 + 1 + 3 = 18.
    """

    name = "F: Luma + Chroma Matrix"
    param_count = 18

    def __init__(self):
        self.luma_coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        self.color_matrix = np.eye(3, dtype=np.float64)
        self.saturation = 1.0
        self.offset = np.zeros(3, dtype=np.float64)

    def _eval_poly(self, x, coeffs):
        result = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(coeffs):
            result += c * np.power(x, i)
        return result

    def fit(self, sdr_linear, hdr_linear):
        """Fit luminance polynomial."""
        if len(sdr_linear) < 20:
            return
        n = len(sdr_linear)
        if n > 5000:
            idx = np.linspace(0, n-1, 5000, dtype=int)
            sx, hx = sdr_linear[idx], hdr_linear[idx]
        else:
            sx, hx = sdr_linear, hdr_linear

        try:
            init = np.polyfit(sx, hx, 4)[::-1]
        except (np.linalg.LinAlgError, ValueError):
            init = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        check_pts = np.linspace(max(0, float(np.min(sx))), min(1, float(np.max(sx))), 50)

        def objective(coeffs):
            pred = self._eval_poly(sx, coeffs)
            mse = float(np.mean((pred - hx)**2))
            deriv = np.zeros_like(check_pts)
            for i in range(1, len(coeffs)):
                deriv += i * coeffs[i] * np.power(check_pts, i-1)
            penalty = float(np.sum(np.minimum(deriv, 0.0)**2))
            return mse + 100.0 * penalty

        result = minimize(objective, init, method="L-BFGS-B",
                         options={"maxiter": 500, "ftol": 1e-10})
        self.luma_coeffs = result.x

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        """Fit luma + 3x3 color matrix."""
        self.fit(sdr_linear, hdr_linear)
        n = len(sdr_rgb)
        if n > 3000:
            idx = np.linspace(0, n-1, 3000, dtype=int)
            S = sdr_rgb[idx]
            H = hdr_rgb[idx]
        else:
            S = sdr_rgb
            H = hdr_rgb

        St = S.T
        Ht = H.T
        lam = 0.5 * St.shape[1]
        reg = lam * np.eye(3)
        try:
            self.color_matrix = (Ht @ St.T + reg) @ np.linalg.inv(St @ St.T + reg)
        except np.linalg.LinAlgError:
            self.color_matrix = np.eye(3)

        sdr_cm = _chroma_magnitude(S)
        hdr_cm = _chroma_magnitude(H)
        valid = sdr_cm > 1e-6
        if np.sum(valid) > 10:
            self.saturation = float(np.median(hdr_cm[valid] / sdr_cm[valid]))

    def apply(self, sdr_image):
        lum = _luminance(sdr_image)
        lum3 = lum[..., np.newaxis]
        hdr_lum = np.maximum(self._eval_poly(lum, self.luma_coeffs), 0.0)
        hdr_lum3 = hdr_lum[..., np.newaxis]

        cc = np.einsum("ij,...j->...i", self.color_matrix, sdr_image)
        cc_lum = _luminance(cc)
        cc_lum3 = cc_lum[..., np.newaxis]
        chroma = cc - cc_lum3

        result = hdr_lum3 + self.saturation * chroma
        return np.maximum(result, 0.0)


# =============================================================================
# Model G: Hybrid - 29 params
# =============================================================================

class ModelG:
    """Hybrid: percentile mapping + highlight/shadow params + chroma regression.

    Luminance: 10-pt piecewise (percentile) + highlight shoulder (3) + shadow lift (2).
    Chroma: saturation (2: base + slope) + 3x3 matrix (9) + hue rotation (3).
    Total: 10 + 3 + 2 + 2 + 9 + 3 = 29.
    """

    name = "G: Hybrid"
    param_count = 29

    def __init__(self, n_luma_pts=10):
        self.n_luma_pts = n_luma_pts
        self.luma_x = np.linspace(0.0, 1.0, n_luma_pts)
        self.luma_y = np.linspace(0.0, 1.0, n_luma_pts)
        self.hl_threshold = 0.8
        self.hl_slope = 1.0
        self.hl_max = 1.0
        self.sh_threshold = 0.05
        self.sh_gain = 1.0
        self.sat_base = 1.0
        self.sat_slope = 0.0
        self.color_matrix = np.eye(3, dtype=np.float64)
        self.hue_angles = np.zeros(3)

    def fit(self, sdr_linear, hdr_linear):
        """Fit all luminance components."""
        if len(sdr_linear) < 20:
            return

        percentiles = np.linspace(1, 99, self.n_luma_pts)
        self.luma_x = np.percentile(sdr_linear, percentiles)
        for i in range(1, len(self.luma_x)):
            if self.luma_x[i] <= self.luma_x[i-1]:
                self.luma_x[i] = self.luma_x[i-1] + 1e-8

        self.luma_y = np.zeros(self.n_luma_pts)
        for i in range(self.n_luma_pts):
            low = 0.0 if i == 0 else (self.luma_x[i-1] + self.luma_x[i]) / 2.0
            high = (float(np.max(sdr_linear)) + 1e-10 if i == self.n_luma_pts - 1
                    else (self.luma_x[i] + self.luma_x[i+1]) / 2.0)
            mask = (sdr_linear >= low) & (sdr_linear < high)
            if np.sum(mask) > 0:
                self.luma_y[i] = float(np.median(hdr_linear[mask]))
            else:
                self.luma_y[i] = self.luma_x[i]

        for i in range(1, self.n_luma_pts):
            if self.luma_y[i] < self.luma_y[i-1]:
                self.luma_y[i] = self.luma_y[i-1]

        p90 = float(np.percentile(sdr_linear, 90))
        self.hl_threshold = max(p90, 1e-6)
        self.hl_max = float(np.max(hdr_linear))
        hi_mask = sdr_linear >= p90
        if np.sum(hi_mask) > 10:
            base_pred = np.interp(sdr_linear[hi_mask], self.luma_x, self.luma_y)
            residual = hdr_linear[hi_mask] - base_pred
            sdr_hi = sdr_linear[hi_mask]
            if float(np.max(sdr_hi)) - float(np.min(sdr_hi)) > 1e-8:
                try:
                    sf = np.polyfit(sdr_hi - p90, residual, 1)
                    self.hl_slope = max(float(sf[0]) + 1.0, 0.1)
                except (np.linalg.LinAlgError, ValueError):
                    self.hl_slope = 1.0
            else:
                self.hl_slope = 1.0

        valid_sdr = sdr_linear[sdr_linear > 1e-8]
        if len(valid_sdr) > 10:
            p10 = float(np.percentile(valid_sdr, 10))
        else:
            p10 = 0.01
        self.sh_threshold = p10
        sh_mask = (sdr_linear <= p10) & (sdr_linear > 1e-8)
        if np.sum(sh_mask) > 5:
            self.sh_gain = float(np.median(hdr_linear[sh_mask] / sdr_linear[sh_mask]))
        else:
            self.sh_gain = 1.0

    def fit_with_chroma(self, sdr_linear, hdr_linear, sdr_rgb, hdr_rgb):
        """Fit all parameters including chroma components."""
        self.fit(sdr_linear, hdr_linear)

        sdr_y = _luminance(sdr_rgb)
        sdr_cm = _chroma_magnitude(sdr_rgb)
        hdr_cm = _chroma_magnitude(hdr_rgb)
        valid = sdr_cm > 1e-6
        if np.sum(valid) > 20:
            ratio = np.clip(hdr_cm[valid] / sdr_cm[valid], 0.1, 10.0)
            lum_v = sdr_y[valid]
            try:
                coeffs = np.polyfit(lum_v, ratio, 1)
                self.sat_base = float(coeffs[1])
                self.sat_slope = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                self.sat_base = float(np.median(ratio))
                self.sat_slope = 0.0
        else:
            self.sat_base = 1.0
            self.sat_slope = 0.0

        n = len(sdr_rgb)
        if n > 3000:
            idx = np.linspace(0, n-1, 3000, dtype=int)
            S, H = sdr_rgb[idx], hdr_rgb[idx]
        else:
            S, H = sdr_rgb, hdr_rgb
        St = S.T
        Ht = H.T
        lam = 1.0 * St.shape[1]
        reg = lam * np.eye(3)
        try:
            self.color_matrix = (Ht @ St.T + reg) @ np.linalg.inv(St @ St.T + reg)
        except np.linalg.LinAlgError:
            self.color_matrix = np.eye(3)

        sdr_c = sdr_rgb - sdr_y[..., np.newaxis]
        hdr_y2 = _luminance(hdr_rgb)
        hdr_c = hdr_rgb - hdr_y2[..., np.newaxis]
        self.hue_angles = np.zeros(3)
        for ch in range(3):
            other = [c for c in range(3) if c != ch]
            sdr_2d = sdr_c[..., other]
            hdr_2d = hdr_c[..., other]
            mag = np.sqrt(np.sum(sdr_2d**2, axis=-1))
            v = mag > 1e-6
            if np.sum(v) < 20:
                continue
            cross = sdr_2d[v, 0] * hdr_2d[v, 1] - sdr_2d[v, 1] * hdr_2d[v, 0]
            dot = np.sum(sdr_2d[v] * hdr_2d[v], axis=-1)
            angle = float(np.median(np.arctan2(cross, dot)))
            self.hue_angles[ch] = np.clip(angle, -0.1, 0.1)

    def _apply_highlight(self, lum, hdr_lum):
        above = lum > self.hl_threshold
        if not np.any(above):
            return hdr_lum
        result = hdr_lum.copy()
        excess = lum[above] - self.hl_threshold
        base_val = np.interp(np.array([self.hl_threshold]), self.luma_x, self.luma_y)[0]
        compressed = base_val + self.hl_max * (
            1.0 - np.exp(-self.hl_slope * excess / max(self.hl_max, 1e-6)))
        result[above] = compressed
        return result

    def _apply_shadow(self, lum, hdr_lum):
        below = lum < self.sh_threshold
        if not np.any(below):
            return hdr_lum
        result = hdr_lum.copy()
        t = lum[below] / max(self.sh_threshold, 1e-8)
        base_val = result[below]
        shadow_val = self.sh_gain * lum[below]
        result[below] = (1.0 - t) * shadow_val + t * base_val
        return result

    def _hue_matrix(self):
        ax, ay, az = self.hue_angles
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az), np.cos(az), 0],
                       [0, 0, 1]], dtype=np.float64)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax), np.cos(ax)]], dtype=np.float64)
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]], dtype=np.float64)
        return Rz @ Ry @ Rx

    def apply(self, sdr_image):
        lum = _luminance(sdr_image)
        hdr_lum = np.interp(lum, self.luma_x, self.luma_y)
        hdr_lum = self._apply_highlight(lum, hdr_lum)
        hdr_lum = self._apply_shadow(lum, hdr_lum)
        hdr_lum3 = hdr_lum[..., np.newaxis]

        cc = np.einsum("ij,...j->...i", self.color_matrix, sdr_image)
        if np.any(np.abs(self.hue_angles) > 1e-6):
            hm = self._hue_matrix()
            cc = np.einsum("ij,...j->...i", hm, cc)

        cc_lum = _luminance(cc)
        cc_lum3 = cc_lum[..., np.newaxis]
        chroma = cc - cc_lum3

        sat_scale = np.clip(self.sat_base + self.sat_slope * lum, 0.1, 10.0)
        sat3 = sat_scale[..., np.newaxis]

        result = hdr_lum3 + sat3 * chroma
        return np.maximum(result, 0.0)


# =============================================================================
# Convenience: get all models
# =============================================================================

def get_all_models():
    """Return instances of all 7 candidate models."""
    return [ModelA(), ModelB(), ModelC(), ModelD(), ModelE(), ModelF(), ModelG()]
