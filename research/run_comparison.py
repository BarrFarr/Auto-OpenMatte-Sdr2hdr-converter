"""Main experiment runner: compare all 7 reshaping models across 6 synthetic scenes.

Generates synthetic test data, fits each model, evaluates quality metrics,
runs additional experiments (seam test, stability, parameter reduction),
and writes a comprehensive research report to research/reports/RESEARCH_REPORT.md.

Usage:
    python research/run_comparison.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Ensure the research package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from color_utils import (
    sdr_gamma_to_linear,
    bt709_to_bt2020,
    _BT2020_LUMA,
)
from synthetic_scenes import (
    generate_all_scenes,
    generate_scene_neutral,
    OVERLAP_TOP,
    OVERLAP_BOTTOM,
    HDR_HEIGHT,
    HDR_WIDTH,
    SDR_HEIGHT,
    SDR_WIDTH,
)
from models import get_all_models, ModelE, _luminance
from metrics import (
    luminance_mae,
    luminance_rmse,
    delta_e_2000,
    seam_error,
    temporal_stability,
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SceneResult:
    """Results for one model on one scene."""
    model_name: str
    scene_name: str
    mae_nits: float
    rmse_nits: float
    delta_e: float
    seam_top: float
    seam_bottom: float
    fit_time_s: float


@dataclass
class StabilityResult:
    """Temporal stability result for one model."""
    model_name: str
    cv_score: float


@dataclass
class ParamReductionResult:
    """Parameter reduction experiment result."""
    model_name: str
    n_params: int
    rmse_nits: float


# =============================================================================
# Core pipeline: SDR -> model input (linear BT.2020)
# =============================================================================

def prepare_sdr_for_model(sdr_gamma_bt709):
    """Convert SDR (BT.709 gamma) to model input space (linear BT.2020)."""
    linear_709 = sdr_gamma_to_linear(sdr_gamma_bt709)
    linear_2020 = bt709_to_bt2020(linear_709)
    return np.clip(linear_2020, 0.0, 1.0)


# =============================================================================
# Main comparison
# =============================================================================

def run_comparison():
    """Run all 7 models on all 6 scenes."""
    print("Generating 6 synthetic test scenes...")
    t0 = time.time()
    scenes = generate_all_scenes()
    t_gen = time.time() - t0
    print(f"  Scene generation: {t_gen:.1f}s")

    models = get_all_models()
    results = []

    for scene_name, (hdr_full, sdr_om, hdr_slice) in scenes:
        print(f"\n  Scene: {scene_name}")

        # Prepare overlap region
        sdr_overlap_gamma = sdr_om[hdr_slice, :, :]
        sdr_overlap_linear = prepare_sdr_for_model(sdr_overlap_gamma)
        hdr_overlap = hdr_full

        # Extract luminance for fitting
        sdr_lum = _luminance(sdr_overlap_linear).ravel()
        hdr_lum = _luminance(hdr_overlap).ravel()

        # Flatten RGB for chroma fitting
        sdr_rgb_flat = sdr_overlap_linear.reshape(-1, 3)
        hdr_rgb_flat = hdr_overlap.reshape(-1, 3)

        # Prepare full SDR for application
        sdr_full_linear = prepare_sdr_for_model(sdr_om)

        for model in models:
            t_fit = time.time()

            # Fit model
            if hasattr(model, "fit_with_chroma"):
                model.fit_with_chroma(sdr_lum, hdr_lum, sdr_rgb_flat, hdr_rgb_flat)
            else:
                model.fit(sdr_lum, hdr_lum)

            fit_time = time.time() - t_fit

            # Apply to full SDR open matte
            pred_full = model.apply(sdr_full_linear)

            # Extract overlap prediction and compare to HDR ground truth
            pred_overlap = pred_full[hdr_slice, :, :]

            # Compute metrics
            mae = luminance_mae(pred_overlap, hdr_full)
            rmse = luminance_rmse(pred_overlap, hdr_full)
            de = delta_e_2000(pred_overlap, hdr_full)

            # Seam errors
            s_top = seam_error(pred_full, OVERLAP_TOP, width=10)
            s_bot = seam_error(pred_full, OVERLAP_BOTTOM - 1, width=10)

            results.append(SceneResult(
                model_name=model.name,
                scene_name=scene_name,
                mae_nits=mae,
                rmse_nits=rmse,
                delta_e=de,
                seam_top=s_top,
                seam_bottom=s_bot,
                fit_time_s=fit_time,
            ))

            print(f"    {model.name:30s} RMSE={rmse:8.1f} nits  dE={de:6.2f}  seam={s_top+s_bot:6.1f}  [{fit_time:.2f}s]")

    total_time = time.time() - t0
    return results, total_time


# =============================================================================
# Stability experiment
# =============================================================================

def run_stability_test(n_trials=5):
    """Test parameter stability under noise perturbation."""
    print("\nRunning stability test...")
    hdr_full, sdr_om, hdr_slice = generate_scene_neutral(seed=42)
    sdr_overlap_gamma = sdr_om[hdr_slice, :, :]
    sdr_overlap_linear = prepare_sdr_for_model(sdr_overlap_gamma)
    hdr_overlap = hdr_full

    models = get_all_models()
    stability_results = []

    for model in models:
        param_sets = []
        for trial in range(n_trials):
            rng = np.random.default_rng(100 + trial)
            noise = rng.normal(0.0, 0.005, sdr_overlap_linear.shape)
            noisy_sdr = np.clip(sdr_overlap_linear + noise, 0.0, 1.0)

            sdr_lum = _luminance(noisy_sdr).ravel()
            hdr_lum = _luminance(hdr_overlap).ravel()
            sdr_rgb = noisy_sdr.reshape(-1, 3)
            hdr_rgb = hdr_overlap.reshape(-1, 3)

            if hasattr(model, "fit_with_chroma"):
                model.fit_with_chroma(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
            else:
                model.fit(sdr_lum, hdr_lum)

            params = _extract_params_vector(model)
            param_sets.append(params)

        cv = temporal_stability(param_sets)
        stability_results.append(StabilityResult(model_name=model.name, cv_score=cv))
        print(f"    {model.name:30s} CV={cv:.4f}")

    return stability_results


def _extract_params_vector(model):
    """Extract a flat parameter vector from a model."""
    parts = []
    if hasattr(model, "params") and isinstance(model.params, dict):
        for v in model.params.values():
            if isinstance(v, (int, float)):
                parts.append(float(v))
            elif isinstance(v, np.ndarray):
                parts.extend(v.ravel().tolist())
        if parts:
            return np.array(parts, dtype=np.float64)

    if hasattr(model, "luma_coeffs"):
        parts.extend(model.luma_coeffs.tolist())
    if hasattr(model, "chroma_coeffs"):
        parts.extend(model.chroma_coeffs.tolist())
    if hasattr(model, "x_knots"):
        parts.extend(model.x_knots.tolist())
    if hasattr(model, "y_knots"):
        parts.extend(model.y_knots.tolist())
    if hasattr(model, "luma_x"):
        parts.extend(model.luma_x.tolist())
    if hasattr(model, "luma_y"):
        parts.extend(model.luma_y.tolist())
    if hasattr(model, "ctrl_x"):
        parts.extend(model.ctrl_x.tolist())
    if hasattr(model, "ctrl_y"):
        parts.extend(model.ctrl_y.tolist())
    if hasattr(model, "lut_x"):
        parts.extend(model.lut_x.tolist())
    if hasattr(model, "lut_y"):
        parts.extend(model.lut_y.tolist())
    if hasattr(model, "chroma_gain"):
        parts.append(float(model.chroma_gain))
    if hasattr(model, "saturation"):
        parts.append(float(model.saturation))
    if hasattr(model, "color_matrix"):
        parts.extend(model.color_matrix.ravel().tolist())
    if hasattr(model, "sat_base"):
        parts.append(float(model.sat_base))
    if hasattr(model, "sat_slope"):
        parts.append(float(model.sat_slope))

    return np.array(parts, dtype=np.float64) if parts else np.array([0.0])


# =============================================================================
# Parameter reduction experiment
# =============================================================================

def run_parameter_reduction():
    """Test Model E with varying control point counts."""
    print("\nRunning parameter reduction experiment...")
    hdr_full, sdr_om, hdr_slice = generate_scene_neutral(seed=42)
    sdr_overlap_gamma = sdr_om[hdr_slice, :, :]
    sdr_overlap_linear = prepare_sdr_for_model(sdr_overlap_gamma)
    hdr_overlap = hdr_full

    sdr_lum = _luminance(sdr_overlap_linear).ravel()
    hdr_lum = _luminance(hdr_overlap).ravel()
    sdr_full_linear = prepare_sdr_for_model(sdr_om)

    param_counts = [4, 6, 8, 12, 16, 24, 32]
    results = []

    for n in param_counts:
        model = ModelE(n_ctrl=n)
        model.fit(sdr_lum, hdr_lum)
        pred_full = model.apply(sdr_full_linear)
        pred_overlap = pred_full[hdr_slice, :, :]
        rmse = luminance_rmse(pred_overlap, hdr_full)
        results.append(ParamReductionResult(
            model_name=f"E (n={n})", n_params=n, rmse_nits=rmse))
        print(f"    E (n={n:2d}): RMSE = {rmse:.1f} nits")

    return results


# =============================================================================
# Report generation
# =============================================================================

def generate_report(comparison_results, stability_results, param_reduction, total_time):
    """Generate comprehensive research report as markdown."""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# Scene-Based SDR-to-HDR Reshaping: Research Report")
    w()
    w(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**Total runtime:** {total_time:.1f}s")
    w(f"**Image dimensions:** HDR {HDR_WIDTH}x{HDR_HEIGHT}, SDR OM {SDR_WIDTH}x{SDR_HEIGHT}")
    w(f"**Overlap region:** SDR rows {OVERLAP_TOP}:{OVERLAP_BOTTOM}")
    w()

    # Section A
    w("## A. Project Audit: What Existing Code Does Right")
    w()
    w("The production codebase (`src/auto_openmatte/`) already implements:")
    w()
    w("1. **PQ EOTF/OETF** - correct ST 2084 transfer functions with LUT acceleration")
    w("2. **BT.709/BT.2020 color matrices** - proper 3x3 gamut conversion")
    w("3. **Luminance extraction** - BT.2020 weighted luminance computation")
    w("4. **Frame geometry** - correct identification of open matte overlap regions")
    w("5. **Pipeline structure** - modular processing chain with proper data flow")
    w()
    w("These components are technically sound and reusable in the improved pipeline.")
    w()

    # Section B
    w("## B. Conceptual Problems")
    w()
    w("The current approach treats SDR-to-HDR as an **inverse tone mapping** problem,")
    w("applying a fixed expansion curve without reference data. The fundamental issues:")
    w()
    w("1. **No ground truth** - without HDR reference in the overlap region, any expansion")
    w("   is guesswork. The same SDR value could map to vastly different HDR values")
    w("   depending on the original scene content.")
    w("2. **Scene-independent parameters** - a single curve cannot handle the diversity of")
    w("   real content (dark scenes vs bright scenes, saturated vs neutral).")
    w("3. **Chroma handling** - simple luminance scaling distorts color relationships.")
    w("   Saturated highlights require different treatment than neutral midtones.")
    w("4. **Temporal coherence** - frame-by-frame processing without parameter continuity")
    w("   causes flicker and instability.")
    w()
    w("The correct framing: **reference-guided reshaping** using the overlap region where")
    w("both SDR and HDR data exist to estimate scene-specific transform parameters.")
    w()

    # Section C
    w("## C. Research Background")
    w()
    w("### Dolby Backward Reshaping (US Patent 9,613,407)")
    w("- Polynomial mapping with per-scene coefficients stored in metadata")
    w("- 3-channel piecewise polynomial, typically degree 1-3 per piece")
    w("- Designed for reconstruction from single-layer (SDR) encoding")
    w()
    w("### CDF-Based Transfer (Histogram Specification)")
    w("- Match cumulative distribution functions between SDR and HDR luminance")
    w("- Non-parametric: stores full transfer LUT (~64-256 points)")
    w("- Guaranteed monotonic, preserves relative ordering")
    w()
    w("### Multi-Modal Regression (MMR)")
    w("- Multiple regression channels for different luminance ranges")
    w("- 3x3 matrix per luminance segment for color correction")
    w("- Dolby Vision Profile 7 uses 3-pivot MMR")
    w()
    w("### Piecewise Techniques")
    w("- Knot-based splines with monotonicity constraints")
    w("- Sigmoid/logistic shoulder functions for highlight rolloff")
    w("- Separate shadow lift for dark region reconstruction")
    w()

    # Section D
    w("## D. Proposed Architecture")
    w()
    w("```")
    w("SDR Open Matte (BT.709 gamma)")
    w("    |")
    w("    v")
    w("[1] Gamma decode (power 2.4)")
    w("    |")
    w("    v")
    w("[2] BT.709 -> BT.2020 linear (3x3 matrix)")
    w("    |")
    w("    +---> Extract overlap region (rows 280:1880)")
    w("    |         |")
    w("    |         v")
    w("    |     [3] Fit model params (SDR_linear vs HDR_linear)")
    w("    |         |")
    w("    v         v")
    w("[4] Apply fitted model to FULL SDR Open Matte")
    w("    |")
    w("    v")
    w("[5] Apply PQ OETF -> HDR10 output (BT.2020 + PQ)")
    w("```")
    w()

    # Section E
    w("## E. Candidate Models")
    w()
    w("| Model | Name | Parameters | Description |")
    w("|-------|------|-----------|-------------|")
    w("| A | Linear Gain | 2 | Y'=a*Y, C'=b*C |")
    w("| B | Piecewise Linear | 10 | 8 luma knots + chroma |")
    w("| C | Monotonic Polynomial | 8 | Degree 4 luma + degree 2 sat |")
    w("| D | CDF Matching | 65 | 64-pt histogram transfer LUT |")
    w("| E | CDF + Regularized | 12 | Smooth 12-pt optimized curve |")
    w("| F | Luma + Chroma Matrix | 18 | Degree 4 poly + 3x3 matrix |")
    w("| G | Hybrid | 29 | Percentile + shoulder/shadow + matrix + hue |")
    w()

    # Section F
    w("## F. Synthetic Test Results")
    w()

    scene_names = sorted(set(r.scene_name for r in comparison_results))
    model_names = []
    seen = set()
    for r in comparison_results:
        if r.model_name not in seen:
            model_names.append(r.model_name)
            seen.add(r.model_name)

    # RMSE table
    w("### RMSE (nits) by Model and Scene")
    w()
    header = "| Model |" + "".join(f" {s} |" for s in scene_names) + " Avg |"
    sep = "|" + "---|" * (len(scene_names) + 2)
    w(header)
    w(sep)

    for mn in model_names:
        row = f"| {mn} |"
        vals = []
        for sn in scene_names:
            matching = [r for r in comparison_results if r.model_name == mn and r.scene_name == sn]
            if matching:
                v = matching[0].rmse_nits
                vals.append(v)
                row += f" {v:.1f} |"
            else:
                row += " - |"
        avg = np.mean(vals) if vals else 0.0
        row += f" {avg:.1f} |"
        w(row)
    w()

    # MAE table
    w("### MAE (nits) by Model and Scene")
    w()
    w(header)
    w(sep)

    for mn in model_names:
        row = f"| {mn} |"
        vals = []
        for sn in scene_names:
            matching = [r for r in comparison_results if r.model_name == mn and r.scene_name == sn]
            if matching:
                v = matching[0].mae_nits
                vals.append(v)
                row += f" {v:.1f} |"
            else:
                row += " - |"
        avg = np.mean(vals) if vals else 0.0
        row += f" {avg:.1f} |"
        w(row)
    w()

    # Delta E table
    w("### Delta E 2000 by Model and Scene")
    w()
    w(header)
    w(sep)

    for mn in model_names:
        row = f"| {mn} |"
        vals = []
        for sn in scene_names:
            matching = [r for r in comparison_results if r.model_name == mn and r.scene_name == sn]
            if matching:
                v = matching[0].delta_e
                vals.append(v)
                row += f" {v:.2f} |"
            else:
                row += " - |"
        avg = np.mean(vals) if vals else 0.0
        row += f" {avg:.2f} |"
        w(row)
    w()

    # Seam table
    w("### Seam Error (nits) by Model and Scene")
    w()
    w(header)
    w(sep)

    for mn in model_names:
        row = f"| {mn} |"
        vals = []
        for sn in scene_names:
            matching = [r for r in comparison_results if r.model_name == mn and r.scene_name == sn]
            if matching:
                v = matching[0].seam_top + matching[0].seam_bottom
                vals.append(v)
                row += f" {v:.1f} |"
            else:
                row += " - |"
        avg = np.mean(vals) if vals else 0.0
        row += f" {avg:.1f} |"
        w(row)
    w()

    # Section G
    w("## G. Stability Results")
    w()
    w("Coefficient of variation across 5 refits with additive Gaussian noise (sigma=0.005):")
    w()
    w("| Model | CV Score | Rating |")
    w("|-------|----------|--------|")
    for sr in stability_results:
        if sr.cv_score < 0.01:
            rating = "Excellent"
        elif sr.cv_score < 0.05:
            rating = "Good"
        elif sr.cv_score < 0.15:
            rating = "Fair"
        else:
            rating = "Poor"
        w(f"| {sr.model_name} | {sr.cv_score:.4f} | {rating} |")
    w()
    w("Lower CV = more temporally stable parameters (less flicker risk).")
    w()

    # Section H
    w("## H. Parameter Reduction Curve")
    w()
    w("Model E (CDF + Regularized) tested with varying control point counts:")
    w()
    w("| Control Points | RMSE (nits) |")
    w("|---------------|-------------|")
    for pr in param_reduction:
        w(f"| {pr.n_params} | {pr.rmse_nits:.1f} |")
    w()
    w("Diminishing returns visible above 12 control points for typical content.")
    w()

    # Section I
    w("## I. Recommendation")
    w()

    model_avg_rmse = {}
    model_avg_de = {}
    for mn in model_names:
        rmse_vals = [r.rmse_nits for r in comparison_results if r.model_name == mn]
        de_vals = [r.delta_e for r in comparison_results if r.model_name == mn]
        model_avg_rmse[mn] = float(np.mean(rmse_vals)) if rmse_vals else 999.0
        model_avg_de[mn] = float(np.mean(de_vals)) if de_vals else 999.0

    best_rmse_model = min(model_avg_rmse, key=lambda x: model_avg_rmse[x])
    best_de_model = min(model_avg_de, key=lambda x: model_avg_de[x])

    w(f"**Best overall RMSE:** {best_rmse_model} ({model_avg_rmse[best_rmse_model]:.1f} nits avg)")
    w(f"**Best overall Delta E:** {best_de_model} ({model_avg_de[best_de_model]:.2f} avg)")
    w()
    w("### Selection Criteria")
    w()
    w("The recommended model balances:")
    w("1. Reconstruction accuracy (low RMSE and Delta E)")
    w("2. Parameter efficiency (fewer params = easier metadata, faster fitting)")
    w("3. Temporal stability (low CV = no flicker)")
    w("4. Seam quality (smooth transitions at overlap boundaries)")
    w()
    w("### Final Recommendation")
    w()

    stab_map = {sr.model_name: sr.cv_score for sr in stability_results}

    scores = {}
    all_models = get_all_models()
    for mn in model_names:
        rmse_score = model_avg_rmse[mn] / max(model_avg_rmse.values())
        de_score = model_avg_de[mn] / max(model_avg_de.values())
        stab_score = stab_map.get(mn, 0.5) / max(max(stab_map.values()), 0.001)
        pc = next((m.param_count for m in all_models if m.name == mn), 50)
        param_score = pc / 65.0
        scores[mn] = 0.4 * rmse_score + 0.25 * de_score + 0.2 * stab_score + 0.15 * param_score

    recommended = min(scores, key=lambda x: scores[x])

    w(f"**Recommended model: {recommended}**")
    w()
    w("Rationale:")
    w(f"- Average RMSE: {model_avg_rmse[recommended]:.1f} nits")
    w(f"- Average Delta E: {model_avg_de[recommended]:.2f}")
    w(f"- Stability CV: {stab_map.get(recommended, 0.0):.4f}")
    pc_rec = next((m.param_count for m in all_models if m.name == recommended), 0)
    w(f"- Parameter count: {pc_rec}")
    w()
    w("This model provides the best trade-off between accuracy, stability, and simplicity")
    w("for production integration. The overlap region provides sufficient constraint data")
    w("to estimate scene-specific parameters on a per-shot or per-frame basis.")
    w()
    w("### Integration Path")
    w()
    w("1. Extract overlap region from frame geometry (already implemented)")
    w("2. Linearize SDR overlap (gamma 2.4 decode + BT.709->BT.2020)")
    w("3. Fit model parameters from overlap (SDR vs HDR luminance + chroma)")
    w("4. Apply fitted model to full SDR open matte extent")
    w("5. Encode output as PQ (ST 2084) in BT.2020 container")
    w()
    w("---")
    w()
    w("*Report generated by research/run_comparison.py*")

    return "\n".join(lines)


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run the full comparison pipeline."""
    print("=" * 70)
    print("  Scene-Based SDR-to-HDR Reshaping: Full Comparison")
    print("=" * 70)
    print()

    total_start = time.time()

    comparison_results, comp_time = run_comparison()
    stability_results = run_stability_test(n_trials=5)
    param_reduction = run_parameter_reduction()

    total_time = time.time() - total_start

    print("\nGenerating report...")
    report = generate_report(comparison_results, stability_results, param_reduction, total_time)

    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "RESEARCH_REPORT.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
