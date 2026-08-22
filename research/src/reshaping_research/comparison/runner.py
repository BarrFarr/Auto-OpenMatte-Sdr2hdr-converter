"""Comparison runner: run all models on all scenes, compute all metrics.

Orchestrates the full comparison by:
1. Generating each synthetic scene
2. Creating open matte pairs
3. Fitting each model on overlap data
4. Applying model to full frame
5. Computing all quality metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..models import get_all_models
from ..synthetic.scene_generator import SceneType, generate_scene
from ..synthetic.open_matte_simulator import create_open_matte_pair
from ..utils.transfer_functions import bt1886_eotf
from ..utils.color_spaces import (
    BT2020_LUMA,
    bt709_to_bt2020,
    compute_luminance,
    linear_to_lab_d65,
    bt2020_to_bt709,
)
from ..metrics import (
    luminance_mae,
    luminance_rmse,
    delta_e_2000,
    delta_e_ictcp,
    gradient_continuity,
    seam_error,
)


ALL_SCENE_TYPES = [
    SceneType.NEUTRAL,
    SceneType.HIGH_KEY,
    SceneType.LOW_KEY,
    SceneType.COLORFUL,
    SceneType.MIXED,
    SceneType.DIFFICULT,
]


@dataclass
class ModelResult:
    """Result for one (model, scene) pair."""

    model_name: str
    scene_type: str
    param_count: int
    fit_time_ms: float
    apply_time_ms: float

    # Center/overlap region metrics
    center_lum_mae_nits: float = 0.0
    center_lum_rmse_nits: float = 0.0
    center_delta_e_2000_mean: float = 0.0
    center_delta_e_ictcp_mean: float = 0.0

    # Seam metrics
    seam_top_gradient_diff: float = 0.0
    seam_bottom_gradient_diff: float = 0.0
    seam_top_continuity: float = 0.0
    seam_bottom_continuity: float = 0.0

    # Extension metrics
    extension_smoothness: float = 0.0
    extension_clipping_ratio: float = 0.0

    # Overall
    overall_quality_score: float = 0.0


@dataclass
class ComparisonResults:
    """Complete comparison results."""

    results: List[ModelResult] = field(default_factory=list)
    total_time_s: float = 0.0


def run_comparison(
    width: int = 256,
    height: int = 256,
    peak_nits: float = 4000.0,
    seed: int = 42,
    verbose: bool = False,
) -> ComparisonResults:
    """Run all models on all scenes and compute metrics.

    Args:
        width: Image width (keep small for speed).
        height: Image height (keep small for speed).
        peak_nits: Peak luminance of the HDR content.
        seed: Random seed for deterministic results.
        verbose: Print progress information.

    Returns:
        ComparisonResults with all model/scene results.
    """
    start_total = time.time()
    all_results: List[ModelResult] = []
    models = get_all_models()

    for scene_idx, scene_type in enumerate(ALL_SCENE_TYPES):
        scene_seed = seed + scene_idx * 100

        # Generate HDR scene
        hdr_scene = generate_scene(
            scene_type, height=height, width=width, seed=scene_seed
        )

        # Create open matte pair
        pair = create_open_matte_pair(
            hdr_scene, extension_rows=48, seed=scene_seed
        )

        # Prepare overlap data for fitting
        sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap_data(
            pair, peak_nits
        )

        # Prepare full SDR image (linearized, BT.2020)
        sdr_full_linear = _linearize_sdr_full(pair.sdr_wide)

        for model in models:
            if verbose:
                print(f"  Running {model.name()} on {scene_type.value}...")

            result = _evaluate_model(
                model=model,
                scene_type=scene_type,
                pair=pair,
                sdr_overlap=sdr_overlap,
                hdr_overlap=hdr_overlap,
                sdr_lum=sdr_lum,
                hdr_lum=hdr_lum,
                sdr_full_linear=sdr_full_linear,
                peak_nits=peak_nits,
            )
            all_results.append(result)

    total_time = time.time() - start_total
    return ComparisonResults(results=all_results, total_time_s=total_time)


def _prepare_overlap_data(
    pair: Any,
    peak_nits: float,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Prepare overlap region data for model fitting.

    Returns:
        sdr_overlap: (N, 3) linear BT.2020 SDR pixels from overlap
        hdr_overlap: (N, 3) linear BT.2020 HDR pixels from overlap (normalized)
        sdr_lum: (N,) SDR luminance
        hdr_lum: (N,) HDR luminance (normalized by peak_nits)
    """
    y_start = pair.hdr_offset_top
    y_end = pair.hdr_offset_bottom

    # SDR overlap: linearize (BT.1886) then convert BT.709 -> BT.2020
    sdr_overlap_gamma = pair.sdr_wide[y_start:y_end, :, :]
    sdr_overlap_linear_709 = bt1886_eotf(sdr_overlap_gamma)
    sdr_overlap_linear = bt709_to_bt2020(sdr_overlap_linear_709)

    # HDR overlap: already in linear BT.2020 nits, normalize by peak
    hdr_overlap_nits = pair.hdr_center
    hdr_overlap_norm = hdr_overlap_nits / peak_nits

    # Flatten to (N, 3)
    H, W, _ = sdr_overlap_linear.shape
    sdr_flat = sdr_overlap_linear.reshape(-1, 3)
    hdr_flat = hdr_overlap_norm.reshape(-1, 3)

    # Compute luminance
    sdr_lum = compute_luminance(sdr_flat, BT2020_LUMA)
    hdr_lum = compute_luminance(hdr_flat, BT2020_LUMA)

    return sdr_flat, hdr_flat, sdr_lum, hdr_lum


def _linearize_sdr_full(sdr_wide: NDArray) -> NDArray:
    """Linearize and convert entire SDR wide frame to BT.2020."""
    linear_709 = bt1886_eotf(sdr_wide)
    linear_2020 = bt709_to_bt2020(linear_709)
    return linear_2020


def _evaluate_model(
    model: Any,
    scene_type: SceneType,
    pair: Any,
    sdr_overlap: NDArray,
    hdr_overlap: NDArray,
    sdr_lum: NDArray,
    hdr_lum: NDArray,
    sdr_full_linear: NDArray,
    peak_nits: float,
) -> ModelResult:
    """Evaluate a single model on a single scene."""
    # Fit model
    t0 = time.time()
    model.fit(
        sdr_linear=sdr_lum,
        hdr_linear=hdr_lum,
        sdr_chroma=sdr_overlap,
        hdr_chroma=hdr_overlap,
    )
    fit_time = (time.time() - t0) * 1000.0

    # Apply model to full frame
    t0 = time.time()
    hdr_output = model.apply(sdr_full_linear)
    apply_time = (time.time() - t0) * 1000.0

    # Scale output to nits for metric computation
    hdr_output_nits = hdr_output * peak_nits

    # --- CENTER REGION METRICS ---
    y_start = pair.hdr_offset_top
    y_end = pair.hdr_offset_bottom
    center_output_nits = hdr_output_nits[y_start:y_end, :, :]
    center_ref_nits = pair.hdr_center

    # Luminance metrics
    out_lum = compute_luminance(center_output_nits, BT2020_LUMA)
    ref_lum = compute_luminance(center_ref_nits, BT2020_LUMA)

    center_mae = luminance_mae(out_lum, ref_lum)
    center_rmse = luminance_rmse(out_lum, ref_lum)

    # DeltaE 2000 (normalize to [0,1] for Lab conversion)
    max_nits = max(float(np.max(center_ref_nits)), 1.0)
    out_norm = np.clip(center_output_nits / max_nits, 0.0, 1.0)
    ref_norm = np.clip(center_ref_nits / max_nits, 0.0, 1.0)

    out_709 = bt2020_to_bt709(out_norm)
    ref_709 = bt2020_to_bt709(ref_norm)
    out_709 = np.clip(out_709, 0.0, 1.0)
    ref_709 = np.clip(ref_709, 0.0, 1.0)

    out_lab = linear_to_lab_d65(out_709)
    ref_lab = linear_to_lab_d65(ref_709)
    de2000 = delta_e_2000(out_lab, ref_lab)
    de2000_mean = float(np.mean(de2000))

    # DeltaE ICtCp
    de_ictcp = delta_e_ictcp(center_output_nits, center_ref_nits)
    de_ictcp_mean = float(np.mean(de_ictcp))

    # --- SEAM METRICS ---
    seam_top = pair.seam_top_row
    seam_bottom = pair.seam_bottom_row

    gc_top = gradient_continuity(hdr_output_nits, seam_top, band_width=8)
    gc_bottom = gradient_continuity(hdr_output_nits, seam_bottom, band_width=8)

    sm_top = seam_error(hdr_output_nits, seam_top, band_width=8)
    sm_bottom = seam_error(hdr_output_nits, seam_bottom, band_width=8)

    # --- EXTENSION METRICS ---
    ext_smoothness, ext_clipping = _compute_extension_metrics(
        hdr_output_nits, y_start, y_end, peak_nits
    )

    # --- OVERALL QUALITY SCORE ---
    lum_score = max(0.0, 1.0 - center_rmse / 500.0)
    color_score = max(0.0, 1.0 - de2000_mean / 20.0)
    seam_score = (sm_top.continuity_score + sm_bottom.continuity_score) / 2.0
    ext_score = ext_smoothness

    overall = 0.4 * lum_score + 0.3 * color_score + 0.2 * seam_score + 0.1 * ext_score

    return ModelResult(
        model_name=model.name(),
        scene_type=scene_type.value,
        param_count=model.param_count(),
        fit_time_ms=fit_time,
        apply_time_ms=apply_time,
        center_lum_mae_nits=center_mae,
        center_lum_rmse_nits=center_rmse,
        center_delta_e_2000_mean=de2000_mean,
        center_delta_e_ictcp_mean=de_ictcp_mean,
        seam_top_gradient_diff=gc_top,
        seam_bottom_gradient_diff=gc_bottom,
        seam_top_continuity=sm_top.continuity_score,
        seam_bottom_continuity=sm_bottom.continuity_score,
        extension_smoothness=ext_smoothness,
        extension_clipping_ratio=ext_clipping,
        overall_quality_score=overall,
    )


def _compute_extension_metrics(
    hdr_output_nits: NDArray,
    y_start: int,
    y_end: int,
    peak_nits: float,
) -> Tuple[float, float]:
    """Compute extension region metrics (gradient smoothness, clipping)."""
    H = hdr_output_nits.shape[0]

    top_ext = hdr_output_nits[:y_start, :, :]
    bottom_ext = hdr_output_nits[y_end:, :, :]

    if top_ext.size == 0 and bottom_ext.size == 0:
        return 1.0, 0.0

    smoothness_scores = []
    for ext in [top_ext, bottom_ext]:
        if ext.shape[0] < 3:
            smoothness_scores.append(1.0)
            continue
        lum = compute_luminance(ext, BT2020_LUMA)
        vert_grad = np.diff(lum, axis=0)
        grad_var = float(np.var(vert_grad))
        score = float(np.exp(-grad_var / 1000.0))
        smoothness_scores.append(score)

    smoothness = float(np.mean(smoothness_scores))

    all_ext = np.concatenate(
        [e for e in [top_ext, bottom_ext] if e.size > 0], axis=0
    )
    total_pixels = all_ext.size
    clipped = np.sum((all_ext <= 0.0) | (all_ext >= peak_nits * 0.99))
    clipping_ratio = float(clipped) / max(total_pixels, 1)

    return smoothness, clipping_ratio


def run_single(
    model_name: str,
    scene_type: SceneType,
    width: int = 256,
    height: int = 256,
    peak_nits: float = 4000.0,
    seed: int = 42,
) -> ModelResult:
    """Run a single model on a single scene (for testing)."""
    models = get_all_models()
    model = None
    for m in models:
        if model_name.lower() in m.name().lower():
            model = m
            break
    if model is None:
        model = models[0]

    hdr_scene = generate_scene(scene_type, height=height, width=width, seed=seed)
    pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=seed)
    sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap_data(pair, peak_nits)
    sdr_full_linear = _linearize_sdr_full(pair.sdr_wide)

    return _evaluate_model(
        model=model,
        scene_type=scene_type,
        pair=pair,
        sdr_overlap=sdr_overlap,
        hdr_overlap=hdr_overlap,
        sdr_lum=sdr_lum,
        hdr_lum=hdr_lum,
        sdr_full_linear=sdr_full_linear,
        peak_nits=peak_nits,
    )
