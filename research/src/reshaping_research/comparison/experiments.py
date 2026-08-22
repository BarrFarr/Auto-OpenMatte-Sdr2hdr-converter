"""Additional experiments beyond basic model comparison.

Implements specialized experiments to answer key research questions:
- Strategy A vs B (all pixels vs representative subsampling)
- Luma-only vs full RGB regression
- Hue correction necessity
- Temporal stability under noise
- Robustness to sample count
- Parameter reduction sweep
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..models import get_all_models
from ..models.model_b_piecewise import ModelBPiecewise
from ..models.model_f_luma_chroma_regression import ModelFLumaChromaRegression
from ..models.model_g_hybrid import ModelGHybrid
from ..synthetic.scene_generator import SceneType, generate_scene
from ..synthetic.open_matte_simulator import create_open_matte_pair
from ..utils.transfer_functions import bt1886_eotf
from ..utils.color_spaces import (
    BT2020_LUMA,
    bt709_to_bt2020,
    compute_luminance,
)
from ..metrics import luminance_rmse, parameter_variance, flicker_risk


@dataclass
class StrategyResult:
    """Result comparing Strategy A (all pixels) vs Strategy B (subsampled)."""
    model_name: str
    scene_type: str
    strategy_a_rmse: float
    strategy_b_rmse: float
    strategy_a_fit_time_ms: float
    strategy_b_fit_time_ms: float
    speedup_ratio: float


@dataclass
class LumaVsRgbResult:
    """Result comparing luma-only vs full RGB regression."""
    scene_type: str
    model_b_rmse: float
    model_f_rmse: float
    quality_ratio: float


@dataclass
class HueNecessityResult:
    """Result testing hue correction necessity."""
    scene_type: str
    with_hue_rmse: float
    without_hue_rmse: float
    hue_shift_mean_degrees: float
    hue_improvement_pct: float


@dataclass
class TemporalStabilityResult:
    """Result from temporal stability test."""
    model_name: str
    scene_type: str
    param_cv_mean: float
    param_cv_max: float
    output_rmse_std: float
    flicker_risk_score: float


@dataclass
class RobustnessResult:
    """Result from robustness/sample count test."""
    model_name: str
    scene_type: str
    sample_counts: List[int] = field(default_factory=list)
    rmse_values: List[float] = field(default_factory=list)
    converged_at: int = 0


@dataclass
class ParameterReductionResult:
    """Result from parameter reduction sweep."""
    model_name: str
    scene_type: str
    configs: List[str] = field(default_factory=list)
    param_counts: List[int] = field(default_factory=list)
    rmse_values: List[float] = field(default_factory=list)
    best_config: str = ""
    best_quality_per_param: float = 0.0


@dataclass
class ExperimentResults:
    """All experiment results combined."""
    strategy_results: List[StrategyResult] = field(default_factory=list)
    luma_vs_rgb_results: List[LumaVsRgbResult] = field(default_factory=list)
    hue_results: List[HueNecessityResult] = field(default_factory=list)
    temporal_results: List[TemporalStabilityResult] = field(default_factory=list)
    robustness_results: List[RobustnessResult] = field(default_factory=list)
    param_reduction_results: List[ParameterReductionResult] = field(default_factory=list)
    total_time_s: float = 0.0


def run_all_experiments(
    width: int = 256,
    height: int = 256,
    peak_nits: float = 4000.0,
    seed: int = 42,
    verbose: bool = False,
) -> ExperimentResults:
    """Run all additional experiments."""
    start = time.time()
    results = ExperimentResults()

    if verbose:
        print("Running strategy comparison...")
    results.strategy_results = strategy_comparison(
        width=width, height=height, peak_nits=peak_nits, seed=seed
    )

    if verbose:
        print("Running luma vs full RGB...")
    results.luma_vs_rgb_results = luma_vs_full_rgb(
        width=width, height=height, peak_nits=peak_nits, seed=seed
    )

    if verbose:
        print("Running hue necessity test...")
    results.hue_results = hue_necessity_test(
        width=width, height=height, peak_nits=peak_nits, seed=seed
    )

    if verbose:
        print("Running temporal stability test...")
    results.temporal_results = temporal_stability_test(
        width=width, height=height, peak_nits=peak_nits, seed=seed
    )

    if verbose:
        print("Running robustness test...")
    results.robustness_results = robustness_test(
        width=width, height=height, peak_nits=peak_nits, seed=seed
    )

    if verbose:
        print("Running parameter reduction sweep...")
    results.param_reduction_results = parameter_reduction_sweep(
        width=width, height=height, peak_nits=peak_nits, seed=seed
    )

    results.total_time_s = time.time() - start
    return results


def strategy_comparison(
    width: int = 256, height: int = 256, peak_nits: float = 4000.0, seed: int = 42,
) -> List[StrategyResult]:
    """Compare Strategy A (all overlap pixels) vs Strategy B (subsampled representative)."""
    results = []
    test_scenes = list(SceneType)

    for scene_idx, scene_type in enumerate(test_scenes):
        scene_seed = seed + scene_idx * 100
        hdr_scene = generate_scene(scene_type, height=height, width=width, seed=scene_seed)
        pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=scene_seed)
        sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap(pair, peak_nits)
        sdr_full = _linearize_full(pair.sdr_wide)

        # Strategy A: all pixels
        model_a_strat = ModelGHybrid()
        t0 = time.time()
        model_a_strat.fit(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)
        time_a = (time.time() - t0) * 1000.0
        out_a = model_a_strat.apply(sdr_full)

        # Strategy B: subsampled representative pixels
        sub_sdr_lum, sub_hdr_lum, sub_sdr_chroma, sub_hdr_chroma = (
            _subsample_representative(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)
        )
        model_b_strat = ModelGHybrid()
        t0 = time.time()
        model_b_strat.fit(sub_sdr_lum, sub_hdr_lum, sub_sdr_chroma, sub_hdr_chroma)
        time_b = (time.time() - t0) * 1000.0
        out_b = model_b_strat.apply(sdr_full)

        # Compare quality
        y_start = pair.hdr_offset_top
        y_end = pair.hdr_offset_bottom
        ref_lum = compute_luminance(pair.hdr_center, BT2020_LUMA)

        rmse_a = luminance_rmse(
            compute_luminance(out_a[y_start:y_end] * peak_nits, BT2020_LUMA), ref_lum
        )
        rmse_b = luminance_rmse(
            compute_luminance(out_b[y_start:y_end] * peak_nits, BT2020_LUMA), ref_lum
        )

        speedup = time_a / max(time_b, 0.01)
        results.append(StrategyResult(
            model_name="Model G: Hybrid",
            scene_type=scene_type.value,
            strategy_a_rmse=rmse_a,
            strategy_b_rmse=rmse_b,
            strategy_a_fit_time_ms=time_a,
            strategy_b_fit_time_ms=time_b,
            speedup_ratio=speedup,
        ))

    return results


def luma_vs_full_rgb(
    width: int = 256, height: int = 256, peak_nits: float = 4000.0, seed: int = 42,
) -> List[LumaVsRgbResult]:
    """Compare Model B (luma + simple chroma) vs Model F (full 3x3 matrix)."""
    results = []
    for scene_idx, scene_type in enumerate(SceneType):
        scene_seed = seed + scene_idx * 100
        hdr_scene = generate_scene(scene_type, height=height, width=width, seed=scene_seed)
        pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=scene_seed)
        sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap(pair, peak_nits)
        sdr_full = _linearize_full(pair.sdr_wide)

        model_b = ModelBPiecewise()
        model_b.fit(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)
        out_b = model_b.apply(sdr_full)

        model_f = ModelFLumaChromaRegression()
        model_f.fit(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)
        out_f = model_f.apply(sdr_full)

        y_start = pair.hdr_offset_top
        y_end = pair.hdr_offset_bottom
        ref_lum = compute_luminance(pair.hdr_center, BT2020_LUMA)

        rmse_b = luminance_rmse(
            compute_luminance(out_b[y_start:y_end] * peak_nits, BT2020_LUMA), ref_lum
        )
        rmse_f = luminance_rmse(
            compute_luminance(out_f[y_start:y_end] * peak_nits, BT2020_LUMA), ref_lum
        )

        quality_ratio = rmse_b / max(rmse_f, 0.001)
        results.append(LumaVsRgbResult(
            scene_type=scene_type.value,
            model_b_rmse=rmse_b,
            model_f_rmse=rmse_f,
            quality_ratio=quality_ratio,
        ))

    return results


def hue_necessity_test(
    width: int = 256, height: int = 256, peak_nits: float = 4000.0, seed: int = 42,
) -> List[HueNecessityResult]:
    """Compare Model G with hue correction enabled vs disabled."""
    results = []
    for scene_idx, scene_type in enumerate(SceneType):
        scene_seed = seed + scene_idx * 100
        hdr_scene = generate_scene(scene_type, height=height, width=width, seed=scene_seed)
        pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=scene_seed)
        sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap(pair, peak_nits)
        sdr_full = _linearize_full(pair.sdr_wide)

        # With hue
        model_with = ModelGHybrid(hue_constraint=0.1)
        model_with.fit(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)
        out_with = model_with.apply(sdr_full)
        hue_angles = np.asarray(model_with.params.get("hue_angles", np.zeros(3)))
        hue_shift_mean = float(np.mean(np.abs(np.degrees(hue_angles))))

        # Without hue
        model_without = ModelGHybrid(hue_constraint=0.0)
        model_without.fit(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)
        out_without = model_without.apply(sdr_full)

        y_start = pair.hdr_offset_top
        y_end = pair.hdr_offset_bottom
        ref_lum = compute_luminance(pair.hdr_center, BT2020_LUMA)

        rmse_with = luminance_rmse(
            compute_luminance(out_with[y_start:y_end] * peak_nits, BT2020_LUMA), ref_lum
        )
        rmse_without = luminance_rmse(
            compute_luminance(out_without[y_start:y_end] * peak_nits, BT2020_LUMA), ref_lum
        )

        improvement = (rmse_without - rmse_with) / max(rmse_without, 0.001) * 100.0
        results.append(HueNecessityResult(
            scene_type=scene_type.value,
            with_hue_rmse=rmse_with,
            without_hue_rmse=rmse_without,
            hue_shift_mean_degrees=hue_shift_mean,
            hue_improvement_pct=improvement,
        ))

    return results


def temporal_stability_test(
    width: int = 256, height: int = 256, peak_nits: float = 4000.0,
    seed: int = 42, n_frames: int = 10, noise_sigma: float = 0.01,
) -> List[TemporalStabilityResult]:
    """Test temporal stability by fitting same scene with added noise."""
    results = []
    test_models = [ModelGHybrid(), ModelFLumaChromaRegression(), ModelBPiecewise()]
    test_scenes = [SceneType.NEUTRAL, SceneType.MIXED, SceneType.DIFFICULT]

    for scene_idx, scene_type in enumerate(test_scenes):
        scene_seed = seed + scene_idx * 100
        hdr_scene = generate_scene(scene_type, height=height, width=width, seed=scene_seed)
        pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=scene_seed)

        for model in test_models:
            frame_params = []
            frame_rmses = []
            rng = np.random.default_rng(seed + scene_idx)

            for frame_i in range(n_frames):
                noise = rng.normal(0.0, noise_sigma, pair.sdr_wide.shape)
                noisy_sdr = np.clip(pair.sdr_wide + noise, 0.0, 1.0)

                y_start = pair.hdr_offset_top
                y_end = pair.hdr_offset_bottom
                sdr_overlap_gamma = noisy_sdr[y_start:y_end]
                sdr_overlap_linear = bt709_to_bt2020(bt1886_eotf(sdr_overlap_gamma))
                hdr_overlap_norm = pair.hdr_center / peak_nits

                sdr_flat = sdr_overlap_linear.reshape(-1, 3)
                hdr_flat = hdr_overlap_norm.reshape(-1, 3)
                sdr_lum_f = compute_luminance(sdr_flat, BT2020_LUMA)
                hdr_lum_f = compute_luminance(hdr_flat, BT2020_LUMA)

                if isinstance(model, ModelGHybrid):
                    m = ModelGHybrid()
                elif isinstance(model, ModelFLumaChromaRegression):
                    m = ModelFLumaChromaRegression()
                else:
                    m = ModelBPiecewise()

                m.fit(sdr_lum_f, hdr_lum_f, sdr_flat, hdr_flat)

                params = m.params
                if "luma_y" in params:
                    luma_y = np.asarray(params["luma_y"])
                    mid_val = float(luma_y[len(luma_y) // 2])
                elif "y_knots" in params:
                    y_knots = np.asarray(params["y_knots"])
                    mid_val = float(y_knots[len(y_knots) // 2])
                elif "a" in params:
                    mid_val = float(params["a"])
                else:
                    mid_val = 1.0

                frame_params.append(mid_val)

                sdr_full = _linearize_full(noisy_sdr)
                out = m.apply(sdr_full)
                out_center_nits = out[y_start:y_end] * peak_nits
                ref_lum = compute_luminance(pair.hdr_center, BT2020_LUMA)
                out_lum = compute_luminance(out_center_nits, BT2020_LUMA)
                frame_rmses.append(luminance_rmse(out_lum, ref_lum))

            cv = parameter_variance(frame_params)
            rmse_std = float(np.std(frame_rmses))
            fr = flicker_risk(frame_params)

            results.append(TemporalStabilityResult(
                model_name=model.name(),
                scene_type=scene_type.value,
                param_cv_mean=cv,
                param_cv_max=cv,
                output_rmse_std=rmse_std,
                flicker_risk_score=fr.flicker_risk,
            ))

    return results


def robustness_test(
    width: int = 256, height: int = 256, peak_nits: float = 4000.0, seed: int = 42,
) -> List[RobustnessResult]:
    """Vary number of sample pixels used for fitting."""
    sample_counts = [100, 1000, 5000, 20000, 100000]
    results = []
    test_scenes = [SceneType.NEUTRAL, SceneType.MIXED, SceneType.DIFFICULT]

    for scene_idx, scene_type in enumerate(test_scenes):
        scene_seed = seed + scene_idx * 100
        hdr_scene = generate_scene(scene_type, height=height, width=width, seed=scene_seed)
        pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=scene_seed)
        sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap(pair, peak_nits)
        sdr_full = _linearize_full(pair.sdr_wide)
        total_pixels = len(sdr_lum)

        y_start = pair.hdr_offset_top
        y_end = pair.hdr_offset_bottom
        ref_lum = compute_luminance(pair.hdr_center, BT2020_LUMA)

        rmse_values = []
        valid_counts = []
        rng = np.random.default_rng(seed)

        for n_samples in sample_counts:
            if n_samples >= total_pixels:
                sub_sdr_lum = sdr_lum
                sub_hdr_lum = hdr_lum
                sub_sdr_chroma = sdr_overlap
                sub_hdr_chroma = hdr_overlap
            else:
                idx = rng.choice(total_pixels, size=n_samples, replace=False)
                sub_sdr_lum = sdr_lum[idx]
                sub_hdr_lum = hdr_lum[idx]
                sub_sdr_chroma = sdr_overlap[idx]
                sub_hdr_chroma = hdr_overlap[idx]

            model = ModelGHybrid()
            model.fit(sub_sdr_lum, sub_hdr_lum, sub_sdr_chroma, sub_hdr_chroma)
            out = model.apply(sdr_full)
            out_center = out[y_start:y_end] * peak_nits
            out_lum = compute_luminance(out_center, BT2020_LUMA)
            rmse = luminance_rmse(out_lum, ref_lum)
            rmse_values.append(rmse)
            valid_counts.append(min(n_samples, total_pixels))

        converged_at = valid_counts[-1]
        for i in range(1, len(rmse_values)):
            if rmse_values[i - 1] > 0:
                improvement = (rmse_values[i - 1] - rmse_values[i]) / rmse_values[i - 1]
                if improvement < 0.05:
                    converged_at = valid_counts[i - 1]
                    break

        results.append(RobustnessResult(
            model_name="Model G: Hybrid",
            scene_type=scene_type.value,
            sample_counts=valid_counts,
            rmse_values=rmse_values,
            converged_at=converged_at,
        ))

    return results


def parameter_reduction_sweep(
    width: int = 256, height: int = 256, peak_nits: float = 4000.0, seed: int = 42,
) -> List[ParameterReductionResult]:
    """Systematically reduce parameters from Model G and measure quality."""
    configs = [
        ("Full G (29p)", {"num_luma_points": 10, "hue_constraint": 0.1, "include_shadow": True}),
        ("No hue (26p)", {"num_luma_points": 10, "hue_constraint": 0.0, "include_shadow": True}),
        ("No shadow (27p)", {"num_luma_points": 10, "hue_constraint": 0.1, "include_shadow": False}),
        ("6 luma pts (25p)", {"num_luma_points": 6, "hue_constraint": 0.1, "include_shadow": True}),
        ("No hue+shadow (24p)", {"num_luma_points": 10, "hue_constraint": 0.0, "include_shadow": False}),
        ("6 luma no hue (22p)", {"num_luma_points": 6, "hue_constraint": 0.0, "include_shadow": True}),
        ("Minimal (18p)", {"num_luma_points": 6, "hue_constraint": 0.0, "include_shadow": False}),
    ]

    results = []
    test_scenes = [SceneType.NEUTRAL, SceneType.MIXED, SceneType.DIFFICULT]

    for scene_idx, scene_type in enumerate(test_scenes):
        scene_seed = seed + scene_idx * 100
        hdr_scene = generate_scene(scene_type, height=height, width=width, seed=scene_seed)
        pair = create_open_matte_pair(hdr_scene, extension_rows=48, seed=scene_seed)
        sdr_overlap, hdr_overlap, sdr_lum, hdr_lum = _prepare_overlap(pair, peak_nits)
        sdr_full = _linearize_full(pair.sdr_wide)

        y_start = pair.hdr_offset_top
        y_end = pair.hdr_offset_bottom
        ref_lum = compute_luminance(pair.hdr_center, BT2020_LUMA)

        config_names = []
        param_counts = []
        rmse_values = []

        for config_name, config in configs:
            n_luma = config["num_luma_points"]
            hue_c = config["hue_constraint"]

            model = ModelGHybrid(num_luma_points=n_luma, hue_constraint=hue_c)

            actual_params = n_luma + 3 + 2 + 9  # luma + highlight + sat + matrix
            if config.get("include_shadow", True):
                actual_params += 2
            if hue_c > 0:
                actual_params += 3

            model.fit(sdr_lum, hdr_lum, sdr_overlap, hdr_overlap)

            if not config.get("include_shadow", True):
                model.params["shadow_threshold"] = 0.0
                model.params["shadow_gain"] = 1.0

            out = model.apply(sdr_full)
            out_center = out[y_start:y_end] * peak_nits
            out_lum = compute_luminance(out_center, BT2020_LUMA)
            rmse = luminance_rmse(out_lum, ref_lum)

            config_names.append(config_name)
            param_counts.append(actual_params)
            rmse_values.append(rmse)

        quality_per_param = [
            (1.0 / max(r, 0.001)) / max(p, 1) for r, p in zip(rmse_values, param_counts)
        ]
        best_idx = int(np.argmax(quality_per_param))

        results.append(ParameterReductionResult(
            model_name="Model G: Hybrid",
            scene_type=scene_type.value,
            configs=config_names,
            param_counts=param_counts,
            rmse_values=rmse_values,
            best_config=config_names[best_idx],
            best_quality_per_param=quality_per_param[best_idx],
        ))

    return results


def _prepare_overlap(pair, peak_nits):
    """Prepare overlap data for fitting."""
    y_start = pair.hdr_offset_top
    y_end = pair.hdr_offset_bottom
    sdr_overlap_gamma = pair.sdr_wide[y_start:y_end, :, :]
    sdr_overlap_linear = bt709_to_bt2020(bt1886_eotf(sdr_overlap_gamma))
    hdr_overlap_norm = pair.hdr_center / peak_nits
    sdr_flat = sdr_overlap_linear.reshape(-1, 3)
    hdr_flat = hdr_overlap_norm.reshape(-1, 3)
    sdr_lum = compute_luminance(sdr_flat, BT2020_LUMA)
    hdr_lum = compute_luminance(hdr_flat, BT2020_LUMA)
    return sdr_flat, hdr_flat, sdr_lum, hdr_lum


def _linearize_full(sdr_wide):
    """Linearize and convert full SDR frame."""
    return bt709_to_bt2020(bt1886_eotf(sdr_wide))


def _subsample_representative(sdr_lum, hdr_lum, sdr_chroma, hdr_chroma):
    """Subsample representative pixels (Strategy B)."""
    n = len(sdr_lum)
    sorted_idx = np.argsort(sdr_lum)

    n_dark = max(int(n * 0.10), 10)
    dark_idx = sorted_idx[:n_dark]

    n_bright = max(int(n * 0.10), 10)
    bright_idx = sorted_idx[-n_bright:]

    mid_start = int(n * 0.50)
    mid_end = int(n * 0.70)
    mid_idx = sorted_idx[mid_start:mid_end]

    sdr_y = compute_luminance(sdr_chroma, BT2020_LUMA)
    chroma_mag = np.sqrt(np.sum((sdr_chroma - sdr_y[:, np.newaxis]) ** 2, axis=-1))
    neutral_threshold = np.percentile(chroma_mag, 20)
    neutral_mask = chroma_mag < neutral_threshold
    neutral_idx = np.where(neutral_mask)[0]
    if len(neutral_idx) > n_dark:
        rng = np.random.default_rng(42)
        neutral_idx = rng.choice(neutral_idx, size=n_dark, replace=False)

    all_idx = np.unique(np.concatenate([dark_idx, bright_idx, mid_idx, neutral_idx]))
    return sdr_lum[all_idx], hdr_lum[all_idx], sdr_chroma[all_idx], hdr_chroma[all_idx]
