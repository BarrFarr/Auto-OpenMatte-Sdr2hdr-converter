#!/usr/bin/env python3
"""Phase 16 — per-actual-shot bounded low-frequency ICtCp residual field.

This runner is deliberately limited to the confirmed Matrix-08 partition. It
opens the existing 186-frame cache exactly once, obtains strict read-only views
for each actual shot, and refuses any cross-shot fit. A is the historical frozen
chroma-field/saturation baseline with a newly measured per-shot frozen boundary
gain. B is A plus the approved additive degree-2 Ct/Cp residual.

No LUT, MMR, temporal state, AI, semantic mask, parameter sweep, or additional
material is available through this program.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
TOOLS = WORKSPACE / "tools" / "openmatte_hdr"
RESEARCH_SRC = WORKSPACE / "research" / "src"
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

from reshaping_research.metrics.color_metrics import delta_e_2000  # noqa: E402
from reshaping_research.utils.color_spaces import BT2020_TO_BT709, linear_to_lab_d65  # noqa: E402


class Phase16Error(RuntimeError):
    """Raised when an immutable Phase 16 contract would be violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase16Error(message)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Phase16Error(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the current frozen decode/cache/backend/encoder implementation, but do
# not call its main routine or any cache-build path.
v1 = _load("phase16_fast_chroma_field", TOOLS / "fast_chroma_field.py")
adapter = _load("phase16_input_adapter", TOOLS / "run_br2049_v04_legacy.py")
fastcore = v1.fastcore
om = v1.om

FROZEN_PROTOCOL = HERE / "PHASE16_FROZEN_PROTOCOL.json"
AMENDMENT = HERE / "PHASE16_PER_SHOT_BOUNDARY_AMENDMENT.json"
EXECUTION_LOCK = HERE / "PHASE16_EXECUTION_LOCK.json"
SEGMENTATION = HERE / "matrix08_verified_segmentation.json"
INPUT_PROFILE = HERE / "matrix08_boundary_confirmation.json"
BASELINE_FIELD = WORKSPACE / "research" / "physical_validation" / "openmatte_hdr_chroma_field" / "matrix08_gpu_milestone.json"
BASELINE_INTENSITY = WORKSPACE / "research" / "physical_validation" / "openmatte_hdr_chroma_field" / "matrix08_intensity_iteration.json"
DEFAULT_OUTPUT = HERE / "results" / "matrix08_phase16_per_shot"

PEAK_NITS = float(fastcore.PEAK_NITS)
EPS = 1e-12
HUE_FLOOR = 1e-4
EXPECTED_INTERVAL = (73274, 73460)
EXPECTED_SHOTS = (
    ("Matrix-08-01", 73274, 73295),
    ("Matrix-08-02", 73295, 73434),
    ("Matrix-08-03", 73434, 73460),
)
VISUAL_TARGETS = (
    "rice paper",
    "skin",
    "walls",
    "dark clothing",
    "green and cyan regions",
    "red regions",
    "saturated highlights",
    "top seam",
    "bottom seam",
)


@dataclass(frozen=True)
class ShotPlan:
    shot_id: str
    start_frame: int
    end_frame_exclusive: int
    cache_start: int
    cache_end: int

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    @property
    def frame_weight(self) -> float:
        return 1.0 / float(self.frame_count)

    def contains(self, absolute_frame: int) -> bool:
        return self.start_frame <= absolute_frame < self.end_frame_exclusive


@dataclass(frozen=True)
class SamplingPlan:
    width: int
    height: int
    train_positions: np.ndarray
    holdout_positions: np.ndarray
    train_basis: np.ndarray
    holdout_basis: np.ndarray
    manifest: dict[str, Any]


@dataclass
class FrozenBaseline:
    field_complex: np.ndarray
    magnitude_gpu: Any
    angle_gpu: Any
    controls: dict[str, float]
    provenance: dict[str, Any]


@dataclass
class ShotFit:
    plan: ShotPlan
    gain_info: dict[str, Any]
    gain_gpu: Any
    coefficient_ct: np.ndarray
    coefficient_cp: np.ndarray
    field_ct_gpu: Any
    field_cp_gpu: Any
    alpha_t_raw: float
    alpha_p_raw: float
    alpha_t_shrunk: float
    alpha_p_shrunk: float
    alpha_t: float
    alpha_p: float
    threshold_p50: float
    threshold_p90: float
    fit_info: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required file is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_array(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def shot_summary(plan: ShotPlan) -> dict[str, Any]:
    return {
        "shot_id": plan.shot_id,
        "interval": [plan.start_frame, plan.end_frame_exclusive],
        "frame_count": plan.frame_count,
        "per_frame_fit_weight": plan.frame_weight,
        "cache_indices": [plan.cache_start, plan.cache_end],
    }


def declared_shots(items: list[dict[str, Any]], frames_key: str) -> list[list[Any]]:
    return [[item["shot_id"], *item[frames_key]] for item in items]


def expected_declared_shots() -> list[list[Any]]:
    return [[shot_id, start, end] for shot_id, start, end in EXPECTED_SHOTS]


def validate_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = read_json(FROZEN_PROTOCOL)
    amendment = read_json(AMENDMENT)
    lock = read_json(EXECUTION_LOCK)
    segmentation = read_json(SEGMENTATION)

    require(frozen["phase"].startswith("Phase 16"), "Wrong frozen protocol")
    require(frozen["status"] == "PREDECLARED BEFORE EXTRACTION", "Frozen protocol state changed")
    require(amendment["status"] == "BOUNDARIES_CONFIRMED__FIT_NOT_RUN", "Per-shot amendment is not in the approved state")
    require(amendment["amends"] == FROZEN_PROTOCOL.name, "Amendment does not target the frozen protocol")
    require(amendment["segmentation_record"] == SEGMENTATION.name, "Amendment references a different segmentation")
    require(lock["status"] == "PREDECLARED_BEFORE_PHASE16_FIT", "Execution lock must be predeclared")
    require(lock["scope"]["segmentation_record"] == SEGMENTATION.name, "Execution lock references a different segmentation")
    require(segmentation["status"] == "CONFIRMED_FROM_ACTUAL_FRAME_REVIEW", "Segmentation is not confirmed")

    analysis = segmentation["analysis_interval"]
    require(
        (analysis["start_frame"], analysis["end_frame_exclusive"], analysis["frame_count"]) == (73274, 73460, 186),
        "Unexpected segmentation interval",
    )
    expected = expected_declared_shots()
    actual = [[item["shot_id"], item["start_frame"], item["end_frame_exclusive"]] for item in segmentation["shots"]]
    require(actual == expected, f"Segmentation differs from confirmed shots: {actual}")
    require(declared_shots(lock["scope"]["shots"], "frames") == expected, "Execution-lock shots differ from confirmed shots")
    require(declared_shots(amendment["scope"]["actual_shots"], "frames") == expected, "Amendment shots differ from confirmed shots")
    require(lock["scope"]["analysis_interval"] == list(EXPECTED_INTERVAL), "Execution lock interval differs")
    require(amendment["scope"]["analysis_interval"] == list(EXPECTED_INTERVAL), "Amendment interval differs")
    require(segmentation["coverage_validation"]["ordered"], "Shot order is invalid")
    require(segmentation["coverage_validation"]["gap_free"], "Shot coverage has a gap")
    require(segmentation["coverage_validation"]["non_overlapping"], "Shots overlap")
    require(segmentation["coverage_validation"]["frame_count_sum"] == 186, "Shot frame total is wrong")
    for shot in segmentation["shots"]:
        require(shot["segmentation_status"] == "VERIFIED", f"{shot['shot_id']} is not verified")
        require(shot["phase16_fit_status"] == "ELIGIBLE_NOT_RUN", f"{shot['shot_id']} is not eligible")
        require(shot["frame_count"] >= 8, f"{shot['shot_id']} is too short")
    return frozen, amendment, lock, segmentation


def load_context(
    output_dir: Path,
    force_cpu: bool,
) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any], list[ShotPlan], dict[str, Any]]:
    _, _, lock, segmentation = validate_documents()
    profile = adapter.load_profile(INPUT_PROFILE)
    require((int(profile["shot_start"]), int(profile["shot_end"])) == EXPECTED_INTERVAL, "Input profile interval changed")
    config = adapter.build_config(profile, output_dir)
    require(config.frame_count == 186, "Input profile must resolve exactly 186 cached frames")
    v1.CACHE_ROOT = adapter.workspace_path(profile["cache_root"])
    cache = v1.cache_for(config, config.frame_count)
    require(cache.directory.name == "73274_186", f"Unexpected cache identity: {cache.directory}")
    require(cache.valid(), f"The existing Matrix-08 cache is required; will not rebuild: {cache.directory}")
    sdr_cache, hdr_cache, manifest = cache.open()
    expected_hashes = segmentation["input_identity"]["cached_source_frame_hashes"]
    require(manifest.get("source_frame_hashes") == expected_hashes, "Cache source hashes differ from verified segmentation")
    require(manifest.get("frames") == 186, "Cache manifest frame count differs")

    plans: list[ShotPlan] = []
    for expected_id, start, end in EXPECTED_SHOTS:
        cache_start, cache_end = start - config.shot_start, end - config.shot_start
        require(0 <= cache_start < cache_end <= len(sdr_cache), f"Out-of-cache shot: {expected_id}")
        plans.append(ShotPlan(expected_id, start, end, cache_start, cache_end))
    require(sum(plan.frame_count for plan in plans) == len(sdr_cache), "Shot plans do not cover cache exactly")

    backend = fastcore.Backend(use_gpu=not force_cpu and fastcore.gpu_available())
    context = {
        "profile": profile,
        "profile_path": str(INPUT_PROFILE),
        "cache": {
            "path": str(cache.directory),
            "gigabytes": cache.gigabytes,
            "fingerprint": manifest.get("fingerprint"),
            "source_frame_hashes": manifest["source_frame_hashes"],
            "reused": True,
        },
        "backend": backend.name,
        "execution_lock": str(EXECUTION_LOCK),
        "segmentation": str(SEGMENTATION),
        "plans": [shot_summary(plan) for plan in plans],
    }
    return backend, config, sdr_cache, hdr_cache, context, plans, lock


def make_sampling_plan(config: Any, lock: dict[str, Any]) -> SamplingPlan:
    _, y1, x2, y2 = config.overlap
    width, height = x2, y2 - y1
    declared = lock["spatial_train_holdout"]
    require([width, height] == declared["overlap_size"], "Unexpected overlap geometry")
    stride = int(declared["pixel_stride"])
    block_width, block_height = (int(value) for value in declared["block_size"])
    require(width % block_width == 0 and height % block_height == 0, "Spatial blocks must tile overlap exactly")

    rows, columns = np.indices((height, width), dtype=np.int32)
    lattice = (rows % stride == 0) & (columns % stride == 0)
    holdout = (((columns // block_width) + (rows // block_height)) % 5 == 2)
    train_positions = np.flatnonzero((lattice & ~holdout).reshape(-1)).astype(np.int64)
    holdout_positions = np.flatnonzero((lattice & holdout).reshape(-1)).astype(np.int64)
    require(train_positions.size == int(declared["expected_train_samples_per_frame"]), "Unexpected TRAIN sample count")
    require(holdout_positions.size == int(declared["expected_holdout_samples_per_frame"]), "Unexpected HOLDOUT sample count")
    require(np.intersect1d(train_positions, holdout_positions).size == 0, "Spatial split overlaps")
    require(train_positions.size + holdout_positions.size == int(np.sum(lattice)), "Spatial split does not cover lattice")

    def basis_for_positions(positions: np.ndarray) -> np.ndarray:
        yy, xx = np.divmod(positions, width)
        x = 2.0 * ((xx.astype(np.float64) + 0.5) / width) - 1.0
        y = 2.0 * ((yy.astype(np.float64) + 0.5) / height) - 1.0
        return np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))

    train_basis = basis_for_positions(train_positions)
    holdout_basis = basis_for_positions(holdout_positions)
    manifest = {
        "overlap_size": [width, height],
        "pixel_stride": stride,
        "block_size": [block_width, block_height],
        "holdout_rule": "((block_x + block_y) mod 5) == 2",
        "train_samples_per_frame": int(train_positions.size),
        "holdout_samples_per_frame": int(holdout_positions.size),
        "train_basis_sha256": sha256_array(train_basis.astype(np.float64)),
        "holdout_basis_sha256": sha256_array(holdout_basis.astype(np.float64)),
        "train_position_sha256": sha256_array(train_positions),
        "holdout_position_sha256": sha256_array(holdout_positions),
    }
    return SamplingPlan(width, height, train_positions, holdout_positions, train_basis, holdout_basis, manifest)


def load_frozen_baseline(config: Any, backend: Any) -> FrozenBaseline:
    field_source = read_json(BASELINE_FIELD)
    intensity_source = read_json(BASELINE_INTENSITY)
    field_payload = field_source["accelerated_full_run"]["chroma_field"]
    controls_payload = intensity_source["new_bounded_controls"]
    require(field_source["source_contract"]["hdr_frame_interval"] == list(EXPECTED_INTERVAL), "Frozen field source interval changed")
    require(intensity_source["source_contract"]["hdr_frame_interval"] == list(EXPECTED_INTERVAL), "Frozen intensity source interval changed")
    require(intensity_source["decision"]["adopt_for_next_poc_default"] == "intensity_saturation", "Frozen baseline is not saturation-only")

    x = (np.arange(config.om_size[0], dtype=np.float64) + 0.5) / config.om_size[0]
    per_column: dict[str, np.ndarray] = {}
    coefficients = field_payload["coefficients_highest_order_first"]
    for edge in ("top", "bottom"):
        values = np.asarray(coefficients[edge], dtype=np.float64)
        real = np.polyval(values[:, 0], x)
        imaginary = np.polyval(values[:, 1], x)
        limited, _ = v1.clamp_z(real + 1j * imaginary)
        per_column[edge] = limited
    z_shot = complex(*field_payload["z_shot"])
    frozen_field = v1.chroma_field(config, {"per_column": per_column, "z_shot": z_shot})
    controls = {
        "intensity_centre": float(controls_payload["intensity_centre"]),
        "intensity_normalization": float(controls_payload["intensity_normalization"]),
        "log_saturation_slope": float(controls_payload["log_saturation_slope"]),
        "hue_slope_radians": 0.0,
    }
    magnitude = backend.asarray(np.abs(frozen_field).astype(np.float32))
    angle = backend.asarray(np.angle(frozen_field).astype(np.float32))
    provenance = {
        "field_source": str(BASELINE_FIELD),
        "intensity_source": str(BASELINE_INTENSITY),
        "field_parameter_count": 12,
        "field_coefficients_highest_order_first": coefficients,
        "field_z_shot": field_payload["z_shot"],
        "field_sha256": sha256_array(frozen_field.astype(np.complex128)),
        "intensity_controls": controls,
        "rule": "Historical frozen baseline constants; not refitted, averaged, selected, or used to estimate the Phase 16 residual.",
    }
    return FrozenBaseline(frozen_field, magnitude, angle, controls, provenance)


def clipped_rgb_with_diagnostics(backend: Any, raw: Any, original_i: Any, i_change_before_clip: float) -> tuple[Any, dict[str, float]]:
    clipped = backend.xp.clip(raw, 0.0, 1.0)
    postclip_i = backend.to_ictcp(clipped * PEAK_NITS)[..., 0]
    return clipped, {
        "negative_rgb_fraction_before_clip": float(backend.tohost(backend.xp.mean(raw < -1e-4))),
        "above_peak_rgb_fraction_before_clip": float(backend.tohost(backend.xp.mean(raw > 1.0001))),
        "I_change_max_before_clip": i_change_before_clip,
        "I_change_max_after_clip": float(backend.tohost(backend.xp.max(backend.xp.abs(postclip_i - original_i)))),
    }


def baseline_rgb(backend: Any, config: Any, sdr: Any, gain: Any, baseline: FrozenBaseline) -> tuple[Any, dict[str, float]]:
    """Apply the frozen field plus accepted saturation-only intensity slope."""
    predicted = v1.predict(backend, config, sdr, backend.blur(sdr, config.base_sigma), gain)
    ictcp = backend.to_ictcp(predicted * PEAK_NITS)
    original_i = ictcp[..., 0].copy()
    chroma = ictcp[..., 1:]
    low = backend.blur(chroma, v1.CHROMA_SIGMA)
    detail = chroma - low
    intensity = backend.blur(ictcp[..., 0], v1.CHROMA_SIGMA)
    controls = baseline.controls
    t = backend.xp.clip(
        (intensity - controls["intensity_centre"]) / controls["intensity_normalization"],
        -1.0,
        1.0,
    )
    magnitude = backend.xp.clip(
        baseline.magnitude_gpu * backend.xp.exp(controls["log_saturation_slope"] * t),
        v1.SCALE_CLAMP[0],
        v1.SCALE_CLAMP[1],
    )
    real = magnitude * backend.xp.cos(baseline.angle_gpu)
    imaginary = magnitude * backend.xp.sin(baseline.angle_gpu)
    ct, cp = low[..., 0], low[..., 1]
    corrected = backend.xp.stack((ct * real - cp * imaginary, ct * imaginary + cp * real), axis=-1)
    output_ictcp = ictcp.copy()
    output_ictcp[..., 1:] = corrected + detail
    i_change = float(backend.tohost(backend.xp.max(backend.xp.abs(output_ictcp[..., 0] - original_i))))
    require(i_change == 0.0, "Frozen baseline changed ICtCp I before RGB clipping")
    raw = backend.from_ictcp(output_ictcp) / PEAK_NITS
    return clipped_rgb_with_diagnostics(backend, raw, original_i, i_change)


def make_full_field(config: Any, coefficient: np.ndarray) -> np.ndarray:
    """Evaluate an overlap-normalized degree-2 field across the full OM height."""
    width, full_height = config.om_size
    _, y1, _, y2 = config.overlap
    overlap_height = y2 - y1
    x = 2.0 * ((np.arange(width, dtype=np.float64) + 0.5) / width) - 1.0
    y = 2.0 * ((np.arange(full_height, dtype=np.float64) - y1 + 0.5) / overlap_height) - 1.0
    y = np.clip(y, -1.0, 1.0)
    xx, yy = np.meshgrid(x, y)
    return (
        coefficient[0]
        + coefficient[1] * xx
        + coefficient[2] * yy
        + coefficient[3] * xx * xx
        + coefficient[4] * xx * yy
        + coefficient[5] * yy * yy
    ).astype(np.float32)


def fit_one_shot(
    backend: Any,
    config: Any,
    sdr_cache: np.ndarray,
    hdr_cache: np.ndarray,
    plan: ShotPlan,
    sampling: SamplingPlan,
    baseline: FrozenBaseline,
    lock: dict[str, Any],
) -> ShotFit:
    """Fit one residual shape, alpha pair, and threshold set from that shot's TRAIN data only."""
    require(plan.cache_end - plan.cache_start == plan.frame_count, "Invalid cache slice")
    sdr_view = sdr_cache[plan.cache_start : plan.cache_end]
    hdr_view = hdr_cache[plan.cache_start : plan.cache_end]
    require(len(sdr_view) == plan.frame_count and len(hdr_view) == plan.frame_count, "Wrong shot cache view")
    _, y1, _, y2 = config.overlap
    started = time.perf_counter()
    gain_info = v1.measure_gain(backend, config, sdr_view, hdr_view, stride=1)
    require(gain_info["frames_used"] == plan.frame_count, "Gain fit used frames outside the active shot")
    gain_gpu = backend.asarray(om.gain_field(config, gain_info))

    normal = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros((6, 2), dtype=np.float64)
    threshold_values: list[np.ndarray] = []
    residual_norm_sum = np.zeros(2, dtype=np.float64)
    residual_norm_count = 0
    frame_weight = plan.frame_weight
    train_basis = sampling.train_basis
    train_count = float(train_basis.shape[0])
    normal_per_frame = (train_basis.T @ train_basis) / train_count
    train_positions_gpu = backend.asarray(sampling.train_positions)

    for local_index in range(plan.frame_count):
        absolute_frame = plan.start_frame + local_index
        require(plan.contains(absolute_frame), f"Cross-shot frame entered fit: {absolute_frame}")
        sdr = backend.asarray(sdr_view[local_index])
        hdr = backend.asarray(hdr_view[local_index])
        baseline_image, _ = baseline_rgb(backend, config, sdr, gain_gpu, baseline)
        baseline_low_full = backend.blur(backend.to_ictcp(baseline_image * PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
        baseline_low_overlap = baseline_low_full[y1:y2]
        hdr_low = backend.blur(backend.to_ictcp(hdr * PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
        require(
            baseline_low_overlap.shape == hdr_low.shape == (sampling.height, sampling.width, 2),
            "Phase 16 fit requires matched SDR/HDR overlap geometry",
        )
        residual = backend.tohost((hdr_low - baseline_low_overlap).reshape(-1, 2)[train_positions_gpu]).astype(np.float64)
        reference = backend.tohost(hdr_low.reshape(-1, 2)[train_positions_gpu]).astype(np.float64)
        normal += frame_weight * normal_per_frame
        rhs += frame_weight * ((train_basis.T @ residual) / train_count)
        threshold_values.append(np.hypot(reference[:, 0], reference[:, 1]).astype(np.float32))
        residual_norm_sum += np.sum(residual * residual, axis=0)
        residual_norm_count += residual.shape[0]

    require(np.isfinite(normal).all() and np.isfinite(rhs).all(), "Nonfinite TRAIN normal equations")
    ridge_relative = float(lock["new_residual_field"]["shape_ridge_relative"])
    ridge = ridge_relative * float(np.trace(normal)) / 6.0
    raw_coefficients = np.linalg.solve(normal + ridge * np.eye(6, dtype=np.float64), rhs)
    raw_ct, raw_cp = raw_coefficients[:, 0], raw_coefficients[:, 1]
    field_ct_train = train_basis @ raw_ct
    field_cp_train = train_basis @ raw_cp
    rms_ct = float(np.sqrt(np.mean(field_ct_train * field_ct_train)))
    rms_cp = float(np.sqrt(np.mean(field_cp_train * field_cp_train)))
    require(rms_ct > EPS and rms_cp > EPS, "Degenerate Phase 16 field shape")

    coefficient_ct = raw_ct / rms_ct
    coefficient_cp = raw_cp / rms_cp
    unit_ct = train_basis @ coefficient_ct
    unit_cp = train_basis @ coefficient_cp
    require(abs(float(np.sqrt(np.mean(unit_ct * unit_ct))) - 1.0) < 1e-10, "Ct field failed unit-RMS normalization")
    require(abs(float(np.sqrt(np.mean(unit_cp * unit_cp))) - 1.0) < 1e-10, "Cp field failed unit-RMS normalization")

    # rhs is the equal-frame-weighted mean of F^T residual. Dotting it with the
    # unit-RMS coefficient is exactly mean_train(unit_shape * residual).
    alpha_t_raw = float(coefficient_ct @ rhs[:, 0])
    alpha_p_raw = float(coefficient_cp @ rhs[:, 1])
    shrinkage = float(lock["new_residual_field"]["alpha_shrinkage"])
    alpha_bound = float(lock["new_residual_field"]["alpha_bound"])
    alpha_t_shrunk = alpha_t_raw / (1.0 + shrinkage)
    alpha_p_shrunk = alpha_p_raw / (1.0 + shrinkage)
    alpha_t = float(np.clip(alpha_t_shrunk, -alpha_bound, alpha_bound))
    alpha_p = float(np.clip(alpha_p_shrunk, -alpha_bound, alpha_bound))

    threshold_chroma = np.concatenate(threshold_values)
    p50, p90 = (float(np.percentile(threshold_chroma, value)) for value in (50.0, 90.0))
    require(math.isfinite(p50) and math.isfinite(p90) and p50 <= p90, "Invalid TRAIN-only chroma thresholds")
    field_ct_gpu = backend.asarray(make_full_field(config, coefficient_ct))
    field_cp_gpu = backend.asarray(make_full_field(config, coefficient_cp))
    fit_info = {
        "frames_used": plan.frame_count,
        "cache_indices_used": [plan.cache_start, plan.cache_end],
        "frame_weight": frame_weight,
        "temporal_aggregation": "equal frame weight within this shot only",
        "holdout_samples_used": 0,
        "train_samples_per_frame": int(train_count),
        "train_total_samples": int(train_count) * plan.frame_count,
        "normal_matrix": normal.tolist(),
        "normal_matrix_trace": float(np.trace(normal)),
        "shape_ridge": ridge,
        "shape_ridge_relative": ridge_relative,
        "raw_shape_coefficients": {"Ct": raw_ct.tolist(), "Cp": raw_cp.tolist()},
        "unit_rms_shape_coefficients": {"Ct": coefficient_ct.tolist(), "Cp": coefficient_cp.tolist()},
        "unit_rms": {
            "Ct": float(np.sqrt(np.mean(unit_ct * unit_ct))),
            "Cp": float(np.sqrt(np.mean(unit_cp * unit_cp))),
        },
        "alpha_formula": "mean_train(unit_rms_shape * lowpass_residual)",
        "alpha_shrinkage": shrinkage,
        "alpha_bound": alpha_bound,
        "alpha_t_clamped": alpha_t != alpha_t_shrunk,
        "alpha_p_clamped": alpha_p != alpha_p_shrunk,
        "train_reference_chroma_thresholds": {
            "P50_low": p50,
            "P90_high": p90,
            "sample_count": int(threshold_chroma.size),
        },
        "train_lowpass_residual_rms": (np.sqrt(residual_norm_sum / max(residual_norm_count, 1))).tolist(),
        "seconds": time.perf_counter() - started,
    }
    return ShotFit(
        plan=plan,
        gain_info=gain_info,
        gain_gpu=gain_gpu,
        coefficient_ct=coefficient_ct,
        coefficient_cp=coefficient_cp,
        field_ct_gpu=field_ct_gpu,
        field_cp_gpu=field_cp_gpu,
        alpha_t_raw=alpha_t_raw,
        alpha_p_raw=alpha_p_raw,
        alpha_t_shrunk=alpha_t_shrunk,
        alpha_p_shrunk=alpha_p_shrunk,
        alpha_t=alpha_t,
        alpha_p=alpha_p,
        threshold_p50=p50,
        threshold_p90=p90,
        fit_info=fit_info,
    )


def apply_extension(backend: Any, baseline_image: Any, fit: ShotFit) -> tuple[Any, dict[str, float]]:
    """Apply only the approved additive low-frequency Ct/Cp residual to A."""
    if fit.alpha_t == 0.0 and fit.alpha_p == 0.0:
        return baseline_image, {
            "negative_rgb_fraction_before_clip": 0.0,
            "above_peak_rgb_fraction_before_clip": 0.0,
            "I_change_max_before_clip": 0.0,
            "I_change_max_after_clip": 0.0,
        }
    ictcp = backend.to_ictcp(baseline_image * PEAK_NITS)
    original_i = ictcp[..., 0].copy()
    chroma = ictcp[..., 1:]
    low = backend.blur(chroma, v1.CHROMA_SIGMA)
    detail = chroma - low
    delta = backend.xp.stack((fit.alpha_t * fit.field_ct_gpu, fit.alpha_p * fit.field_cp_gpu), axis=-1)
    output_ictcp = ictcp.copy()
    output_ictcp[..., 1:] = low + delta + detail
    i_change = float(backend.tohost(backend.xp.max(backend.xp.abs(output_ictcp[..., 0] - original_i))))
    require(i_change == 0.0, "Phase 16 changed ICtCp I before RGB clipping")
    raw = backend.from_ictcp(output_ictcp) / PEAK_NITS
    return clipped_rgb_with_diagnostics(backend, raw, original_i, i_change)


def exact_hdr_composite(backend: Any, config: Any, prediction: Any, hdr: Any) -> Any:
    """Copy every HDR overlap row exactly, as the Phase 16 visual contract requires."""
    _, y1, _, y2 = config.overlap
    output = prediction.copy()
    output[y1:y2] = hdr
    return output


def delta_e_itp(predicted_ictcp: np.ndarray, reference_ictcp: np.ndarray) -> np.ndarray:
    delta = predicted_ictcp - reference_ictcp
    return 720.0 * np.sqrt(delta[..., 0] ** 2 + (0.5 * delta[..., 1]) ** 2 + delta[..., 2] ** 2)


def delta_e_2000_203(predicted_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """Auxiliary CIEDE2000 under the frozen 203-nit BT.2408 white convention."""
    white_nits = 203.0
    predicted_709 = np.einsum("ij,...j->...i", BT2020_TO_BT709, predicted_rgb * PEAK_NITS / white_nits)
    reference_709 = np.einsum("ij,...j->...i", BT2020_TO_BT709, reference_rgb * PEAK_NITS / white_nits)
    predicted_lab = linear_to_lab_d65(np.clip(predicted_709, 0.0, 1.0))
    reference_lab = linear_to_lab_d65(np.clip(reference_709, 0.0, 1.0))
    return delta_e_2000(predicted_lab, reference_lab)


def wrapped_hue_distance(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    predicted_angle = np.degrees(np.arctan2(predicted[:, 1], predicted[:, 0]))
    reference_angle = np.degrees(np.arctan2(reference[:, 1], reference[:, 0]))
    return np.abs((predicted_angle - reference_angle + 180.0) % 360.0 - 180.0)


class MetricAccumulator:
    """Streaming means plus exact sampled P95s for one variant/split/shot."""

    def __init__(self, threshold_p50: float, threshold_p90: float) -> None:
        self.p50 = threshold_p50
        self.p90 = threshold_p90
        self.count = 0
        self.delta_e_itp_sum = 0.0
        self.lowpass_chroma_sum = 0.0
        self.chroma_abs_sum = 0.0
        self.luminance_abs_sum = 0.0
        self.luminance_squared_sum = 0.0
        self.hue_low_sum = 0.0
        self.hue_low_count = 0
        self.hue_high_sum = 0.0
        self.hue_high_count = 0
        self.delta_e_itp_values: list[np.ndarray] = []
        self.delta_e_2000_values: list[np.ndarray] = []

    def update(
        self,
        predicted_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        predicted_ictcp: np.ndarray,
        reference_ictcp: np.ndarray,
        predicted_low: np.ndarray,
        reference_low: np.ndarray,
    ) -> None:
        require(predicted_rgb.shape == reference_rgb.shape, "Metric RGB shape mismatch")
        require(predicted_ictcp.shape == reference_ictcp.shape, "Metric ICtCp shape mismatch")
        require(predicted_low.shape == reference_low.shape, "Metric low-pass shape mismatch")
        values = delta_e_itp(predicted_ictcp, reference_ictcp)
        aux = delta_e_2000_203(predicted_rgb, reference_rgb)
        predicted_chroma = np.hypot(predicted_ictcp[:, 1], predicted_ictcp[:, 2])
        reference_chroma = np.hypot(reference_ictcp[:, 1], reference_ictcp[:, 2])
        reference_low_chroma = np.hypot(reference_low[:, 0], reference_low[:, 1])
        hue = wrapped_hue_distance(predicted_ictcp[:, 1:], reference_ictcp[:, 1:])
        hue_valid = (predicted_chroma > HUE_FLOOR) & (reference_chroma > HUE_FLOOR)
        low_mask = hue_valid & (reference_low_chroma < self.p50)
        high_mask = hue_valid & (reference_low_chroma > self.p90)
        predicted_luma = np.einsum("...c,c->...", predicted_rgb, np.asarray(fastcore.BT2020_LUMA)) * PEAK_NITS
        reference_luma = np.einsum("...c,c->...", reference_rgb, np.asarray(fastcore.BT2020_LUMA)) * PEAK_NITS
        luminance_delta = predicted_luma - reference_luma
        lowpass_delta = np.hypot(predicted_low[:, 0] - reference_low[:, 0], predicted_low[:, 1] - reference_low[:, 1])
        self.count += int(values.size)
        self.delta_e_itp_sum += float(values.sum())
        self.lowpass_chroma_sum += float(lowpass_delta.sum())
        self.chroma_abs_sum += float(np.abs(predicted_chroma - reference_chroma).sum())
        self.luminance_abs_sum += float(np.abs(luminance_delta).sum())
        self.luminance_squared_sum += float(np.square(luminance_delta).sum())
        self.hue_low_sum += float(hue[low_mask].sum())
        self.hue_low_count += int(np.sum(low_mask))
        self.hue_high_sum += float(hue[high_mask].sum())
        self.hue_high_count += int(np.sum(high_mask))
        self.delta_e_itp_values.append(values.astype(np.float32, copy=False))
        self.delta_e_2000_values.append(aux.astype(np.float32, copy=False))

    def result(self) -> dict[str, Any]:
        require(self.count > 0, "Metric accumulator has no samples")
        de_itp = np.concatenate(self.delta_e_itp_values)
        de_2000 = np.concatenate(self.delta_e_2000_values)
        return {
            "sample_count": self.count,
            "deltaE_ITP": {
                "mean": self.delta_e_itp_sum / self.count,
                "P95": float(np.percentile(de_itp, 95)),
            },
            "low_frequency_chroma_residual": {"mean": self.lowpass_chroma_sum / self.count},
            "chroma_absolute_error": {"mean": self.chroma_abs_sum / self.count},
            "hue_absolute_error_degrees": {
                "low_chroma_P50": self.hue_low_sum / max(self.hue_low_count, 1),
                "high_chroma_P90": self.hue_high_sum / max(self.hue_high_count, 1),
                "low_chroma_valid_samples": self.hue_low_count,
                "high_chroma_valid_samples": self.hue_high_count,
            },
            "luminance_nits": {
                "MAE": self.luminance_abs_sum / self.count,
                "RMSE": math.sqrt(self.luminance_squared_sum / self.count),
            },
            "deltaE2000_auxiliary_203nit": {
                "mean": float(de_2000.mean()),
                "P95": float(np.percentile(de_2000, 95)),
            },
        }


class SeamAccumulator:
    """Measure exact-composite steps directly across each HDR-overlap boundary."""

    def __init__(self) -> None:
        self.values: dict[str, dict[str, list[np.ndarray]]] = {
            side: {"chroma": [], "hue": [], "luminance": []} for side in ("top", "bottom")
        }

    def update(self, backend: Any, config: Any, composite: Any) -> None:
        _, y1, _, y2 = config.overlap
        for side, outside, inside in (("top", y1 - 1, y1), ("bottom", y2 - 1, y2)):
            rgb = backend.tohost(composite[[outside, inside]])
            ictcp = backend.tohost(backend.to_ictcp(composite[[outside, inside]] * PEAK_NITS))
            chroma = np.hypot(ictcp[..., 1], ictcp[..., 2])
            chroma_step = np.abs(chroma[0] - chroma[1])
            hue_valid = (chroma[0] > HUE_FLOOR) & (chroma[1] > HUE_FLOOR)
            hue = wrapped_hue_distance(ictcp[0, :, 1:], ictcp[1, :, 1:])
            luminance = np.einsum("...c,c->...", rgb, np.asarray(fastcore.BT2020_LUMA)) * PEAK_NITS
            self.values[side]["chroma"].append(chroma_step.astype(np.float32))
            self.values[side]["hue"].append(hue[hue_valid].astype(np.float32))
            self.values[side]["luminance"].append(np.abs(luminance[0] - luminance[1]).astype(np.float32))

    def result(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for side, values in self.values.items():
            chroma = np.concatenate(values["chroma"])
            hue = np.concatenate(values["hue"])
            luminance = np.concatenate(values["luminance"])
            output[side] = {
                "chroma_step_mean": float(chroma.mean()),
                "hue_step_degrees_mean": float(hue.mean()) if hue.size else 0.0,
                "hue_valid_samples": int(hue.size),
                "luminance_step_nits_mean": float(luminance.mean()),
                "luminance_step_nits_P95": float(np.percentile(luminance, 95)),
            }
        return output


def host_samples(backend: Any, value: Any, positions: np.ndarray) -> np.ndarray:
    channel_count = int(value.shape[-1])
    return backend.tohost(value.reshape(-1, channel_count)[backend.asarray(positions)]).astype(np.float64)


def preview_tile(image: np.ndarray, label: str, width: int = 480, height: int = 270) -> np.ndarray:
    preview = np.round(om.preview(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)) * 255.0).astype(np.uint8)
    cv2.rectangle(preview, (0, 0), (width, 22), (0, 0, 0), -1)
    cv2.putText(preview, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def residual_image(value: np.ndarray, label: str, display_gain: float) -> np.ndarray:
    gray = np.clip(value * display_gain, 0.0, 1.0)
    heat = cv2.applyColorMap(np.round(gray * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.rectangle(heat, (0, 0), (heat.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(heat, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return heat


def seam_context_tile(image: np.ndarray, config: Any, variant: str) -> np.ndarray:
    _, y1, _, y2 = config.overlap
    band = 64
    top = image[max(0, y1 - band) : min(image.shape[0], y1 + band)]
    bottom = image[max(0, y2 - band) : min(image.shape[0], y2 + band)]
    return np.concatenate(
        (
            preview_tile(top, f"{variant} top seam", 960, 144),
            preview_tile(bottom, f"{variant} bottom seam", 960, 144),
        ),
        axis=0,
    )


def select_highest_chroma_window(hdr_low: np.ndarray) -> dict[str, Any]:
    """Choose a fixed-size window from HDR reference low-pass chroma only."""
    height, width = hdr_low.shape[:2]
    window_width = min(384, width)
    window_height = min(216, height)
    chroma = np.hypot(hdr_low[..., 0], hdr_low[..., 1]).astype(np.float64, copy=False)
    integral = np.pad(chroma.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    sums = (
        integral[window_height:, window_width:]
        - integral[:-window_height, window_width:]
        - integral[window_height:, :-window_width]
        + integral[:-window_height, :-window_width]
    )
    top, left = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return {
        "source": "HDR reference only",
        "measurement": "mean sigma-16 ICtCp chroma magnitude",
        "selection_tiebreak": "first maximum in row-major overlap order",
        "overlap_x": int(left),
        "overlap_y": int(top),
        "width": int(window_width),
        "height": int(window_height),
        "mean_reference_chroma": float(sums[top, left] / (window_width * window_height)),
    }


def highest_chroma_triptych(
    a_composite: np.ndarray,
    b_composite: np.ndarray,
    hdr: np.ndarray,
    config: Any,
    selection: dict[str, Any],
) -> np.ndarray:
    _, y1, _, _ = config.overlap
    left, top = selection["overlap_x"], selection["overlap_y"]
    width, height = selection["width"], selection["height"]
    full_top = y1 + top
    a_crop = a_composite[full_top : full_top + height, left : left + width]
    b_crop = b_composite[full_top : full_top + height, left : left + width]
    h_crop = hdr[top : top + height, left : left + width]
    return np.concatenate(
        (
            preview_tile(a_crop, "A exact-HDR-centre composite", 480, 270),
            preview_tile(b_crop, "B exact-HDR-centre composite", 480, 270),
            preview_tile(h_crop, "HDR-selected high-chroma window", 480, 270),
        ),
        axis=1,
    )


def save_visual_artifacts(
    directory: Path,
    representative_tiles: list[np.ndarray],
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    amplified_composite_difference: np.ndarray,
    seam_a: np.ndarray,
    seam_b: np.ndarray,
    high_chroma_window: np.ndarray,
    high_chroma_selection: dict[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    require(bool(representative_tiles), "Contact sheet has no representative frames")

    contact_path = directory / "contact_sheet.png"
    om.write_png(contact_path, np.concatenate(representative_tiles, axis=0))

    residual_dir = directory / "residual_maps"
    residual_dir.mkdir(parents=True, exist_ok=True)
    residual_a_path = residual_dir / "lowpass_chroma_A_overlap.png"
    residual_b_path = residual_dir / "lowpass_chroma_B_overlap.png"
    amplified_path = residual_dir / "amplified_B_minus_A_composite.png"
    om.write_png(residual_a_path, residual_image(residual_a, "A low-pass Ct/Cp residual vs HDR overlap x128", 128.0))
    om.write_png(residual_b_path, residual_image(residual_b, "B low-pass Ct/Cp residual vs HDR overlap x128", 128.0))
    om.write_png(amplified_path, residual_image(amplified_composite_difference, "|B-A| exact composite x64; HDR centre must be black", 64.0))

    seam_dir = directory / "seam_maps"
    seam_dir.mkdir(parents=True, exist_ok=True)
    seam_path = seam_dir / "top_bottom_A_B_context.png"
    om.write_png(seam_path, np.concatenate((seam_a, seam_b), axis=1))

    review_dir = directory / "review_windows"
    review_dir.mkdir(parents=True, exist_ok=True)
    high_chroma_path = review_dir / "highest_chroma_hdr_reference_only.png"
    om.write_png(high_chroma_path, high_chroma_window)
    manifest_path = directory / "visual_manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "openmatte-hdr-phase16-visual-artifacts/v1",
            "contact_sheet": str(contact_path),
            "residual_maps": [str(residual_a_path), str(residual_b_path), str(amplified_path)],
            "seam_maps": [str(seam_path)],
            "highest_chroma_window": {"path": str(high_chroma_path), **high_chroma_selection},
            "manual_review_targets": list(VISUAL_TARGETS),
            "manual_review_note": "No semantic segmentation or mask was created. The contact sheet and seam maps support manual review of the named material targets; the high-chroma crop is selected only from HDR reference chroma.",
        },
    )
    return {
        "contact_sheet": str(contact_path),
        "lowpass_residual_A": str(residual_a_path),
        "lowpass_residual_B": str(residual_b_path),
        "amplified_composite_difference": str(amplified_path),
        "seam_map": str(seam_path),
        "highest_chroma_window": str(high_chroma_path),
        "visual_manifest": str(manifest_path),
        "high_chroma_selection": high_chroma_selection,
        "manual_review_targets": list(VISUAL_TARGETS),
    }


def evaluate_one_shot(
    backend: Any,
    config: Any,
    sdr_cache: np.ndarray,
    hdr_cache: np.ndarray,
    fit: ShotFit,
    sampling: SamplingPlan,
    baseline: FrozenBaseline,
    visual_dir: Path,
) -> dict[str, Any]:
    """Evaluate both variants per shot; HOLDOUT receives no fitting operation."""
    plan = fit.plan
    sdr_view = sdr_cache[plan.cache_start : plan.cache_end]
    hdr_view = hdr_cache[plan.cache_start : plan.cache_end]
    require(len(sdr_view) == len(hdr_view) == plan.frame_count, "Wrong evaluation cache view")
    accumulators = {
        "A": {
            "train": MetricAccumulator(fit.threshold_p50, fit.threshold_p90),
            "holdout": MetricAccumulator(fit.threshold_p50, fit.threshold_p90),
        },
        "B": {
            "train": MetricAccumulator(fit.threshold_p50, fit.threshold_p90),
            "holdout": MetricAccumulator(fit.threshold_p50, fit.threshold_p90),
        },
    }
    seams = {"A": SeamAccumulator(), "B": SeamAccumulator()}
    range_sum = {
        "A": {"negative": 0.0, "above": 0.0, "postclip_i_max": 0.0},
        "B": {"negative": 0.0, "above": 0.0, "postclip_i_max": 0.0},
    }
    representative_local = sorted({0, max(0, (plan.frame_count - 1) // 2), plan.frame_count - 1})
    contact_rows: list[np.ndarray] = []
    residual_a: np.ndarray | None = None
    residual_b: np.ndarray | None = None
    amplified_composite_difference: np.ndarray | None = None
    seam_a: np.ndarray | None = None
    seam_b: np.ndarray | None = None
    high_chroma_window: np.ndarray | None = None
    high_chroma_selection: dict[str, Any] | None = None
    center_max = {"A": 0.0, "B": 0.0}
    application_seconds = 0.0
    started = time.perf_counter()
    _, y1, _, y2 = config.overlap

    for local_index in range(plan.frame_count):
        absolute_frame = plan.start_frame + local_index
        require(plan.contains(absolute_frame), f"Evaluation crossed shot boundary: {absolute_frame}")
        sdr = backend.asarray(sdr_view[local_index])
        hdr = backend.asarray(hdr_view[local_index])

        application_started = time.perf_counter()
        a_image, a_range = baseline_rgb(backend, config, sdr, fit.gain_gpu, baseline)
        b_image, b_range = apply_extension(backend, a_image, fit)
        application_seconds += time.perf_counter() - application_started

        a_ictcp = backend.to_ictcp(a_image * PEAK_NITS)
        b_ictcp = backend.to_ictcp(b_image * PEAK_NITS)
        h_ictcp = backend.to_ictcp(hdr * PEAK_NITS)
        a_low_full = backend.blur(a_ictcp[..., 1:], v1.CHROMA_SIGMA)
        b_low_full = backend.blur(b_ictcp[..., 1:], v1.CHROMA_SIGMA)
        h_low = backend.blur(h_ictcp[..., 1:], v1.CHROMA_SIGMA)
        a_low = a_low_full[y1:y2]
        b_low = b_low_full[y1:y2]
        a_ictcp_overlap = a_ictcp[y1:y2]
        b_ictcp_overlap = b_ictcp[y1:y2]
        require(a_low.shape == b_low.shape == h_low.shape == (sampling.height, sampling.width, 2), "Evaluation low-pass geometry mismatch")
        require(a_ictcp_overlap.shape == b_ictcp_overlap.shape == h_ictcp.shape, "Evaluation ICtCp geometry mismatch")
        variants = {
            "A": (a_image[y1:y2], a_ictcp_overlap, a_low),
            "B": (b_image[y1:y2], b_ictcp_overlap, b_low),
        }
        for split, positions in (("train", sampling.train_positions), ("holdout", sampling.holdout_positions)):
            reference_rgb = host_samples(backend, hdr, positions)
            reference_ictcp = host_samples(backend, h_ictcp, positions)
            reference_low = host_samples(backend, h_low, positions)
            for name, (image, ictcp, low) in variants.items():
                accumulators[name][split].update(
                    host_samples(backend, image, positions),
                    reference_rgb,
                    host_samples(backend, ictcp, positions),
                    reference_ictcp,
                    host_samples(backend, low, positions),
                    reference_low,
                )

        a_composite = exact_hdr_composite(backend, config, a_image, hdr)
        b_composite = exact_hdr_composite(backend, config, b_image, hdr)
        for name, composite in (("A", a_composite), ("B", b_composite)):
            center_max[name] = max(
                center_max[name],
                float(backend.tohost(backend.xp.max(backend.xp.abs(composite[y1:y2] - hdr)))),
            )
            seams[name].update(backend, config, composite)
        for name, diagnostics in (("A", a_range), ("B", b_range)):
            range_sum[name]["negative"] += diagnostics["negative_rgb_fraction_before_clip"]
            range_sum[name]["above"] += diagnostics["above_peak_rgb_fraction_before_clip"]
            range_sum[name]["postclip_i_max"] = max(range_sum[name]["postclip_i_max"], diagnostics["I_change_max_after_clip"])

        if local_index in representative_local:
            a_host = backend.tohost(a_composite)
            b_host = backend.tohost(b_composite)
            h_host = backend.tohost(hdr)
            reference_full = np.zeros_like(a_host)
            reference_full[y1:y2] = h_host
            contact_rows.append(
                np.concatenate(
                    (
                        preview_tile(a_host, f"{absolute_frame} / A exact-HDR-centre composite"),
                        preview_tile(b_host, f"{absolute_frame} / B exact-HDR-centre composite"),
                        preview_tile(reference_full, f"{absolute_frame} / HDR overlap reference"),
                    ),
                    axis=1,
                )
            )
            if local_index == representative_local[len(representative_local) // 2]:
                residual_a = np.hypot(*(backend.tohost(a_low - h_low).transpose(2, 0, 1))).astype(np.float32)
                residual_b = np.hypot(*(backend.tohost(b_low - h_low).transpose(2, 0, 1))).astype(np.float32)
                amplified_composite_difference = np.max(np.abs(b_host - a_host), axis=2).astype(np.float32)
                require(float(np.max(amplified_composite_difference[y1:y2])) == 0.0, "Amplified composite residual is nonzero in HDR centre")
                seam_a = seam_context_tile(a_host, config, "A")
                seam_b = seam_context_tile(b_host, config, "B")
                high_chroma_selection = select_highest_chroma_window(backend.tohost(h_low))
                high_chroma_window = highest_chroma_triptych(a_host, b_host, h_host, config, high_chroma_selection)

    require(
        residual_a is not None
        and residual_b is not None
        and amplified_composite_difference is not None
        and seam_a is not None
        and seam_b is not None
        and high_chroma_window is not None
        and high_chroma_selection is not None,
        "Visual sample missing",
    )
    require(center_max["A"] == 0.0 and center_max["B"] == 0.0, "HDR centre changed in Phase 16 composite")
    artifacts = save_visual_artifacts(
        visual_dir,
        contact_rows,
        residual_a,
        residual_b,
        amplified_composite_difference,
        seam_a,
        seam_b,
        high_chroma_window,
        high_chroma_selection,
    )
    metrics = {name: {split: accumulator.result() for split, accumulator in groups.items()} for name, groups in accumulators.items()}
    expected_counts = {
        "train": sampling.train_positions.size * plan.frame_count,
        "holdout": sampling.holdout_positions.size * plan.frame_count,
    }
    for variant in ("A", "B"):
        for split, expected_count in expected_counts.items():
            require(metrics[variant][split]["sample_count"] == expected_count, f"Unexpected {variant} {split} metric count")
    seams_result = {name: accumulator.result() for name, accumulator in seams.items()}
    ranges = {
        name: {
            "negative_rgb_fraction_before_clip_mean": values["negative"] / plan.frame_count,
            "above_peak_rgb_fraction_before_clip_mean": values["above"] / plan.frame_count,
            "I_change_max_after_clip": values["postclip_i_max"],
        }
        for name, values in range_sum.items()
    }
    return {
        "metrics": metrics,
        "seam": seams_result,
        "range": ranges,
        "visual_artifacts": artifacts,
        "hdr_centre_max_abs_residual": center_max,
        "timing": {
            "application_seconds": application_seconds,
            "application_seconds_per_frame": application_seconds / plan.frame_count,
            "evaluation_seconds": time.perf_counter() - started,
        },
    }


def percentage_improvement(before: float, after: float) -> float | None:
    return (before - after) / before if before > EPS else None


def previsual_verdict(result: dict[str, Any], fit: ShotFit) -> dict[str, Any]:
    a = result["evaluation"]["metrics"]["A"]["holdout"]
    b = result["evaluation"]["metrics"]["B"]["holdout"]
    chroma_gain = percentage_improvement(a["low_frequency_chroma_residual"]["mean"], b["low_frequency_chroma_residual"]["mean"])
    hue_gain = percentage_improvement(
        a["hue_absolute_error_degrees"]["high_chroma_P90"],
        b["hue_absolute_error_degrees"]["high_chroma_P90"],
    )
    luminance_change = (b["luminance_nits"]["MAE"] / max(a["luminance_nits"]["MAE"], EPS)) - 1.0
    seam_a = np.mean([value["chroma_step_mean"] for value in result["evaluation"]["seam"]["A"].values()])
    seam_b = np.mean([value["chroma_step_mean"] for value in result["evaluation"]["seam"]["B"].values()])
    seam_change = (seam_b / max(seam_a, EPS)) - 1.0
    alpha_bound = float(fit.fit_info["alpha_bound"])
    comfort_limit = alpha_bound * 0.5
    saturated = abs(fit.alpha_t) >= alpha_bound or abs(fit.alpha_p) >= alpha_bound
    comfortable = abs(fit.alpha_t) < comfort_limit and abs(fit.alpha_p) < comfort_limit
    metric_pass = bool(
        chroma_gain is not None
        and hue_gain is not None
        and chroma_gain >= 0.05
        and hue_gain >= 0.05
        and luminance_change <= 0.01
        and seam_change <= 0.02
        and comfortable
        and not saturated
    )
    return {
        "previsual_status": "METRIC_GATE_PASS__VISUAL_REVIEW_REQUIRED" if metric_pass else ("MODEL_SATURATION__FAIL" if saturated else "NO_VALUE_DEMONSTRATED"),
        "holdout_low_frequency_chroma_improvement_fraction": chroma_gain,
        "holdout_high_chroma_hue_improvement_fraction": hue_gain,
        "holdout_luminance_MAE_change_fraction": luminance_change,
        "seam_chroma_step_change_fraction": seam_change,
        "alpha_comfort_limit": comfort_limit,
        "alpha_comfortable": comfortable,
    }


def serializable_gain(gain_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "shot_gain_stops": float(gain_info["shot_gain_stops"]),
        "top_profile_range_stops": [float(value) for value in gain_info["top_profile_range_stops"]],
        "bottom_profile_range_stops": [float(value) for value in gain_info["bottom_profile_range_stops"]],
        "frames_used": int(gain_info["frames_used"]),
        "seconds": float(gain_info["seconds"]),
        "top_profile_sha256": sha256_array(np.asarray(gain_info["top_profile_stops"], dtype=np.float64)),
        "bottom_profile_sha256": sha256_array(np.asarray(gain_info["bottom_profile_stops"], dtype=np.float64)),
    }


def serializable_fit(fit: ShotFit) -> dict[str, Any]:
    return {
        "gain": serializable_gain(fit.gain_info),
        "alpha_t": {"raw": fit.alpha_t_raw, "shrunk": fit.alpha_t_shrunk, "applied": fit.alpha_t},
        "alpha_p": {"raw": fit.alpha_p_raw, "shrunk": fit.alpha_p_shrunk, "applied": fit.alpha_p},
        "new_residual_field": {
            "degree": 2,
            "basis": ["1", "x", "y", "x^2", "x*y", "y^2"],
            "coefficient_Ct": fit.coefficient_ct.tolist(),
            "coefficient_Cp": fit.coefficient_cp.tolist(),
            "unit_rms_region": "spatial TRAIN only",
        },
        "train_reference_chroma_thresholds": {"P50_low": fit.threshold_p50, "P90_high": fit.threshold_p90},
        "fit": fit.fit_info,
    }


def video_frame_count(config: Any, path: Path) -> int:
    command = [
        str(config.ffmpeg.with_name("ffprobe.exe")),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    require(completed.returncode == 0, f"Cannot count video frames: {path}")
    return int(json.loads(completed.stdout.decode("utf-8"))["streams"][0]["nb_read_frames"])


def render_full_interval(
    backend: Any,
    config: Any,
    sdr_cache: np.ndarray,
    hdr_cache: np.ndarray,
    fits: list[ShotFit],
    baseline: FrozenBaseline,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = config.om_size
    outputs = {
        "A_baseline": output_dir / "Matrix08_A_baseline_hdr10.mkv",
        "B_extended": output_dir / "Matrix08_B_extended_chroma_hdr10.mkv",
        "A_vs_B": output_dir / "Matrix08_A_vs_B_hdr10.mkv",
    }
    pipes = [
        v1.nvenc(config, outputs["A_baseline"], width, height),
        v1.nvenc(config, outputs["B_extended"], width, height),
        v1.nvenc(config, outputs["A_vs_B"], width * 2, height),
    ]
    writer = fastcore.ParallelWriter(pipes)
    fit_by_cache_index: dict[int, ShotFit] = {}
    for fit in fits:
        for cache_index in range(fit.plan.cache_start, fit.plan.cache_end):
            require(cache_index not in fit_by_cache_index, "A cache frame belongs to multiple shot fits")
            fit_by_cache_index[cache_index] = fit
    require(len(fit_by_cache_index) == len(sdr_cache), "Not every cache frame routes to one shot fit")

    _, y1, _, y2 = config.overlap
    centre_residual = {"A": 0.0, "B": 0.0}
    application_seconds = 0.0
    started = time.perf_counter()
    try:
        for cache_index in range(len(sdr_cache)):
            fit = fit_by_cache_index[cache_index]
            absolute_frame = config.shot_start + cache_index
            require(fit.plan.contains(absolute_frame), f"Cross-shot render dispatch: {absolute_frame}")
            sdr = backend.asarray(sdr_cache[cache_index])
            hdr = backend.asarray(hdr_cache[cache_index])
            application_started = time.perf_counter()
            a_image, _ = baseline_rgb(backend, config, sdr, fit.gain_gpu, baseline)
            b_image, _ = apply_extension(backend, a_image, fit)
            application_seconds += time.perf_counter() - application_started
            a_composite = exact_hdr_composite(backend, config, a_image, hdr)
            b_composite = exact_hdr_composite(backend, config, b_image, hdr)
            centre_residual["A"] = max(
                centre_residual["A"],
                float(backend.tohost(backend.xp.max(backend.xp.abs(a_composite[y1:y2] - hdr)))),
            )
            centre_residual["B"] = max(
                centre_residual["B"],
                float(backend.tohost(backend.xp.max(backend.xp.abs(b_composite[y1:y2] - hdr)))),
            )
            a_pq = backend.to_pq16(a_composite)
            b_pq = backend.to_pq16(b_composite)
            writer.submit([a_pq.tobytes(), b_pq.tobytes(), np.concatenate((a_pq, b_pq), axis=1).tobytes()])
    finally:
        writer.close()
        for pipe in pipes:
            if pipe.stdin:
                pipe.stdin.close()
        codes = [pipe.wait() for pipe in pipes]
        errors = [pipe.stderr.read().decode("utf-8", errors="replace") for pipe in pipes]
        require(all(code == 0 for code in codes), f"Phase 16 HDR10 render failed: {codes}; {errors}")

    require(centre_residual == {"A": 0.0, "B": 0.0}, "Full output altered HDR overlap")
    video: dict[str, Any] = {}
    for name, path in outputs.items():
        require(path.is_file(), f"Missing rendered video: {path}")
        count = video_frame_count(config, path)
        require(count == len(sdr_cache), f"Unexpected rendered video frame count for {path.name}: {count}")
        video[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "frames": count,
            "tags": om.verify_hdr_tags(config, path),
        }
    elapsed = time.perf_counter() - started
    return {
        "timing_seconds": elapsed,
        "seconds_per_frame": elapsed / len(sdr_cache),
        "application_seconds": application_seconds,
        "application_seconds_per_frame": application_seconds / len(sdr_cache),
        "video": video,
        "hdr_centre_max_abs_residual": centre_residual,
    }


def macro_aggregate(shot_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive equal-shot macro aggregate only; never contains model parameters."""
    require(len(shot_results) == 3, "Aggregate requires exactly three completed shots")
    metric_paths = {
        "deltaE_ITP_mean": ("deltaE_ITP", "mean"),
        "deltaE_ITP_P95": ("deltaE_ITP", "P95"),
        "low_frequency_chroma_residual_mean": ("low_frequency_chroma_residual", "mean"),
        "chroma_absolute_error_mean": ("chroma_absolute_error", "mean"),
        "hue_low_chroma_P50_degrees": ("hue_absolute_error_degrees", "low_chroma_P50"),
        "hue_high_chroma_P90_degrees": ("hue_absolute_error_degrees", "high_chroma_P90"),
        "luminance_MAE_nits": ("luminance_nits", "MAE"),
        "luminance_RMSE_nits": ("luminance_nits", "RMSE"),
        "deltaE2000_auxiliary_mean": ("deltaE2000_auxiliary_203nit", "mean"),
        "deltaE2000_auxiliary_P95": ("deltaE2000_auxiliary_203nit", "P95"),
    }
    output: dict[str, Any] = {"type": "equal-shot descriptive macro mean", "shot_count": len(shot_results), "variants": {}}
    for variant in ("A", "B"):
        variant_result: dict[str, float] = {}
        for label, path in metric_paths.items():
            values = [item["evaluation"]["metrics"][variant]["holdout"][path[0]][path[1]] for item in shot_results]
            variant_result[label] = float(np.mean(values))
        seam_values = []
        for item in shot_results:
            seam_values.extend(value["chroma_step_mean"] for value in item["evaluation"]["seam"][variant].values())
        variant_result["seam_chroma_step_mean"] = float(np.mean(seam_values))
        output["variants"][variant] = variant_result
    output["holdout_improvement_fraction_B_vs_A"] = {
        label: percentage_improvement(output["variants"]["A"][label], output["variants"]["B"][label])
        for label in output["variants"]["A"]
    }
    return output


def format_metric_table(metrics: dict[str, Any]) -> list[str]:
    return [
        "| Metric | A frozen baseline | B + residual |",
        "|---|---:|---:|",
        f"| ΔE_ITP mean | {metrics['A']['deltaE_ITP']['mean']:.6f} | {metrics['B']['deltaE_ITP']['mean']:.6f} |",
        f"| ΔE_ITP P95 | {metrics['A']['deltaE_ITP']['P95']:.6f} | {metrics['B']['deltaE_ITP']['P95']:.6f} |",
        f"| Low-frequency chroma residual | {metrics['A']['low_frequency_chroma_residual']['mean']:.7f} | {metrics['B']['low_frequency_chroma_residual']['mean']:.7f} |",
        f"| Chroma absolute error | {metrics['A']['chroma_absolute_error']['mean']:.7f} | {metrics['B']['chroma_absolute_error']['mean']:.7f} |",
        f"| Low-chroma hue error | {metrics['A']['hue_absolute_error_degrees']['low_chroma_P50']:.4f}° | {metrics['B']['hue_absolute_error_degrees']['low_chroma_P50']:.4f}° |",
        f"| High-chroma hue error | {metrics['A']['hue_absolute_error_degrees']['high_chroma_P90']:.4f}° | {metrics['B']['hue_absolute_error_degrees']['high_chroma_P90']:.4f}° |",
        f"| Luminance MAE | {metrics['A']['luminance_nits']['MAE']:.4f} nits | {metrics['B']['luminance_nits']['MAE']:.4f} nits |",
        f"| Luminance RMSE | {metrics['A']['luminance_nits']['RMSE']:.4f} nits | {metrics['B']['luminance_nits']['RMSE']:.4f} nits |",
        f"| ΔE2000 auxiliary mean | {metrics['A']['deltaE2000_auxiliary_203nit']['mean']:.6f} | {metrics['B']['deltaE2000_auxiliary_203nit']['mean']:.6f} |",
        f"| ΔE2000 auxiliary P95 | {metrics['A']['deltaE2000_auxiliary_203nit']['P95']:.6f} | {metrics['B']['deltaE2000_auxiliary_203nit']['P95']:.6f} |",
    ]


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 16 — per-actual-shot low-frequency chroma-field extension",
        "",
        "The run uses the verified Matrix-08 temporal partition. Every field, alpha, TRAIN threshold, gain record, and metric accumulator is isolated by actual shot. The final aggregate is descriptive only and contains no averaged parameter.",
        "",
    ]
    for item in payload["shots"]:
        plan = item["plan"]
        fit = item["fit"]
        evaluation = item["evaluation"]
        visual = evaluation["visual_artifacts"]
        lines.extend(
            [
                f"## {plan['shot_id']} — frames [{plan['interval'][0]}, {plan['interval'][1]}) ({plan['frame_count']} frames)",
                "",
                f"- `alpha_t`: raw `{fit['alpha_t']['raw']:+.7f}`, shrunk `{fit['alpha_t']['shrunk']:+.7f}`, applied `{fit['alpha_t']['applied']:+.7f}`.",
                f"- `alpha_p`: raw `{fit['alpha_p']['raw']:+.7f}`, shrunk `{fit['alpha_p']['shrunk']:+.7f}`, applied `{fit['alpha_p']['applied']:+.7f}`.",
                f"- TRAIN-derived HDR base-chroma thresholds: P50 `{fit['train_reference_chroma_thresholds']['P50_low']:.7f}`, P90 `{fit['train_reference_chroma_thresholds']['P90_high']:.7f}`.",
                f"- Fit time `{fit['fit']['seconds']:.3f}s`; application `{evaluation['timing']['application_seconds_per_frame']:.6f}s/frame` during evaluation.",
                f"- Pre-visual gate: **{item['previsual_verdict']['previsual_status']}**; manual visual review remains required.",
                "",
                "### TRAIN metrics",
                *format_metric_table({"A": evaluation["metrics"]["A"]["train"], "B": evaluation["metrics"]["B"]["train"]}),
                "",
                "### HOLDOUT metrics",
                *format_metric_table({"A": evaluation["metrics"]["A"]["holdout"], "B": evaluation["metrics"]["B"]["holdout"]}),
                "",
                "### Seam",
                "| Quantity | A | B |",
                "|---|---:|---:|",
            ]
        )
        for side in ("top", "bottom"):
            a = evaluation["seam"]["A"][side]
            b = evaluation["seam"]["B"][side]
            lines.extend(
                [
                    f"| {side} chroma step | {a['chroma_step_mean']:.7f} | {b['chroma_step_mean']:.7f} |",
                    f"| {side} hue step | {a['hue_step_degrees_mean']:.4f}° | {b['hue_step_degrees_mean']:.4f}° |",
                    f"| {side} luminance step | {a['luminance_step_nits_mean']:.4f} nits | {b['luminance_step_nits_mean']:.4f} nits |",
                ]
            )
        lines.extend(
            [
                "",
                "### Visual",
                f"- Composite contact sheet: `{visual['contact_sheet']}`",
                f"- A/B low-frequency residual maps: `{visual['lowpass_residual_A']}`, `{visual['lowpass_residual_B']}`",
                f"- Amplified exact-composite B−A residual (HDR centre must be zero): `{visual['amplified_composite_difference']}`",
                f"- Top/bottom seam context: `{visual['seam_map']}`",
                f"- Deterministic HDR-reference-only highest-chroma window: `{visual['highest_chroma_window']}`",
                f"- Visual manifest / manual target checklist: `{visual['visual_manifest']}`",
                "",
            ]
        )
    aggregate = payload["aggregate"]
    lines.extend(
        [
            "## Aggregate — only after per-shot reports",
            "",
            "This is an equal-shot descriptive macro mean. It does not average or select gain, field coefficients, thresholds, `alpha_t`, or `alpha_p`.",
            "",
            "| Holdout metric | A macro mean | B macro mean | B-vs-A improvement |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, a_value in aggregate["variants"]["A"].items():
        b_value = aggregate["variants"]["B"][key]
        improvement = aggregate["holdout_improvement_fraction_B_vs_A"][key]
        text = "n/a" if improvement is None else f"{improvement * 100:+.2f}%"
        lines.append(f"| {key} | {a_value:.7f} | {b_value:.7f} | {text} |")
    lines.extend(
        [
            "",
            "## Full Matrix-08 review outputs",
            "",
            f"- A: `{payload['full_interval_render']['video']['A_baseline']['path']}`",
            f"- B: `{payload['full_interval_render']['video']['B_extended']['path']}`",
            f"- A vs B: `{payload['full_interval_render']['video']['A_vs_B']['path']}`",
            f"- Full render: `{payload['full_interval_render']['timing_seconds']:.3f}s`; application `{payload['full_interval_render']['application_seconds_per_frame']:.6f}s/frame`.",
            "",
            "No additional material was run after Matrix-08.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_preflight(
    output_dir: Path,
    force_cpu: bool,
) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any], list[ShotPlan], SamplingPlan, FrozenBaseline, dict[str, Any]]:
    backend, config, sdr_cache, hdr_cache, context, plans, lock = load_context(output_dir, force_cpu)
    sampling = make_sampling_plan(config, lock)
    baseline = load_frozen_baseline(config, backend)
    width, height = config.om_size
    _, y1, x2, y2 = config.overlap
    require(sdr_cache.shape == (186, height, width, 3), f"Unexpected SDR cache shape: {sdr_cache.shape}")
    require(hdr_cache.shape == (186, y2 - y1, x2, 3), f"Unexpected HDR cache shape: {hdr_cache.shape}")

    first_plan = plans[0]
    sdr = backend.asarray(sdr_cache[first_plan.cache_start])
    gain_info = v1.measure_gain(
        backend,
        config,
        sdr_cache[first_plan.cache_start : first_plan.cache_end],
        hdr_cache[first_plan.cache_start : first_plan.cache_end],
        stride=1,
    )
    require(gain_info["frames_used"] == first_plan.frame_count, "Preflight gain crossed the first shot boundary")
    gain = backend.asarray(om.gain_field(config, gain_info))
    a, _ = baseline_rgb(backend, config, sdr, gain, baseline)
    zero_fit = ShotFit(
        plan=first_plan,
        gain_info={},
        gain_gpu=gain,
        coefficient_ct=np.zeros(6),
        coefficient_cp=np.zeros(6),
        field_ct_gpu=backend.asarray(np.zeros((height, width), dtype=np.float32)),
        field_cp_gpu=backend.asarray(np.zeros((height, width), dtype=np.float32)),
        alpha_t_raw=0.0,
        alpha_p_raw=0.0,
        alpha_t_shrunk=0.0,
        alpha_p_shrunk=0.0,
        alpha_t=0.0,
        alpha_p=0.0,
        threshold_p50=0.0,
        threshold_p90=0.0,
        fit_info={},
    )
    zero, _ = apply_extension(backend, a, zero_fit)
    require(float(backend.tohost(backend.xp.max(backend.xp.abs(zero - a)))) == 0.0, "alpha=0 does not reproduce A")
    preflight = {
        "schema": "openmatte-hdr-phase16-per-shot-preflight/v1",
        "status": "PASS",
        "context": context,
        "spatial_sampling": sampling.manifest,
        "frozen_baseline": baseline.provenance,
        "invariants": {
            "cache_reused_without_build": True,
            "shot_count": len(plans),
            "cache_partition": [[plan.cache_start, plan.cache_end] for plan in plans],
            "alpha_zero_equals_A": True,
            "cross_shot_fit_disallowed": True,
            "holdout_never_enters_fit": True,
            "luminance_ICtCp_I_copied_before_rgb_clip": True,
            "hdr_centre_exact_copy_contract": True,
            "preflight_gain_scope": first_plan.shot_id,
        },
    }
    write_json(output_dir / "preflight.json", preflight)
    return backend, config, sdr_cache, hdr_cache, context, plans, sampling, baseline, lock


def validate_completed_payload(payload: dict[str, Any], sampling: SamplingPlan) -> None:
    require(payload["status"] == "COMPLETED_MATRIX08_ONLY", "Unexpected completion status")
    require([item["plan"]["shot_id"] for item in payload["shots"]] == [item[0] for item in EXPECTED_SHOTS], "Output shot order changed")
    require(payload["aggregate"]["type"] == "equal-shot descriptive macro mean", "Wrong aggregate type")
    require(payload["aggregate"]["shot_count"] == 3, "Wrong aggregate shot count")
    require(not any("alpha" in key.lower() for key in payload["aggregate"].keys()), "Aggregate contains a model parameter")

    for item, (_, start, end) in zip(payload["shots"], EXPECTED_SHOTS, strict=True):
        plan = item["plan"]
        fit = item["fit"]
        evaluation = item["evaluation"]
        require(plan["interval"] == [start, end], f"Output interval differs for {plan['shot_id']}")
        require(plan["cache_indices"] == [start - EXPECTED_INTERVAL[0], end - EXPECTED_INTERVAL[0]], f"Output cache slice differs for {plan['shot_id']}")
        require(fit["gain"]["frames_used"] == plan["frame_count"], f"Gain scope differs for {plan['shot_id']}")
        require(fit["fit"]["frames_used"] == plan["frame_count"], f"Fit scope differs for {plan['shot_id']}")
        require(fit["fit"]["holdout_samples_used"] == 0, f"HOLDOUT entered fit for {plan['shot_id']}")
        require(fit["fit"]["train_total_samples"] == sampling.train_positions.size * plan["frame_count"], f"Wrong TRAIN total for {plan['shot_id']}")
        require(fit["fit"]["train_reference_chroma_thresholds"]["sample_count"] == sampling.train_positions.size * plan["frame_count"], f"Wrong threshold scope for {plan['shot_id']}")
        require(abs(fit["alpha_t"]["applied"]) <= 0.25 and abs(fit["alpha_p"]["applied"]) <= 0.25, f"Alpha bound exceeded for {plan['shot_id']}")
        require(evaluation["hdr_centre_max_abs_residual"] == {"A": 0.0, "B": 0.0}, f"HDR centre changed for {plan['shot_id']}")
        for variant in ("A", "B"):
            require(evaluation["metrics"][variant]["train"]["sample_count"] == sampling.train_positions.size * plan["frame_count"], f"Wrong TRAIN metrics for {plan['shot_id']}")
            require(evaluation["metrics"][variant]["holdout"]["sample_count"] == sampling.holdout_positions.size * plan["frame_count"], f"Wrong HOLDOUT metrics for {plan['shot_id']}")
        artifacts = evaluation["visual_artifacts"]
        for artifact_key in (
            "contact_sheet",
            "lowpass_residual_A",
            "lowpass_residual_B",
            "amplified_composite_difference",
            "seam_map",
            "highest_chroma_window",
            "visual_manifest",
        ):
            require(Path(artifacts[artifact_key]).is_file(), f"Missing {artifact_key} for {plan['shot_id']}")
        require(artifacts["high_chroma_selection"]["source"] == "HDR reference only", f"Wrong high-chroma source for {plan['shot_id']}")

    full_render = payload["full_interval_render"]
    require(full_render["hdr_centre_max_abs_residual"] == {"A": 0.0, "B": 0.0}, "Full HDR centre changed")
    require(set(full_render["video"]) == {"A_baseline", "B_extended", "A_vs_B"}, "Wrong video output set")
    for info in full_render["video"].values():
        require(Path(info["path"]).is_file(), "Missing final review video")
        require(info["frames"] == 186, "Final review video frame count changed")
        require(info["bytes"] > 0, "Final review video is empty")


def main() -> int:
    parser = argparse.ArgumentParser(prog="run-chroma-field-extension")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-fit", action="store_true", help="Required to perform the approved Matrix-08 Phase 16 fit/render")
    parser.add_argument("--cpu", action="store_true", help="Force NumPy/OpenCV instead of the available CuPy GPU backend")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend, config, sdr_cache, hdr_cache, context, plans, sampling, baseline, lock = run_preflight(output_dir, args.cpu)
    print(f"Phase 16 preflight PASS: backend {backend.name}; cache {context['cache']['path']}")
    for plan in plans:
        print(f"  {plan.shot_id}: [{plan.start_frame}, {plan.end_frame_exclusive}) = {plan.frame_count} frames")
    if not args.run_fit:
        print("Fit deliberately not run. Use --run-fit only after reviewing preflight.")
        return 0

    total_started = time.perf_counter()
    shot_results: list[dict[str, Any]] = []
    fits: list[ShotFit] = []
    for plan in plans:
        print(f"\nFitting {plan.shot_id} independently...")
        fit = fit_one_shot(backend, config, sdr_cache, hdr_cache, plan, sampling, baseline, lock)
        fits.append(fit)
        print(f"  alpha_t={fit.alpha_t:+.7f}; alpha_p={fit.alpha_p:+.7f}; fit={fit.fit_info['seconds']:.1f}s")
        shot_dir = output_dir / "shots" / plan.shot_id
        evaluation = evaluate_one_shot(backend, config, sdr_cache, hdr_cache, fit, sampling, baseline, shot_dir / "visual")
        result = {
            "schema": "openmatte-hdr-phase16-per-shot-result/v1",
            "plan": shot_summary(plan),
            "frozen_baseline": baseline.provenance,
            "fit": serializable_fit(fit),
            "evaluation": evaluation,
        }
        result["previsual_verdict"] = previsual_verdict(result, fit)
        write_json(shot_dir / "phase16_result.json", result)
        shot_results.append(result)
        holdout = evaluation["metrics"]
        print(
            "  HOLDOUT low-frequency chroma "
            f"A={holdout['A']['holdout']['low_frequency_chroma_residual']['mean']:.7f}; "
            f"B={holdout['B']['holdout']['low_frequency_chroma_residual']['mean']:.7f}"
        )

    print("\nRendering the exact-HDR-centre Matrix-08 A/B review videos...")
    full_render = render_full_interval(backend, config, sdr_cache, hdr_cache, fits, baseline, output_dir / "video")
    aggregate = macro_aggregate(shot_results)
    payload = {
        "schema": "openmatte-hdr-phase16-matrix08-per-shot/v1",
        "status": "COMPLETED_MATRIX08_ONLY",
        "context": context,
        "spatial_sampling": sampling.manifest,
        "frozen_baseline": baseline.provenance,
        "shots": shot_results,
        "aggregate": aggregate,
        "full_interval_render": full_render,
        "timing_seconds": {"total": time.perf_counter() - total_started},
        "hard_stop": "Matrix-08 completed; no other shot, model, parameter sweep, LUT, MMR, temporal method, or AI was run.",
    }
    validate_completed_payload(payload, sampling)
    write_json(output_dir / "phase16_results.json", payload)
    (output_dir / "PHASE16_REPORT.md").write_text(markdown_report(payload), encoding="utf-8")
    print(f"\nCompleted: {output_dir / 'phase16_results.json'}")
    print(output_dir / "PHASE16_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
