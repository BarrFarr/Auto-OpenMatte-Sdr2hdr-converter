"""Strict P2.1–P2.6 validation.

Diagnostic-only.  The fitter is local to this script so production's current
log-domain implementation cannot leak into the Linear or PQ reference paths.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import gc
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import pq_eotf, pq_oetf
from auto_openmatte.processing.sampling import sample_overlap_luminance
from auto_openmatte.utils.math_utils import fit_monotonic_spline, percentile_bins

HDR = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUT = ROOT / "test_output" / "p2_strict_validation"
PEAK, SEED = 10000.0, 42
SEGMENTS = (("37m", 2220.0, 2250.0), ("45m", 2700.0, 2730.0), ("75m", 4500.0, 4530.0))
FINE_BANDS = ((0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95),
              (95, 99), (99, 99.5), (99.5, 99.9), (99.9, 100))


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def forward(values: np.ndarray, domain: str, epsilon: float) -> np.ndarray:
    if domain == "linear":
        return values
    if domain == "pq":
        return pq_oetf(values * PEAK)
    return np.log10(values * PEAK + epsilon)


def inverse(values: np.ndarray, domain: str, epsilon: float) -> np.ndarray:
    if domain == "linear":
        return values
    if domain == "pq":
        return pq_eotf(values) / PEAK
    return np.maximum(10.0**values - epsilon, 0.0) / PEAK


def fit(sdr: np.ndarray, hdr: np.ndarray, domain: str, config: ColorConfig, epsilon: float = 1e-6):
    """One implementation for all domains: same bins, 64 raw PCHIP points, no anchors."""
    x, y = forward(sdr, domain, epsilon), forward(hdr, domain, epsilon)
    centers, medians = percentile_bins(x, y, config.luminance_bins,
                                       (config.low_percentile, config.high_percentile))
    if len(centers) < 5:
        raise RuntimeError(f"too few valid bins for {domain}")
    x, y = fit_monotonic_spline(centers, medians)
    index = np.linspace(0, len(x) - 1, min(64, len(x)), dtype=int)
    x, y = x[index], y[index]
    interpolator = PchipInterpolator(x, y, extrapolate=True)

    def apply(values: np.ndarray) -> np.ndarray:
        return inverse(interpolator(forward(values, domain, epsilon)), domain, epsilon)

    return apply, x, y


def mask_for_band(sdr: np.ndarray, low: float, high: float) -> np.ndarray:
    lower = np.percentile(sdr, low)
    if high == 100:
        return sdr >= lower
    return (sdr >= lower) & (sdr < np.percentile(sdr, high))


def metrics(apply, sdr: np.ndarray, hdr: np.ndarray, bands=FINE_BANDS) -> dict:
    prediction = apply(sdr) * PEAK
    target = hdr * PEAK
    error = prediction - target
    answer = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
              "bias": float(error.mean()), "bands": {}}
    for low, high in bands:
        mask = mask_for_band(sdr, low, high)
        answer["bands"][f"P{low:g}-P{high:g}"] = {
            "n": int(mask.sum()), "mae": float(np.abs(error[mask]).mean()),
            "signed": float(error[mask].mean()),
        }
    return answer


def cache_paths(label: str) -> tuple[Path, Path]:
    return OUT / f"{label}_sdr_f32.npy", OUT / f"{label}_hdr_f32.npy"


def legacy_cache_paths(label: str) -> tuple[Path, Path]:
    return OUT / f"{label}_sdr.npy", OUT / f"{label}_hdr.npy"


def collect() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    OUT.mkdir(parents=True, exist_ok=True)
    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # Convert the already-collected float64 cache one segment at a time.  This
    # preserves every paired sample while avoiding the 2x working-set penalty.
    for label, *_ in SEGMENTS:
        sdr_path, hdr_path = cache_paths(label)
        legacy_sdr, legacy_hdr = legacy_cache_paths(label)
        if not sdr_path.exists() and legacy_sdr.exists():
            np.save(sdr_path, np.load(legacy_sdr, mmap_mode="r").astype(np.float32))
        if not hdr_path.exists() and legacy_hdr.exists():
            np.save(hdr_path, np.load(legacy_hdr, mmap_mode="r").astype(np.float32))
    if all(all(path.exists() for path in cache_paths(label)) for label, *_ in SEGMENTS):
        for label, *_ in SEGMENTS:
            sdr_path, hdr_path = cache_paths(label)
            cached[label] = (np.load(sdr_path, mmap_mode="r"), np.load(hdr_path, mmap_mode="r"))
        print("Using cached paired samples.")
        return cached

    hdr_source, om_source = inspect_source(HDR), inspect_source(OM)
    select_video_stream(hdr_source)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps
    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167 / fps)
    geometry = GeometryModel(scale_x=1, scale_y=1, offset_x=0, offset_y=280,
                             overlap_bbox=[0, 280, 3840, 1880], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256, low_percentile=1, high_percentile=99)
    for label, start, end in SEGMENTS:
        start_frame, end_frame = round(start * fps), round(end * fps)
        shot = Shot(0, start_frame, end_frame, start_frame + 1167, end_frame + 1167,
                    end_frame - start_frame)
        sample = sample_overlap_luminance(hdr_source, om_source, sync, geometry, shot,
                                          config=config, n_samples=30, proxy_width=960, rng_seed=SEED)
        sdr_path, hdr_path = cache_paths(label)
        # float32 halves cache and working-set size; it still has far more
        # precision than the decoded 16-bit source samples.
        sample_sdr = sample.sdr_luminance.astype(np.float32)
        sample_hdr = sample.hdr_luminance.astype(np.float32)
        np.save(sdr_path, sample_sdr)
        np.save(hdr_path, sample_hdr)
        cached[label] = (sample_sdr, sample_hdr)
        print(f"{label}: {sample.n_valid_pairs:,} paired samples cached")
    return cached


def split(sdr: np.ndarray, hdr: np.ndarray):
    index = np.random.default_rng(SEED).permutation(len(sdr))
    cut = int(len(index) * 0.8)
    return sdr[index[:cut]], hdr[index[:cut]], sdr[index[cut:]], hdr[index[cut:]], index[cut:]


def main() -> None:
    started = time.perf_counter()
    config = ColorConfig(samples_per_shot=30, luminance_bins=256, low_percentile=1, high_percentile=99)
    print("P2.1–P2.6 strict domain-only validation")
    samples = collect()
    sdr = np.concatenate([samples[label][0] for label, *_ in SEGMENTS], dtype=np.float32)
    hdr = np.concatenate([samples[label][1] for label, *_ in SEGMENTS], dtype=np.float32)
    total_pairs = len(sdr)
    sdr_hash, hdr_hash = digest(sdr), digest(hdr)
    sdr_train, hdr_train, sdr_val, hdr_val, val_index = split(sdr, hdr)
    # The global and per-segment arrays are not needed by aggregate fitting.
    # Drop them before allocating transformed fitting-domain arrays.
    del sdr, hdr
    gc.collect()
    print(f"Shared pairs: {total_pairs:,}; train={len(sdr_train):,}; val={len(sdr_val):,}")

    fitted = {domain: fit(sdr_train, hdr_train, domain, config) for domain in ("linear", "pq", "log")}
    result = {domain: metrics(fitted[domain][0], sdr_val, hdr_val) for domain in fitted}
    print("\nVariant          MAE       RMSE      Bias      P99-P100")
    for domain, value in result.items():
        print(f"{domain:<15} {value['mae']:8.4f} {value['rmse']:9.4f} {value['bias']:9.4f} {value['bands']['P99-P100']['mae']:11.4f}")

    epsilon_report = {}
    for epsilon in (1e-7, 1e-6, 1e-5, 1e-4):
        apply, _, _ = fit(sdr_train, hdr_train, "log", config, epsilon)
        epsilon_report[f"{epsilon:.0e}"] = metrics(apply, sdr_val, hdr_val, ((0, 10), (99, 100)))

    segment_report = {}
    for label, pair in samples.items():
        seg_train_sdr, seg_train_hdr, seg_val_sdr, seg_val_hdr, _ = split(*pair)
        segment_report[label] = {}
        for domain in ("linear", "log"):
            apply, _, _ = fit(seg_train_sdr, seg_train_hdr, domain, config)
            segment_report[label][domain] = metrics(apply, seg_val_sdr, seg_val_hdr, ((99, 100),))

    points = {}
    for domain, (_, x, y) in fitted.items():
        physical_x, physical_y = inverse(x, domain, 1e-6) * PEAK, inverse(y, domain, 1e-6) * PEAK
        points[domain] = [{"internal_x": float(a), "internal_y": float(b),
                           "sdr_domain_nits": float(c), "hdr_nits": float(d)}
                          for a, b, c, d in zip(x, y, physical_x, physical_y)]
    report = {
        "method": "single local fitter; 256 bins; P1-P99; 64 raw PCHIP points; no synthetic endpoints",
        "manifest": {"pairs": int(total_pairs), "train": int(len(sdr_train)), "val": int(len(sdr_val)),
                     "seed": SEED, "sdr_sha256": sdr_hash, "hdr_sha256": hdr_hash,
                     "validation_index_sha256": digest(val_index)},
        "aggregate": result, "epsilon": epsilon_report, "per_segment": segment_report,
        "control_points": points, "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
