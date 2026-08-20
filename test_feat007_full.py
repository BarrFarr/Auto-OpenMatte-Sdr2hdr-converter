"""FEAT-007 full real-material test: 3 segments + global + comparison."""
import sys, time
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.sampling import (
    sample_overlap_luminance, train_validation_split, compute_diagnostics
)

HDR = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")

hdr_source = inspect_source(HDR)
select_video_stream(hdr_source)
om_source = inspect_source(OM)
select_video_stream(om_source)
fps = hdr_source.selected_stream.fps

sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                 frame_locked=True, offset_seconds=1167/fps)
geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                     overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
config = ColorConfig(samples_per_shot=30, luminance_bins=256,
                     low_percentile=1.0, high_percentile=99.0)

results = []

def test_segment(start_s, end_s, label):
    sf = int(round(start_s * fps))
    ef = int(round(end_s * fps))
    shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
    t0 = time.perf_counter()
    samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                       config=config, n_samples=30, proxy_width=960, peak_nits=10000.0)
    t1 = time.perf_counter()
    if samples.n_valid_pairs < 1000:
        print(f"{label}: FAILED - only {samples.n_valid_pairs} valid pairs")
        return None
    train, val = train_validation_split(samples, train_ratio=0.8, seed=42)
    curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=config)
    diag = compute_diagnostics(samples, curve, train_samples=train, val_samples=val)
    hdr_nits = samples.hdr_luminance * 10000.0
    sdr = samples.sdr_luminance
    print(f"\n{label}:")
    print(f"  valid={samples.n_valid_pairs:,} frames={samples.n_frames} time={t1-t0:.1f}s")
    print(f"  confidence={diag.confidence:.4f} monotonic={diag.monotonic}")
    print(f"  train_MAE={diag.train_mae:.6f} val_MAE={diag.val_mae:.6f}")
    print(f"  SDR: P1={np.percentile(sdr,1):.4f} P50={np.percentile(sdr,50):.4f} P99={np.percentile(sdr,99):.4f}")
    print(f"  HDR nits: P1={np.percentile(hdr_nits,1):.1f} P50={np.percentile(hdr_nits,50):.1f} "
          f"P99={np.percentile(hdr_nits,99):.1f} MAX={np.max(hdr_nits):.1f}")
    results.append({"label": label, "samples": samples, "curve": curve, "diag": diag})
    return True

print("=== FEAT-007 MULTI-SEGMENT LUMINANCE TEST ===")
test_segment(2700.0, 2730.0, "TEST1_45m00")
test_segment(4500.0, 4530.0, "TEST2_1h15m")
test_segment(2220.0, 2250.0, "TEST3_37m00")

# Global
if len(results) == 3:
    all_sdr = np.concatenate([r["samples"].sdr_luminance for r in results])
    all_hdr = np.concatenate([r["samples"].hdr_luminance for r in results])
    rng = np.random.default_rng(42)
    n = len(all_sdr)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    gc = estimate_luminance_curve(all_sdr[idx[:split]], all_hdr[idx[:split]], config=config)
    mv = apply_luminance_curve(all_sdr[idx[split:]], gc)
    gv_mae = float(np.mean(np.abs(mv - all_hdr[idx[split:]])))
    gv_corr = float(np.corrcoef(mv, all_hdr[idx[split:]])[0, 1])
    print(f"\nGLOBAL TRANSFORM:")
    print(f"  samples={n:,} curve_pts={len(gc)} val_MAE={gv_mae:.6f} confidence={gv_corr:.4f}")
    for r in results:
        m = apply_luminance_curve(r["samples"].sdr_luminance, gc)
        mae = float(np.mean(np.abs(m - r["samples"].hdr_luminance)))
        print(f"  {r['label']} per-segment MAE={mae:.6f}")

print("\n=== DONE ===")
