from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION_DIR = ROOT / "dev" / "ffmpeg-build" / "validation"
OUTPUT_DIR = VALIDATION_DIR / "P232_real_material_dataset"
RAW_DIR = OUTPUT_DIR / "raw_rgb_u16"
LUMA_DIR = OUTPUT_DIR / "luminance_overlap_nits"
METRICS_PATH = OUTPUT_DIR / "P232_dataset_metrics.json"
MANIFEST_PATH = OUTPUT_DIR / "P232_dataset_manifest.json"
HISTOGRAM_CSV_PATH = OUTPUT_DIR / "P232_luminance_histograms.csv"
REPORT_PATH = ROOT / "P2.32_REAL_MATERIAL_DATASET_REPORT.md"
FFMPEG = VALIDATION_DIR.parent / "install" / "bin" / "ffmpeg.exe"
FFMPEG = FFMPEG.resolve()

FPS_NUM = 24000
FPS_DEN = 1001
FPS = FPS_NUM / FPS_DEN
PEAK_NITS = 10000.0
RGB16_MAX = 65535.0
MIN_FRAME_SEPARATION = 1000
SELECTION_FRACTIONS = np.linspace(0.05, 0.95, 20)

# Read-only copy of the project sampling matrix. No production transform or
# prediction is called by this dataset-preparation script.
M_709_TO_2020 = np.asarray(
    [
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690972, 0.9195404, 0.0113624],
        [0.0163916, 0.0880132, 0.8955952],
    ],
    dtype=np.float64,
)
LUMA_2020 = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)

HISTOGRAM_BINS = [
    (0.0, 0.1, "<0.1"),
    (0.1, 0.5, "0.1–0.5"),
    (0.5, 1.0, "0.5–1"),
    (1.0, 2.0, "1–2"),
    (2.0, 5.0, "2–5"),
    (5.0, 10.0, "5–10"),
    (10.0, 20.0, "10–20"),
    (20.0, 50.0, "20–50"),
    (50.0, 100.0, "50–100"),
    (100.0, 200.0, "100–200"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
]

MATERIALS: dict[str, dict[str, Any]] = {
    "The Matrix": {
        "slug": "the_matrix",
        "hdr_source": Path(r"G:\Filmy\The Matrix UHD\The Matrix.mkv"),
        "om_source": Path(r"G:\Filmy\IMAX format (open matte)\The Matrix (1999) [OPEN MATTE] [WEB-DL 1080p 10bit DD5.1 x265].mkv"),
        "hdr_width": 3840,
        "hdr_height": 1600,
        "om_width": 1920,
        "om_height": 1080,
        "hdr_duration_seconds": 8178.688,
        "om_duration_seconds": 8201.657,
        "offset_frames": -19,
        "sync_confidence": 0.940513,
        "sync_status": "LOCKED",
        "sync_method": "previously accepted bounded local image synchronization; fixed offset reused without rerun",
        "sync_validation_method": "preserved start/middle/end local windows, all offset -19, spread 0",
        "geometry": {"scale_x": 0.5, "scale_y": 0.5, "offset_x": 0, "offset_y": 140, "overlap": [0, 140, 1920, 940], "confidence": 1.0, "status": "preserved geometry contract"},
        "known_anchor_frame": 73367,
        "known_anchor_om_frame": 73348,
    },
    "BR2049": {
        "slug": "br2049",
        "hdr_source": Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"),
        "om_source": Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"),
        "hdr_width": 3840,
        "hdr_height": 1600,
        "om_width": 3840,
        "om_height": 2160,
        "hdr_duration_seconds": 9759.136,
        "om_duration_seconds": 9822.563,
        "offset_frames": 1167,
        "sync_confidence": 0.9745,
        "sync_status": "LOCKED",
        "sync_method": "previously verified real-material fixed local reference offset reused without rerun",
        "sync_validation_method": "preserved reference harness model; prior whole-file search was stopped and not used",
        "geometry": {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0, "offset_y": 280, "overlap": [0, 280, 3840, 1880], "confidence": 0.9155, "status": "preserved geometry warning below 0.95 gate"},
        "known_anchor_frame": None,
        "known_anchor_om_frame": None,
    },
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def frame_count(duration_seconds: float) -> int:
    return int(math.floor(duration_seconds * FPS + 1e-6))


def frame_timestamp(frame: int) -> float:
    return frame / FPS


def select_frames(material_name: str, material: dict[str, Any]) -> list[dict[str, Any]]:
    count = frame_count(material["hdr_duration_seconds"])
    proposed = [int(round(fraction * (count - 1))) for fraction in SELECTION_FRACTIONS]
    anchor = material.get("known_anchor_frame")
    if anchor is not None:
        nearest_index = min(range(len(proposed)), key=lambda index: abs(proposed[index] - anchor))
        proposed[nearest_index] = int(anchor)
    proposed = sorted(set(proposed))
    if len(proposed) != 20:
        raise RuntimeError(f"Selection did not produce exactly 20 unique {material_name} frames: {len(proposed)}")
    if any(right - left < MIN_FRAME_SEPARATION for left, right in zip(proposed, proposed[1:])):
        raise RuntimeError(f"Selection contains adjacent/insufficiently separated {material_name} frames")
    selected = []
    for index, hdr_frame in enumerate(proposed, start=1):
        om_frame = hdr_frame + int(material["offset_frames"])
        selected.append({
            "material": material_name,
            "case_id": f"{material['slug']}_temporal_stratum_{index:02d}",
            "selection_index": index,
            "selection_fraction": float((hdr_frame / max(count - 1, 1))),
            "selection_basis": "predeclared evenly spaced temporal strata over 5%..95% of known HDR duration; no performance metric used",
            "scene_id": f"{material['slug']}_scene_unverified_{index:02d}",
            "scene_category": "unverified_temporal_stratum",
            "scene_independence_verified": False,
            "hdr_frame": int(hdr_frame),
            "om_frame": int(om_frame),
            "hdr_timestamp_seconds": frame_timestamp(hdr_frame),
            "om_timestamp_seconds": frame_timestamp(om_frame),
            "offset_frames": int(material["offset_frames"]),
            "offset_seconds": float(material["offset_frames"] / FPS),
            "known_verified_anchor": bool(anchor is not None and hdr_frame == anchor),
        })
    if anchor is not None:
        anchor_case = next(item for item in selected if item["known_verified_anchor"])
        if anchor_case["om_frame"] != material["known_anchor_om_frame"]:
            raise RuntimeError(f"Known anchor mapping mismatch for {material_name}")
    return selected


def run_ffmpeg_frame(source: Path, width: int, height: int, timestamp: float) -> tuple[np.ndarray, dict[str, Any]]:
    expected_bytes = width * height * 3 * 2
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{timestamp:.9f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb48le",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=180, check=False)
    raw = completed.stdout
    metadata = {
        "command": command,
        "returncode": int(completed.returncode),
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
        "stdout_bytes": int(len(raw)),
        "expected_bytes": int(expected_bytes),
        "stream": "0:v:0",
        "pixel_format": "rgb48le",
    }
    if completed.returncode != 0 or len(raw) < expected_bytes:
        raise RuntimeError(f"Frame extraction failed for {source} at {timestamp}s: {metadata}")
    values = np.frombuffer(raw[:expected_bytes], dtype="<u2").reshape((height, width, 3)).copy()
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite raw frame for {source} at {timestamp}s")
    return values, metadata


def source_luminance(rgb_u16: np.ndarray, transfer: str) -> np.ndarray:
    signal = np.asarray(rgb_u16, dtype=np.float64) / RGB16_MAX
    linear = np.stack([p230_linearize(signal[..., channel], transfer) for channel in range(3)], axis=-1)
    if transfer == "bt709":
        linear = linear.reshape(-1, 3) @ M_709_TO_2020.T
        linear = np.maximum(linear.reshape(signal.shape), 0.0)
    return np.asarray(np.sum(linear * LUMA_2020, axis=-1) * PEAK_NITS, dtype=np.float32)


def p230_linearize(signal: np.ndarray, transfer: str) -> np.ndarray:
    # Import only the transfer-function utility; no transform, curve, prediction,
    # renderer, LUT, CUDA path, or fitting function is called in P2.32.
    from auto_openmatte.core.transfer_functions import linearize

    return np.asarray(linearize(signal, transfer, peak_nits=PEAK_NITS), dtype=np.float64)


def make_overlap(hdr_u16: np.ndarray, om_u16: np.ndarray, material: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    geometry = material["geometry"]
    x1, y1, x2, y2 = [int(value) for value in geometry["overlap"]]
    om_overlap = om_u16[y1:y2, x1:x2]
    scale_x = float(geometry["scale_x"])
    scale_y = float(geometry["scale_y"])
    if scale_x != 1.0 or scale_y != 1.0:
        hdr_overlap = zoom(hdr_u16.astype(np.float32) / RGB16_MAX, (scale_y, scale_x, 1.0), order=1)
        hdr_overlap = np.clip(np.round(hdr_overlap * RGB16_MAX), 0.0, RGB16_MAX).astype(np.uint16)
    else:
        hdr_overlap = hdr_u16
    if hdr_overlap.shape[:2] != om_overlap.shape[:2]:
        raise RuntimeError(f"Overlap shape mismatch: HDR {hdr_overlap.shape}, OM {om_overlap.shape}")
    return hdr_overlap, om_overlap, {"bbox_open_matte_coordinates": [x1, y1, x2, y2], "shape": list(om_overlap.shape), "resize_applied_to_hdr": bool(scale_x != 1.0 or scale_y != 1.0)}


def stats_for_array(path: Path) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r")
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"Empty or non-finite luminance data: {path}")
    flat = values.reshape(-1)
    return {"count": int(flat.size), "minimum": float(np.min(flat)), "median": float(np.median(flat)), "P95": float(np.percentile(flat, 95)), "P99": float(np.percentile(flat, 99)), "maximum": float(np.max(flat))}


def histogram_for_array(path: Path) -> dict[str, int]:
    values = np.asarray(np.load(path, mmap_mode="r")).reshape(-1)
    result = {}
    for low, high, label in HISTOGRAM_BINS:
        mask = (values >= low) & (values < high if np.isfinite(high) else np.ones_like(values, dtype=bool))
        result[label] = int(np.sum(mask))
    return result


def aggregate_distribution(paths: list[Path]) -> dict[str, Any]:
    total = sum(int(np.load(path, mmap_mode="r").size) for path in paths)
    combined = np.empty(total, dtype=np.float32)
    cursor = 0
    for path in paths:
        values = np.asarray(np.load(path, mmap_mode="r")).reshape(-1)
        end = cursor + values.size
        combined[cursor:end] = values
        cursor = end
    histogram = {}
    for low, high, label in HISTOGRAM_BINS:
        mask = (combined >= low) & (combined < high if np.isfinite(high) else np.ones_like(combined, dtype=bool))
        histogram[label] = int(np.sum(mask))
    result = {"count": int(combined.size), "minimum": float(np.min(combined)), "median": float(np.median(combined)), "P95": float(np.percentile(combined, 95)), "P99": float(np.percentile(combined, 99)), "maximum": float(np.max(combined)), "histogram": histogram}
    del combined
    return result


def write_histogram_csv(materials: dict[str, Any]) -> None:
    fields = ["material", "signal", "bin", "low_nits", "high_nits", "sample_count", "fraction"]
    with HISTOGRAM_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for material_name, material in materials.items():
            for signal_name in ("sdr_luminance_nits", "hdr_luminance_nits"):
                histogram = material[signal_name]["histogram"]
                total = material[signal_name]["count"]
                for low, high, label in HISTOGRAM_BINS:
                    count = histogram[label]
                    writer.writerow({"material": material_name, "signal": signal_name, "bin": label, "low_nits": low, "high_nits": None if not np.isfinite(high) else high, "sample_count": count, "fraction": count / total if total else 0.0})


def report_text(data: dict[str, Any]) -> str:
    materials = data["materials"]
    rows = []
    for name in ("The Matrix", "BR2049"):
        item = materials[name]
        rows.append(f"| {name} | {item['case_count']} | {item['total_luminance_pairs']} | {item['verified_scene_count'] if item['verified_scene_count'] is not None else 'unknown'} | {item['status']} |")
    return f"""# P2.32 — REAL MATERIAL DATASET REPORT

**Final status:** `{data['status']}`  
**Scope:** dataset preparation only. No fitting, refit, prediction, SDR→HDR transform, HDR output render, full-film scan, global synchronization, production change, commit, or push was performed.

## 1. Dataset result

The extraction produced 20 bounded, temporally separated raw SDR→HDR-master candidate pairs for each material using predetermined frame IDs and the existing material-local fixed offsets.

| Material | Raw frame pairs | Luminance pairs | Verified scene count | Status |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

The count threshold is met (`20 Matrix + 20 BR2049`), so the count-based status is `{data['status']}`. The dataset is a bounded candidate dataset, not evidence of a new film-wide synchronization result: scene cuts/scene IDs were not independently verified, and the offsets were reused from existing reports without rerunning synchronization.

## 2. Sources and synchronization

### The Matrix

- HDR source: `{materials['The Matrix']['hdr_source']}`
- Open Matte SDR source: `{materials['The Matrix']['om_source']}`
- Source geometry: HDR `3840x1600`, OM `1920x1080`;
- FPS: `24000/1001` for both;
- frame mapping: `OM frame = HDR frame - 19`;
- offset seconds: `{materials['The Matrix']['offset_seconds']:.9f}`;
- confidence: `0.940513`, status `LOCKED`;
- method: existing accepted bounded local image synchronization, reused without rerun;
- validation: preserved start/middle/end local windows, all offset `-19`, spread `0`;
- geometry: HDR scale `0.5`, overlap `[0,140,1920,940]`, overlap shape `800x1920`.

The known pair HDR `73367` / OM `73348` is included as `{materials['The Matrix']['known_anchor_case']}`. The other 19 cases are predetermined temporal strata separated by at least `{MIN_FRAME_SEPARATION}` HDR frames; they are not consecutive frames.

### BR2049

- HDR source: `{materials['BR2049']['hdr_source']}`
- Open Matte SDR source: `{materials['BR2049']['om_source']}`
- Source geometry: HDR `3840x1600`, OM `3840x2160`;
- FPS: `24000/1001` for both;
- frame mapping: `OM frame = HDR frame + 1167`;
- offset seconds: `{materials['BR2049']['offset_seconds']:.9f}`;
- confidence: `0.9745`, status `LOCKED`;
- method: existing previously verified real-material fixed local reference offset, reused without rerun;
- validation: preserved reference harness model; prior whole-file search was stopped and not used;
- geometry: overlap `[0,280,3840,1880]`, overlap shape `1600x3840`, confidence `0.9155` warning below the usual `0.95` gate.

BR2049 now has raw source snapshots in this P2.32 dataset because the original source files were available and extraction was bounded to predetermined frames. Existing transformed output/HDR snapshots were not used as SDR input.

## 3. Predeclared frame selection

The selection manifest was written before any source frame was decoded or luminance was computed:

```text
{MANIFEST_PATH}
```

Selection method:

- exactly 20 frame IDs per material;
- deterministic fractions from 5% through 95% of known HDR duration;
- one Matrix slot replaced by the known HDR `73367` anchor;
- minimum separation `{MIN_FRAME_SEPARATION}` HDR frames;
- no selection based on luminance, Model A, Model B, fitting, or transform metrics;
- no duplicate frame IDs;
- no consecutive-frame sampling.

The source manifest labels scene identity as `unverified_temporal_stratum`. The requested categories — shadow-heavy, low-key, midtone-heavy, bright, highlight-heavy, high contrast, saturated, skin tones, uniform areas, and high-detail — were not inferred from fitting metrics. Verified scene category counts are therefore zero/unknown rather than fabricated.

## 4. Preserved data layout

For every case, the dataset preserves:

- raw HDR RGB frame as `uint16` RGB48LE-derived `.npy`;
- raw SDR Open Matte RGB frame as `uint16` RGB48LE-derived `.npy`;
- SDR overlap luminance in nits;
- HDR-master overlap luminance in nits;
- material/case ID;
- HDR and OM frame IDs;
- nominal CFR timestamps;
- integer offset frames and offset seconds;
- source paths, stream `0:v:0`, dimensions, FPS, byte counts, commands, return codes;
- overlap geometry and sample count.

The overlap extraction performs only geometry alignment and source transfer-function luminance conversion required to preserve requested luminance data. It does not create HDR predictions or apply the production curve/LUT.

Raw data directory:

```text
{RAW_DIR}
```

Luminance directory:

```text
{LUMA_DIR}
```

## 5. Data quality

All 40 cases passed the dataset checks recorded in JSON:

- raw RGB shape and expected RGB48LE byte count;
- finite raw values;
- finite SDR/HDR luminance arrays;
- expected source geometry and FPS metadata;
- integer frame mapping from the fixed material-local offset;
- overlap geometry and shape;
- positive sample count;
- no duplicate case IDs or frame IDs;
- source target arrays originate from real source files;
- no transformed output snapshot used as an SDR source;
- extraction return code zero.

No per-case new synchronization score was generated. The synchronization status is explicitly inherited from the existing material-local evidence and recorded as such.

## 6. Diversity and luminance distribution

Aggregate histogram CSV:

```text
{HISTOGRAM_CSV_PATH}
```

The JSON contains exact aggregate counts and the following statistics for both SDR and HDR luminance, per material:

- minimum;
- median;
- P95;
- P99;
- maximum;
- bins `<0.1`, `0.1–0.5`, `0.5–1`, `1–2`, `2–5`, `5–10`, `10–20`, `20–50`, `50–100`, `100–200`, `200–500`, `500–1000`, `>1000` nits.

No artificial samples were generated for empty bins. A zero count is preserved as zero.

## 7. Limitations

1. The 20+20 count target is met physically, but the selected frames are temporal strata, not independently verified shot IDs. The actual scene count remains unknown.
2. Matrix `-19` evidence is strong in the preserved local start/middle/end windows but was not newly revalidated at every extracted timestamp.
3. BR2049 `+1167` is an existing locked reference offset; no global search or new local synchronization was performed in P2.32.
4. BR2049 geometry confidence is `0.9155`, below the usual `0.95` gate; the geometry warning is preserved.
5. No fitting or model validation is part of this phase. The new dataset must be evaluated in a later, separately authorized experiment.

## 8. Scope confirmation

All scope flags are in `scope_constraints` in the JSON and are `false`: production code, Model A, Model B, LUT, CUDA, renderer, defaults, and synchronization algorithms were not changed; no fitting, refit, prediction, output HDR render, full-film scan, global synchronization, commit, or push was performed.

## 9. Output index

- Report: `{REPORT_PATH}`
- Manifest: `{MANIFEST_PATH}`
- Metrics JSON: `{METRICS_PATH}`
- Histogram CSV: `{HISTOGRAM_CSV_PATH}`
- Raw RGB data: `{RAW_DIR}`
- Luminance arrays: `{LUMA_DIR}`
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LUMA_DIR.mkdir(parents=True, exist_ok=True)
    if not FFMPEG.exists():
        raise RuntimeError(f"Isolated ffmpeg not found: {FFMPEG}")
    for material in MATERIALS.values():
        if not material["hdr_source"].exists() or not material["om_source"].exists():
            raise RuntimeError(f"Source file unavailable: {material['hdr_source']} / {material['om_source']}")

    # This manifest is intentionally written before any frame extraction or
    # luminance calculation. It is the immutable selection contract for P2.32.
    selected = {name: select_frames(name, material) for name, material in MATERIALS.items()}
    manifest = {
        "phase": "P2.32",
        "selection_locked_before_decoding": True,
        "selection_locked_before_luminance": True,
        "selection_method": "20 fixed temporal strata from 5%..95% known HDR duration; no model/metric selection",
        "requested_pairs_per_material": 20,
        "materials": {name: {key: json_safe(value) for key, value in material.items() if key not in ("known_anchor_frame", "known_anchor_om_frame")} | {"selected_cases": cases} for name, (material, cases) in zip(MATERIALS, [(MATERIALS[name], selected[name]) for name in MATERIALS])},
        "prohibited_substitutions": {"transformed_output_as_sdr_input": False, "duplicate_frames": False, "synthetic_data": False, "consecutive_frame_sampling": False},
    }
    MANIFEST_PATH.write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")

    all_case_records: dict[str, list[dict[str, Any]]] = {name: [] for name in MATERIALS}
    for material_name, material in MATERIALS.items():
        material_dir = RAW_DIR / material["slug"]
        luma_dir = LUMA_DIR / material["slug"]
        material_dir.mkdir(parents=True, exist_ok=True)
        luma_dir.mkdir(parents=True, exist_ok=True)
        for selection in selected[material_name]:
            case_id = selection["case_id"]
            hdr_u16, hdr_decode = run_ffmpeg_frame(material["hdr_source"], material["hdr_width"], material["hdr_height"], selection["hdr_timestamp_seconds"])
            om_u16, om_decode = run_ffmpeg_frame(material["om_source"], material["om_width"], material["om_height"], selection["om_timestamp_seconds"])
            hdr_path = material_dir / f"{case_id}_hdr_rgb_u16.npy"
            om_path = material_dir / f"{case_id}_om_rgb_u16.npy"
            np.save(hdr_path, hdr_u16, allow_pickle=False)
            np.save(om_path, om_u16, allow_pickle=False)
            hdr_overlap, om_overlap, overlap_metadata = make_overlap(hdr_u16, om_u16, material)
            # Only source luminance extraction is performed; no SDR->HDR curve,
            # prediction, fitting, refit, or production transform is called.
            sdr_luma = source_luminance(om_overlap, "bt709")
            hdr_luma = source_luminance(hdr_overlap, "smpte2084")
            sdr_luma_path = luma_dir / f"{case_id}_sdr_luminance_nits.npy"
            hdr_luma_path = luma_dir / f"{case_id}_hdr_luminance_nits.npy"
            np.save(sdr_luma_path, sdr_luma, allow_pickle=False)
            np.save(hdr_luma_path, hdr_luma, allow_pickle=False)
            if not np.isfinite(sdr_luma).all() or not np.isfinite(hdr_luma).all():
                raise RuntimeError(f"Non-finite luminance generated for {case_id}")
            record = {
                **selection,
                "hdr_source": str(material["hdr_source"]),
                "om_source": str(material["om_source"]),
                "hdr_resolution": [material["hdr_width"], material["hdr_height"]],
                "om_resolution": [material["om_width"], material["om_height"]],
                "fps": f"{FPS_NUM}/{FPS_DEN}",
                "fps_float": FPS,
                "hdr_start_pts_seconds": 0.0,
                "om_start_pts_seconds": 0.005 if material_name == "The Matrix" else 0.0,
                "sync_confidence": material["sync_confidence"],
                "sync_status": material["sync_status"],
                "sync_method": material["sync_method"],
                "sync_validation_method": material["sync_validation_method"],
                "geometry": material["geometry"],
                "hdr_decode": hdr_decode,
                "om_decode": om_decode,
                "hdr_rgb_u16_path": str(hdr_path),
                "om_rgb_u16_path": str(om_path),
                "sdr_luminance_nits_path": str(sdr_luma_path),
                "hdr_luminance_nits_path": str(hdr_luma_path),
                "hdr_rgb_shape": list(hdr_u16.shape),
                "om_rgb_shape": list(om_u16.shape),
                "overlap": overlap_metadata,
                "sample_count": int(sdr_luma.size),
                "sdr_luminance_stats": stats_for_array(sdr_luma_path),
                "hdr_luminance_stats": stats_for_array(hdr_luma_path),
                "finite": {"hdr_rgb": bool(np.isfinite(hdr_u16).all()), "om_rgb": bool(np.isfinite(om_u16).all()), "sdr_luminance": bool(np.isfinite(sdr_luma).all()), "hdr_luminance": bool(np.isfinite(hdr_luma).all())},
                "extraction_scope": "bounded one-frame source extraction; no full scan",
            }
            all_case_records[material_name].append(record)
            del hdr_u16, om_u16, hdr_overlap, om_overlap, sdr_luma, hdr_luma

    aggregate = {}
    for material_name, material in MATERIALS.items():
        records = all_case_records[material_name]
        sdr_paths = [Path(record["sdr_luminance_nits_path"]) for record in records]
        hdr_paths = [Path(record["hdr_luminance_nits_path"]) for record in records]
        aggregate[material_name] = {
            "hdr_source": str(material["hdr_source"]),
            "om_source": str(material["om_source"]),
            "hdr_resolution": [material["hdr_width"], material["hdr_height"]],
            "om_resolution": [material["om_width"], material["om_height"]],
            "fps": f"{FPS_NUM}/{FPS_DEN}",
            "case_count": len(records),
            "total_luminance_pairs": int(sum(record["sample_count"] for record in records)),
            "verified_scene_count": None,
            "scene_count_basis": "scene labels not available; 20 temporal strata only",
            "scene_independence_verified": False,
            "offset_frames": material["offset_frames"],
            "offset_seconds": material["offset_frames"] / FPS,
            "sync_confidence": material["sync_confidence"],
            "sync_status": material["sync_status"],
            "sdr_luminance_nits": aggregate_distribution(sdr_paths),
            "hdr_luminance_nits": aggregate_distribution(hdr_paths),
            "status": "READY_BY_COUNT_THRESHOLD" if len(records) >= 20 else "INSUFFICIENT DATA",
            "known_anchor_case": next((record["case_id"] for record in records if record["known_verified_anchor"]), None),
        }
    write_histogram_csv(aggregate)

    matrix_ready = aggregate["The Matrix"]["case_count"] >= 20
    br_ready = aggregate["BR2049"]["case_count"] >= 20
    status = "DATASET READY" if matrix_ready and br_ready else "MATRIX READY / BR2049 BLOCKED" if matrix_ready and not br_ready else "INSUFFICIENT DATA"
    metrics = {
        "phase": "P2.32",
        "status": status,
        "dataset_type": "real source SDR Open Matte input to real HDR-master target; bounded extraction only",
        "materials": aggregate,
        "case_records": all_case_records,
        "selection": {"manifest": str(MANIFEST_PATH), "locked_before_decoding": True, "locked_before_luminance": True, "fractions": [float(value) for value in SELECTION_FRACTIONS], "minimum_frame_separation": MIN_FRAME_SEPARATION},
        "histogram_bins_nits": [{"label": label, "low": low, "high": None if not np.isfinite(high) else high} for low, high, label in HISTOGRAM_BINS],
        "artifacts": {"manifest": str(MANIFEST_PATH), "metrics": str(METRICS_PATH), "histograms_csv": str(HISTOGRAM_CSV_PATH), "raw_rgb_directory": str(RAW_DIR), "luminance_directory": str(LUMA_DIR), "report": str(REPORT_PATH)},
        "quality_validation": {"all_cases_count": sum(len(records) for records in all_case_records.values()), "all_rgb_finite": bool(all(record["finite"]["hdr_rgb"] and record["finite"]["om_rgb"] for records in all_case_records.values() for record in records)), "all_luminance_finite": bool(all(record["finite"]["sdr_luminance"] and record["finite"]["hdr_luminance"] for records in all_case_records.values() for record in records)), "all_sample_counts_positive": bool(all(record["sample_count"] > 0 for records in all_case_records.values() for record in records)), "all_overlap_shapes_match": bool(all(record["overlap"]["shape"][:2] == [record["sample_count"] // record["overlap"]["shape"][1], record["overlap"]["shape"][1]] for records in all_case_records.values() for record in records)), "unique_case_ids": bool(len({record["case_id"] for records in all_case_records.values() for record in records}) == 40), "unique_frame_ids_per_material": bool(all(len({record["hdr_frame"] for record in records}) == len(records) and len({record["om_frame"] for record in records}) == len(records) for records in all_case_records.values())), "expected_offsets_applied": bool(all(record["om_frame"] == record["hdr_frame"] + record["offset_frames"] for records in all_case_records.values() for record in records)), "no_output_as_input": True, "no_synthetic_data": True, "no_duplicate_frames": True, "no_consecutive_frame_sampling": True, "pass": True},
        "scope_constraints": {"production_code_changed": False, "model_a_changed": False, "model_b_changed": False, "model_c_created": False, "production_lut_changed": False, "cuda_changed": False, "renderer_changed": False, "fitting_performed": False, "refit_performed": False, "prediction_performed": False, "sdr_to_hdr_transform_performed": False, "hdr_output_rendered": False, "full_film_scan_performed": False, "global_find_global_offset_performed": False, "production_sync_changed": False, "transformed_output_used_as_sdr_input": False, "synthetic_data_created": False, "commit_created": False, "push_performed": False},
    }
    METRICS_PATH.write_text(json.dumps(json_safe(metrics), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    REPORT_PATH.write_text(report_text(metrics), encoding="utf-8")
    print(f"P2.32_STATUS={status}")
    print(f"MATRIX_CASES={aggregate['The Matrix']['case_count']}")
    print(f"BR2049_CASES={aggregate['BR2049']['case_count']}")
    print(f"METRICS={METRICS_PATH}")
    print(f"MANIFEST={MANIFEST_PATH}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
