# Auto OpenMatte — SDR to HDR Converter

Automatic HDR Open Matte extension and SDR-to-HDR conversion tool.

## Overview

This CLI application combines an HDR (21:9) master with its SDR Open Matte (16:9) counterpart
to produce a full 16:9 HDR output where:

- The center HDR region is preserved as-is (MASTER)
- The extension areas from Open Matte are transformed to match HDR characteristics
- Synchronization, shot detection, geometry, and color matching are fully automatic

## Modes

### Mode A: Extend HDR

Compose a 16:9 HDR from HDR master + Open Matte extension areas:

```bash
auto_openmatte analyze --hdr film_hdr.mkv --openmatte film_openmatte.mkv
auto_openmatte extend --project project.json --output final_extended_hdr.mkv
```

Or in one step:

```bash
auto_openmatte extend --hdr film_hdr.mkv --openmatte film_openmatte.mkv --output final.mkv
```

### Mode B: Convert SDR Open Matte to HDR

Transform the entire Open Matte to HDR using an HDR reference:

```bash
auto_openmatte convert-hdr --input film_openmatte.mkv --reference film_hdr.mkv --output om_hdr.mkv
```

## Pipeline

1. **Inspect** — ffprobe metadata, HDR detection, stream selection
2. **Synchronize** — Frame-offset detection and drift validation
3. **Shot Detection** — Detect cuts on HDR, map to Open Matte
4. **Geometry** — Align HDR within Open Matte frame
5. **Luminance Mapping** — Robust monotonic SDR→HDR luminance transform
6. **Color Matching** — 3x3 correction matrix per shot
7. **Compose/Render** — Frame-locked output generation

## Requirements

- Python ≥ 3.10
- FFmpeg ≥ 5.0 (on PATH)
- NumPy, SciPy, OpenCV

## Installation

```bash
pip install -e ".[dev]"
```

## Key Principles

- **HDR is MASTER** — never modify the HDR reference pixels
- **Frame-locked sync** — constant frame offset, validated for drift
- **No audio** — video processing only
- **No hard-coded aspect ratios** — geometry derived from actual images
- **Deterministic** — classical algorithms, no AI/ML color matching
- **Fail-safe** — confidence scoring, halt on low confidence
