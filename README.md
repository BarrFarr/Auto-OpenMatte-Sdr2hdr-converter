# Auto Open-Matte HDR Extender

Automatically combine HDR 21:9 reference video with SDR 16:9 Open Matte video to produce a full 16:9 HDR output.

## Overview

Many films are mastered in HDR at a cinema aspect ratio (e.g., 2.39:1) but also have an SDR "Open Matte" version at 16:9 that reveals additional image content above and below the widescreen frame. This tool combines both sources to create a 16:9 HDR output that preserves the HDR quality of the original while extending the frame using the Open Matte's additional content.

**Key principle:** HDR is always the master reference. The Open Matte content is adapted to match the HDR, never the reverse.

## Features

- **Source Inspection** - Automatically detect video properties, HDR metadata, and frame characteristics
- **Image-Based Synchronization** - Align sources temporally using visual content analysis (no audio)
- **Shot Detection** - Identify scene boundaries and transitions
- **Geometry Alignment** - Sub-pixel alignment of overlapping regions between sources
- **Luminance Mapping** - Fit monotonic tone curves to map SDR luminance to HDR range
- **Color Transform** - Match color characteristics between sources per-shot
- **Seamless Composition** - Blend extended regions with smooth transitions
- **Preview Generation** - Quick visual previews before committing to full render
- **Full HDR Render** - Produce final output with proper HDR10/HLG metadata

## Installation

```bash
pip install -e '.[dev]'
```

### System Requirements

- Python 3.10+
- FFmpeg with HDR support (for video decode/encode)
- FFprobe (typically bundled with FFmpeg)

## Usage

The pipeline runs in three stages: analyze, preview, and render.

### 1. Analyze

Analyze both sources and compute all alignment parameters:

```bash
auto_openmatte analyze \
    --hdr /path/to/hdr_master.mkv \
    --openmatte /path/to/openmatte_sdr.mkv \
    --sync-search 120 \
    --samples-per-shot 30 \
    --alignment-confidence 0.95 \
    --color-confidence 0.90
```

Options:
- `--hdr` - Path to HDR 21:9 reference video (required)
- `--openmatte` - Path to SDR 16:9 Open Matte video (required)
- `--sync-search` - Frame range to search for sync (default: 120)
- `--samples-per-shot` - Frames to sample per shot for analysis (default: 30)
- `--alignment-confidence` - Minimum geometry alignment confidence (default: 0.95)
- `--color-confidence` - Minimum color match confidence (default: 0.90)
- `--debug` - Enable debug output
- `--debug-sync` - Enable sync-specific debug output

### 2. Preview

Generate a quick preview of the composited output:

```bash
auto_openmatte preview --project analysis.json
```

### 3. Render

Render the final full-resolution HDR output:

```bash
auto_openmatte render --project analysis.json --output final_output.mkv
```

## Pipeline Architecture

```
Source Inspection --> Synchronization --> Shot Detection
        |                   |                  |
        v                   v                  v
   Geometry          Luminance Curve     Color Transform
   Alignment            Fitting             Analysis
        |                   |                  |
        +-------------------+------------------+
                            |
                            v
                      Composition
                            |
                            v
                     HDR Output Render
```

Each stage produces confidence scores. The pipeline will halt or warn if confidence drops below thresholds.

## Technology Stack

- **Python 3.10+** - Core language
- **NumPy** - Array operations and image processing math
- **SciPy** - Curve fitting, optimization, and signal processing
- **OpenCV** - Frame extraction, geometric transforms, template matching
- **Pillow** - Image format support and basic operations
- **FFmpeg** - Video decode/encode via subprocess

## Project Structure

```
src/auto_openmatte/
    __init__.py          # Package version
    __main__.py          # python -m entry point
    cli.py               # Argument parsing and CLI
    core/
        models.py        # Dataclass definitions
        config.py        # Constants and defaults
        exceptions.py    # Custom exceptions
    analysis/            # Source inspection, sync, shot detection
    processing/          # Geometry, luminance, color, composition
    pipeline/            # Orchestration logic
    output/              # Preview and render
```

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Run type checker
make typecheck

# Format code
make format

# Run all checks
make all
```

## License

MIT
