# OpenMatte HDR — chroma-field GPU milestone

**Date:** 2026-08-20
**Status:** reproducible Matrix-08 engineering checkpoint; not a claim of final visual acceptance.

## What is frozen

This checkpoint preserves the first practical accelerated implementation of the bounded low-frequency chroma-field idea.

- The original v0.1 luminance path remains fixed: boundary-conditioned gain from a 48 px seam band, Gaussian base sigma 16 px, bounded channel detail ratio, exponent 0.85.
- HDR centre rows are copied directly from the HDR master, with only the existing internal feather bands; they are never regraded or regenerated.
- Only the low-frequency chroma plane is changed: ICtCp Ct/Cp is Gaussian-smoothed at sigma 16 px, transformed by a spatial complex factor, and recombined with untouched chroma detail and intensity.
- The spatial factor is deliberately limited to a degree-2 polynomial in x at each of the two seams, vertically blended toward a scene-level factor: **12 real fitted controls for the whole scene**.
- There is no temporal state, AI, LUT, MMR, segmentation, or extension-side HDR ground truth.

## Reproducible source contract

| Item | Value |
|---|---|
| HDR source | `G:\Filmy\The Matrix UHD\The Matrix.mkv` |
| SDR Open Matte source | `G:\Filmy\IMAX format (open matte)\The Matrix (1999) [OPEN MATTE] [WEB-DL 1080p 10bit DD5.1 x265].mkv` |
| HDR frame interval | `[73274, 73460)` |
| SDR offset | `-19` frames |
| Frame count | `186` at `24000/1001` |
| Output / SDR geometry | `1920×1080` |
| HDR overlap inside SDR | `[0, 140, 1920, 940]` |
| First HDR / SDR raw-frame SHA-256 | `38739765…D72454E` / `19027AE…B911CE` |
| Last HDR / SDR raw-frame SHA-256 | `6DB9BA0A…79A76E6` / `DAE36C24…15F9768` |

The full hashes, coefficients, timings and configuration are stored in [`matrix08_gpu_milestone.json`](matrix08_gpu_milestone.json). The decode cache provenance is `tools/openmatte_hdr/cache/73274_186/manifest.json`.

## Accelerated execution result

Implementation:

- `tools/openmatte_hdr/fastcore.py` — float32 decode-once memmap cache, CuPy backend, multicore CPU fallback configuration, threaded encoder-pipe writes.
- `tools/openmatte_hdr/fast_chroma_field.py` — equivalent bounded chroma-field execution over the cache.

Hardware path: `cupy/gpu` on the NVIDIA GPU; direct HDR10 review encoding uses NVENC.

| Stage | Time |
|---|---:|
| Decode once to cache | `154.149 s` |
| Gain measurement (stride 3) | `1.870 s` |
| Chroma measurement (stride 3) | `1.509 s` |
| Full two-variant + side-by-side render | `12.828 s` (`0.069 s/frame`) |
| Cold-cache end-to-end total | `170.443 s` |

The cache occupies `8.0566 GB`; subsequent compatible runs avoid the 154-second decode step. GPU/CPU floating point execution is visually near-equivalent but intentionally not claimed bit-identical: measured transfer/blur differences can create occasional one-code 16-bit differences.

## Matrix-08 evidence

Baseline reference artifact: `tools/openmatte_hdr/out/chroma_field_poc/`
Accelerated artifact: `tools/openmatte_hdr/out/chroma_field_fast/`

The accelerated run was verified with `ffprobe`:

- `hevc`, `1920×1080`, `yuv420p10le`, 186 decoded frames;
- limited (`tv`) range, `bt2020nc`, `smpte2084`, `bt2020`;
- review media: `v01_luminance_only_hdr10.mkv`, `v02_luminance_plus_chroma_hdr10.mkv`, `side_by_side_hdr10.mkv`.

On the full CPU-reference PoC, low-pass seam-band chroma disagreement changed from `0.021148` to `0.012714` (**39.9% reduction**) without negative RGB after the correction. The fast full run shows the same final residual within the intended GPU/stride tolerance (`0.012711`).

This is only a diagnostic improvement, not a visual acceptance test. The existing average hue-seam diagnostic worsened on the CPU reference (top `0.615° → 0.735°`, bottom `0.740° → 0.777°`); the fast scalar is not directly comparable because it currently includes near-neutral pixels. Human HDR review remains the actual criterion, particularly for rice paper, skin, walls, dark clothing and saturated elements.

## Explicit limitations retained

1. The Open Matte extensions have no HDR ground truth, so overlap agreement cannot prove a portable creative grade.
2. The field is static per shot. It avoids temporal flicker by design but cannot follow a true temporal grade change.
3. `fast_chroma_field.py` currently samples both gain and chroma measurements at stride 3; use `--stride 1` when exact full-frame measurement parity is required.
4. A cache provenance check currently verifies the key, dimensions and stored first/last hashes, but does not rehash sources on every reuse. Delete the cache after replacing a source at the same path.
5. This checkpoint does not add model flexibility until a no-regression diagnostic and visual comparison demonstrate a benefit.

## Next bounded iteration

The next iteration must stay inside the same low-frequency chroma-field family. It will retain the 12-control spatial field and test **one additional bounded global hue-damping scalar** `λ ∈ [0, 1]`:

`z_damped(x, y) = |z(x, y)| · exp(i · λ · arg(z(x, y)))`

`λ=1` is the frozen milestone; `λ=0` retains the fitted saturation correction while suppressing the fitted hue rotation. The scalar will be selected only from held-out overlap samples/frames against circular hue error, while rejecting candidates that regress low-pass chroma residual, RGB range safety, or HDR visual review. It adds one scalar, no spatial degrees of freedom, no LUT, no MMR, no temporal state and no AI.

## Outcome of the next bounded iterations

Two constrained follow-ups were completed after this checkpoint.

1. **Hue damping guardrail:** the one-scalar `lambda` experiment was implemented in `tools/openmatte_hdr/fast_chroma_field_hue_damped.py`. Its GPU smoke test selected `lambda = 1.00`; damping hue increased both held-out low-pass residual and held-out boundary hue disagreement. It is rejected as a default-model direction rather than rendered as a redundant full-shot replacement.
2. **Intensity-conditioned residual:** `tools/openmatte_hdr/fast_chroma_field_intensity.py` retains the 12 spatial controls and adds only two shot-global slopes: log saturation versus sigma-16 predicted ICtCp intensity and hue versus the same intensity. With an interleaved 148/38 training/held-out split on full Matrix-08, the saturation-only ablation reduced held-out low-pass chroma disagreement from `0.01270496` to `0.01240557` (**2.36%**) with no measurable low-pass hue-step regression. It is accepted as the bounded PoC default pending visual review.

The hue-plus-saturation ablation achieved the lower held-out residual (`0.01230515`, **3.15%**) but increased held-out low-pass hue step by `0.158°`, beyond the predeclared `0.02°` limit. It remains a clearly labelled visual-review candidate, not the default.

Durable details, full metrics, outputs and validation are in [`INTENSITY_ITERATION.md`](INTENSITY_ITERATION.md) and [`matrix08_intensity_iteration.json`](matrix08_intensity_iteration.json). The accepted default review pair is at `tools/openmatte_hdr/out/chroma_field_intensity_default/`; the non-default joint review pair is at `tools/openmatte_hdr/out/chroma_field_intensity/`.
