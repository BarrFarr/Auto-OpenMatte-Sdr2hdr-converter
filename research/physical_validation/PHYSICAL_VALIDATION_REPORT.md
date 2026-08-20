# Physical Validation Report — Matrix 73367 / 73348

## Scope and safety
This isolated experiment uses one pre-extracted P2.32 Matrix anchor only. It did not decode a film, modify P2 artifacts, run P2.34, run a production render, or integrate either model into production. Model fitting only receives the SDR/HDR common region; the full SDR Open Matte is used only after fitting.

## Inputs and geometry
- HDR frame: `73367`; Open Matte frame: `73348`.
- Verified overlap in Open Matte coordinates: `[0, 140, 1920, 940]`; geometry confidence `1.0`.
- Input files were programmatically resolved within the experimental root and are recorded in `metrics.json`.
- Full Open Matte: 1920×1080. HDR was scaled from 3840×1600 to the 1920×800 common region.

## Color pipeline
- SDR source: `rgb48le`-derived `uint16` RGB code values, BT.709 primaries/matrix, modeled with BT.1886-style `code**2.4` EOTF.
- SDR fit/apply RGB: linear BT.2020 after the documented 709→2020 matrix.
- HDR reference: `rgb48le`-derived `uint16` RGB code values, BT.2020 primaries, ST.2084/PQ transfer; PQ EOTF is applied once.
- Model domain: linear BT.2020 RGB; normalized HDR `1.0 = 10,000 nits`.
- Scientific arrays are linear BT.2020 nits (`.npy`); scientific PNGs are PQ-coded 16-bit. Display previews are logarithmically tone-mapped and are not HDR scientific outputs.
- Reconstructed-vs-saved P2.32 luminance consistency is reported in `metrics.json`. The OpenCV bilinear resize is not expected to be bit-identical to the original FFmpeg bilinear resize.

## Model fit
| Model | Parameters | CPU fit s | CPU apply s | Common MAE (nits) | Common RMSE (nits) |
|---|---:|---:|---:|---:|---:|
| Model E | 12 | 0.203 | 0.063 | 9.5171 | 18.4240 |
| Model G | 29 | 0.586 | 0.106 | 22.9814 | 56.8441 |

Full fitted parameter values, objectives, regularization, warnings, output-range diagnostics, luminance ranges, ΔE2000, ΔEICtCp, chroma, hue, and seam proxies are in `metrics.json` and `fitted_parameters.json`.

## Common-region quality
The metrics above are **common-region reconstruction quality**, not evidence for objective correctness in the unseen Open Matte extension. ΔE2000 is computed as a documented peak-relative Rec.2020→XYZ D65→Lab diagnostic; ΔEICtCp is also reported for HDR-native color comparison.

## Seam analysis
`10_ModelE_Seam.png` and `11_ModelG_Seam.png` show each model prediction and an HDR-center composite around the top/bottom crop edges. Numeric seam values measure HDR-center versus predicted-extension discontinuities. Because Matrix has no full HDR Open Matte reference, these are continuity proxies only, not ground-truth extension errors.

| Model | Top seam mean / P95 luma (nits) | Bottom seam mean / P95 luma (nits) | Top / bottom hue discontinuity (°) |
|---|---:|---:|---:|
| Model E | 8.508 / 28.185 | 7.892 / 24.707 | 28.166 / 38.400 |
| Model G | 10.708 / 32.176 | 7.902 / 26.012 | 34.670 / 40.657 |

## Visual findings
The corrected 8-bit display panels use a log-to-1,000-nit preview transform and are only for review; `.pq16.png` and `.linear_bt2020_nits.npy` remain the scientific deliverables. Direct review of `comparison_full_A_B_C_D.png` shows a substantial yellow/red chromatic shift and contrast exaggeration in the full-frame extensions for **both** models relative to the neutral HDR center; the crop boundaries are visibly apparent in `10_ModelE_Seam.png` and `11_ModelG_Seam.png`. Model E has lower numeric discontinuities, but neither transformed Open Matte is visually coherent enough to be accepted as an HDR extension on this anchor. This visual result is qualitative; no physical HDR Open Matte reference exists outside the crop.

## Synthetic hidden-region test
A deterministic full HDR Open Matte was created, converted to SDR, and then restricted to a central HDR crop for fitting. This creates known full-frame ground truth.

| Model | Synthetic common MAE (nits) | Synthetic hidden-region MAE (nits) | Hidden RMSE (nits) | Hidden mean ΔE2000 |
|---|---:|---:|---:|---:|
| Model E | 14.0820 | 12.5435 | 19.1965 | 5.3676 |
| Model G | 94.8115 | 31.6391 | 67.4617 | 5.4712 |

This controlled test is supportive only: its SDR creation function is known and does not establish transfer to real unseen imagery.

## CUDA and performance
CUDA status: `True`. Research Models E/G are NumPy/SciPy CPU implementations; no CUDA fit/apply path exists, so no CUDA model benchmark was run. CPU timing above is the only model timing measured.

## Final decision A–F
**A — Common-region adequacy.** Model E is adequate only for **luminance** reconstruction in this limited low/mid-range anchor: MAE is 9.5171 nits and P95 is 42.0198 nits. It does not adequately reproduce common-region RGB appearance: the corrected panel shows a strong warm chromatic shift, and the anchor has no samples at or above 500 nits. Therefore Model E fails an overall physical-image acceptance criterion despite its luminance score.

**B — Value of Model G's extra parameters.** No. Model G's 29 parameters are not justified here versus Model E's 12: it has worse MAE/RMSE/P95 (22.9814/56.8441/174.0634 nits), longer CPU fit/apply time, and slightly higher mean ΔE2000 (4.5585 vs 4.1245).

**C — Coherent grading outside HDR.** No for this anchor. The corrected full-frame and seam panels visibly show yellow/red chromatic shift, contrast exaggeration, and crop-boundary mismatch in both extensions. Model E is less discontinuous numerically but is not visually acceptable as an HDR extension; lack of HDR ground truth outside the overlap remains an additional limitation.

**D — Observed failure modes.** Both models exhibit global warm/yellow-red grading drift outside the reference crop, visible seams, large hue error, and crop-boundary hue discontinuity. Model G additionally has materially larger 10–100 and 100–500 nit errors. The anchor contains no 500+ nit reference pixels, leaving shoulder/highlight behavior untested.

**E — Minimum sufficient complexity.** Twelve parameters (Model E) are the minimum sufficient tested complexity only for common-region reconstruction on this single anchor; neither tested complexity is sufficient for visually acceptable physical extension.

**F — Next action.** Do not advance either model toward production or full-video processing. Model E may be retained solely as the lower-complexity baseline for a separately authorized diagnostic real-anchor study aimed at correcting the extension color/continuity failure; do not alter P2 artifacts.

## Limitations and recommendation
- The Matrix anchor is one temporally selected frame, not independently verified scene ground truth.
- The fitting mask is exactly the verified SDR∩HDR overlap (1,536,000 pixels); neither model sees HDR information outside it during fitting.
- Full fitted parameter values and fitting metadata are preserved in `fitted_parameters.json` and `metrics.json`; Model E uses 12 fitted controls, while Model G uses 29.
- No HDR ground truth exists for the physical Open Matte extension; the result can only claim visual/continuity assessment there.
- Model E exposes no optimizer convergence object; Model G uses separate analytical/statistical components rather than one global RGB loss.
- OpenEXR could not be encoded by this OpenCV build. Scientific outputs are instead full-precision linear-BT.2020-nit `.npy` arrays and 16-bit PQ PNGs.
- Any output above 10,000 nits is retained in scientific arrays and clipped only for PQ PNG serialization.
- The recommendation is research-only: Model E may proceed to another isolated anchor, with no production integration.
