# PQ-Code vs Linear-BT.2020 Resize Diagnostic

## Scope
Matrix anchor HDR 73367 / SDR-Open-Matte 73348 only. Geometry is `[0,140,1920,940]`; fitting sees only this `1920×800` / 1,536,000-pixel overlap. No production code, P2 artifact, Model H/G, or video decode was modified or used.

## Controlled pipelines
- **A — current:** HDR PQ code `3840×1600` → OpenCV `INTER_LINEAR` → PQ EOTF → linear BT.2020 `1920×800`.
- **B — linear resize:** HDR PQ code → PQ EOTF at native `3840×1600` → OpenCV `INTER_LINEAR` in linear BT.2020 → `1920×800`.
- SDR source, geometry, bilinear kernel, Model E luminance fitting, ratio scaling, outputs, and fit mask are otherwise identical.

## Direct target A/B difference
| Metric A vs B | Value |
|---|---:|
| Luminance MAE / RMSE | 0.015662 / 0.059043 nits |
| Mean / P95 ΔE2000 | 0.001338 / 0.006273 |
| Mean abs chroma / hue error | 0.00002425 / 0.024224° |

## Downstream H0 comparison
| Branch | luma MAE / RMSE vs its target | mean ΔE2000 | mean abs hue error | top / bottom seam mean nits |
|---|---:|---:|---:|---:|
| H0-A | 8.554624 / 17.722706 | 0.946417 | 20.804764° | 7.039246 / 5.494014 |
| H0-B | 8.557641 / 17.723390 | 0.947067 | 20.803656° | 7.049002 / 5.502168 |

## Conclusion
**The resampling order is not materially affecting this Matrix anchor or its H0 baseline.** Direct A/B reference change is only mean/P95 ΔE2000 `0.001338` / `0.006273`. H0-A versus H0-B is mean/P95 ΔE2000 `0.001924` / `0.006199`, with only `0.017624`-nit mean luminance difference. Using B changes H0's mean ΔE2000 against its own matching target by `+0.000650` (+0.069%), from `0.946417` to `0.947067`.

Linear-domain filtering is still the physically appropriate ordering when resampling scene-linear RGB, but at this exact 2× geometry/content its observed impact is negligible compared with H0's roughly 0.946 mean ΔE2000 residual. Therefore it does **not** explain a visible localized blue/purple artifact that emerges only after the later H1 3×3 matrix transform. The direct reference hue maximum is not used as evidence because hue is ill-conditioned around near-neutral pixels; its robust mean/P95 values are the relevant observations.

This A/B test isolates **reference resampling order only**. It cannot establish which branch is a display-master ground truth; the P2.32 source provides encoded pixels rather than a separate linear-resampled reference. Scientific arrays are float32 linear-BT.2020 nits; PQ PNGs are uint16; preview panels are display-only uint8 log views.
