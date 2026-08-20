# Color Pipeline Diagnostic — Matrix 73367 / 73348

## Scope and isolation
This research-only causal experiment uses exactly the verified P2.32 Matrix anchor: HDR frame 73367, Open Matte frame 73348, overlap `[0, 140, 1920, 940]` (1,536,000 pixels, geometry confidence 1.0). It reads existing pre-extracted arrays only; it does not decode video, modify P2, alter production, use Model G, or use HDR outside the overlap for any fitted control.

SDR is decoded as `uint16 code -> BT.1886 code**2.4 -> linear BT.709 -> linear BT.2020`. HDR is decoded once as `uint16 code -> resize to common geometry -> PQ EOTF -> linear BT.2020`, where 1.0 equals 10,000 nits. All image panels are display-only log previews; scientific arrays are linear-BT.2020 nits and PQ16 PNGs.

## Causal stages
- **T1:** SDR color-management path only; no grading.
- **T2:** luminance-only Model E fit, with the original SDR chroma deviation left unchanged in absolute linear units.
- **T3:** the requested safe ratio scaling: `RGB_new = RGB_old * Y_new / max(Y_old, eps)`; zero-luminance pixels become neutral at `Y_new`.
- **T4:** T3 plus one overlap-fitted global chroma scalar `k` around `Y_new`.
- **T5:** T3 plus the minimum luminance-dependent chroma model, `k(Y_new)=clip(a+b*normalized(Y_new), 0, 4)`, fitted only on the overlap.

## Metric comparison
| Test | Luma MAE nits | Luma RMSE nits | mean ΔE2000 | mean |chroma error| | mean |hue error| ° | top / bottom seam nits |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 718.342 | 1390.685 | 16.724 | 0.05435 | 20.954 | 521.849 / 700.885 |
| T2 | 31.879 | 60.771 | 10.094 | 0.26756 | 37.272 | 36.069 / 35.366 |
| T3 | 8.555 | 17.723 | 0.946 | 0.02815 | 20.805 | 7.039 / 5.494 |
| T4 | 8.555 | 17.723 | 0.996 | 0.04032 | 21.092 | 7.039 / 5.494 |
| T5 | 8.555 | 17.723 | 0.935 | 0.02036 | 20.805 | 7.039 / 5.494 |

## Base color-management checks
- BT.709→BT.2020 white-neutral preservation error: `1.800e-06`.
- Gray-ramp channel-spread maximum after the primary conversion: `1.600e-06`.
- Matrix has no negative primary response for a neutral input, and the implementation uses the canonical research conversion helper directly.

## Fitted controls
- Model E was fit **only on luminance** with the 1,536,000-pixel overlap; its native chroma fit/apply path was not used.
- T4 global `k`: `1.166081` (unclamped `1.166081`).
- T5 `a=0.808891`, `b=1.234088`; common-region clamp fractions low/high: `0.000000%` / `0.000000%`.

## Answers
1. **Base BT.709→BT.2020 pipeline:** correct. White-neutral preservation error is `1.800e-06` and the gray-ramp channel spread is `1.600e-06`. The canonical conversion therefore cannot create a systematic yellow/red cast. T1's SDR↔HDR difference reflects differently mastered source material, not a fitted transform or a color-management failure.
2. **Model E luminance mapping:** conditionally correct for this limited common-region anchor when evaluated with a hue-preserving reconstruction. T3, using the same overlap-only Model E LUT as T2, has luminance MAE/RMSE `8.555` / `17.723` nits. T2's `31.879`-nit MAE is worse because additive chroma plus non-negative clipping destroys the requested luminance, not because the LUT changed.
3. **Current chroma treatment:** yes, it is the source of the cast. T2 isolates the additive reconstruction `Y_new + C_old`: mean ΔE2000 `10.094`, mean absolute chroma error `0.26756`, and mean absolute hue error `37.272°`. The previous Model E chroma path is the same additive family (`Y_new + scale(Y_old)*C_old`), so this is a causal architectural failure, not evidence against the primary matrix.
4. **Ratio scaling:** yes, it removes the major color failure while holding the exact same Model E luminance LUT fixed. T3 reaches mean ΔE2000 `0.946`, mean absolute chroma error `0.02815`, and hue `20.805°`, versus T2 `37.272°`. Its visible result and seam proxy are correspondingly much closer to the HDR center.
5. **Minimum additional chroma model:** **none beyond ratio scaling** for this anchor. T4 global `k` worsens ΔE2000 to `0.996` and hue to `21.092°`. T5's two controls make only a marginal ΔE2000 change to `0.935` and leave hue at `20.805°`; a scalar saturation function cannot rotate hue. No extra chroma parameter is justified before testing more anchors.
6. **Root cause classification:** mathematical/model-related chroma reconstruction, not a BT.709→BT.2020 implementation error. Replacing luminance while retaining an absolute SDR chroma deviation causes clipping, large hue error, and the yellow/red cast. Multiplicative ratio scaling preserves RGB proportions and removes the principal failure.

## Limitations
- HDR reference exists only inside the overlap. Seam metrics are continuity proxies, not extension ground truth.
- No 500+ nit common-reference pixels exist in this anchor, so high-luminance behavior remains unproven.
- OpenCV `INTER_LINEAR` is used for the documented P2.32 geometry approximation; it is not claimed bit-exact to the original FFmpeg resize.
- This result does not authorize Model E development, production integration, or full-video processing.
