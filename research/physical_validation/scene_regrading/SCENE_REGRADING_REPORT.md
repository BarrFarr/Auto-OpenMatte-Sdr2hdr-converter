# Scene Regrading Report — Model H

## Model definition and color space
H0 is Model E fitted on **overlap luminance only**, followed by safe ratio scaling: `RGB_base = RGB_sdr * Y_target / max(Y_sdr, eps)`. H1 then applies exactly one scene-local 3×3 matrix in linear BT.2020: `RGB_hdr = M_scene @ RGB_base`. HDR uses one ST.2084 EOTF; SDR uses BT.1886-style `code**2.4` then BT.709→BT.2020. HDR outside `[0,140,1920,940]` was never loaded for fitting because it does not exist in the P2.32 input.

## Fit objective and regularization
The matrix was fitted on deterministic spatially separated overlap samples, selected by held-out multi-metric objective, not raw RGB MSE alone:
`0.35*scaled luminance MSE + 0.20*scaled ICtCp chroma MSE + 0.20*scaled non-neutral hue MSE + 0.25*scaled DeltaE2000 MSE + lambda*||M-I||_F^2`.
Tested regularization lambdas: `[0.001, 0.01, 0.1]`. Constraints A/B/C respectively tested unbounded regularized, positive diagonal, and limited cross-channel mixing. Candidates with condition number >30 or |determinant|<0.05 were rejected.

## Matrix anchor: Matrix 73367 / 73348
- Verified overlap: `[0,140,1920,940]`; geometry confidence 1.0; fit pixels 1,536,000.
- Selected configuration: `A_unconstrained_regularized`, lambda `0.001`.
- Matrix: `[[ 0.6315911  0.4723792 -0.256661 ]
 [ 0.1644166  0.6947885 -0.0376797]
 [ 0.2805169 -0.3087712  0.9101822]]`.
- Determinant `0.3794291`, condition number `2.97778`, `||M-I||_F` `0.8534901`.
- Effective scene parameters: 10 compact Model E luminance controls + 9 RGB-matrix controls = 19. The sampled LUT ordinates are not counted as independent scene controls.

| Variant | luma MAE / RMSE nits | mean / P95 ΔE2000 | mean |chroma| | mean |hue| ° |
|---|---:|---:|---:|---:|
| H0 ratio scaling | 8.5546 / 17.7227 | 0.9464 / 3.7763 | 0.02815 | 20.8048 |
| H1 3×3 regrading | 8.3467 / 17.2917 | 0.9120 / 3.8783 | 0.02026 | 19.5258 |

H0/H1 seam data, full candidate coefficients/objectives, range diagnostics, and preview/scientific image locations are in `results/matrix_73367_73348/metrics.json`.

## Seam and CPU performance
| Variant | top seam mean / P95 nits | bottom seam mean / P95 nits |
|---|---:|---:|
| H0 | 7.0392 / 24.8688 | 5.4940 / 20.9775 |
| H1 | 6.5453 / 22.9487 | 5.2469 / 20.7529 |

Model E luminance-only fit took `0.0744` s CPU; selected H1 matrix fitting took `4.2459` s CPU on its deterministic overlap samples; H1 full-frame application took `0.0972` s CPU. CUDA was intentionally not used because correctness is the subject of this reference experiment.

## Physical visual review
The H1 comparison and seam panels show only a modest scene-wide visual change from H0. The dark garment contains localized blue/purple speckling in the H1 preview, an unacceptable artifact until its source is understood. Consequently, the small held-out metric reduction is not accepted as evidence of an improved physical full-frame regrade. H1 reduces the top/bottom continuity proxies slightly, but those proxies are not ground truth for the unseen Open Matte.

## Synthetic hidden Open Matte
| Variant | common mean ΔE2000 | hidden mean ΔE2000 | hidden luma MAE nits |
|---|---:|---:|---:|
| H0 | 2.2577 | 1.8934 | 2.1712 |
| H1 | 0.0279 | 0.0628 | 1.8856 |

## Multi-temporal-stratum and stability
{
  "performed": false
}

## Decision A–H
A. **No material acceptance over H0 on Matrix.** H1 improves mean ΔE2000 from `0.9464` to `0.9120` (3.64%), below the predeclared 5% gate.
B. **Representationally yes, experimentally only marginally.** The stable 3×3 changes RGB relationships and reduces mean hue error from `20.8048°` to `19.5258°`, but not enough to establish robust HDR studio regrading.
C. **No demonstrated physical-image improvement.** The visual panel is close to H0 and contains localized blue/purple artifacts in a dark garment; the small numerical gain is therefore not sufficient for physical acceptance.
D. **Synthetic yes; physical unseen region unproven.** H1 substantially improves the synthetic hidden region, where the transform is known global. For the real Open Matte extension, only seam continuity proxies exist and they are not ground truth.
E. **Parameters necessary:** H0's 10 compact luminance controls remain the supported baseline. The 9 additional matrix controls (19 total) are not justified by this single real anchor.
F. **A single 3×3 is not established as sufficient.** Its stable selected matrix gives only a below-gate local gain, no verified multi-scene confirmation, and no same-shot stability evidence.
G. **No basis yet for a luminance-dependent matrix.** That next hypothesis requires demonstrated residual hue/chroma behavior that changes systematically by luminance after H1. This experiment instead finds a small aggregate gain plus a physical artifact, so it must not expand parameters.
H. **Implementation compatibility is retained.** H0 plus a 3×3 multiply is compact, local CPU/GPU-compatible, AI-free, and LUT-free. This is a feasibility statement only; it does not authorize production integration.

## Limitations, failure cases, and recommendation
- **Failure case:** the selected unconstrained-but-stable matrix (`||M-I||_F=0.8535`) produces localized blue/purple artifacts in a dark garment preview. Numerical stability does not guarantee visually plausible regrading.
- Only stratum 08 is a known verified Matrix anchor. The other predeclared temporal records were not run because the 5% H1 gate failed; they are not verified independent scenes in any event.
- P2.32 contains no same-shot adjacent-frame groups, so same-shot matrix stability is unavailable.
- No full-video processing, P2 modification, production changes, Model G work, or Model H production integration occurred.
- **Recommendation:** retain H0 ratio scaling as the validated research baseline. Reject H1 as an accepted scene regrading model for now; do not add M(Y) until a separate residual-by-luminance diagnostic identifies a specific failure it would solve.
