# Minimal Scene-Based MMR Chroma Regrading — Phase 5

## Scope and literature basis
Research-only, one-shot test. It does not modify production, P2.30/P2.33/P2.34, sync, geometry, previous artifacts, or source video. The HDR reference is used only in each pair's SDR∩HDR overlap. Full Open-Matte extensions are produced solely for seam/visual-consistency inspection, never claimed as objectively reconstructed outside overlap.

**A — published conceptual basis.** The local research report identifies scene-local backward reshaping and multivariate regression as relevant concepts. Dolby's multi-channel MMR patent describes predicting a higher-dynamic-range image from a corresponding lower-dynamic-range signal with compact regression and optional cross-products ([US20170264898A1](https://patents.google.com/patent/US20170264898A1/en)).

**B — simplifications.** This is not Dolby metadata/residual coding: there is no bitstream, residual layer, spatial/neighbour predictor, large LUT, multiple luma segments, polynomial square term, or temporal/framewise adaptation. H0 exclusively supplies luminance. MMR predicts only two chroma components.

**C — project decisions.** ICtCp is used because this repository already implements it and exposes chroma/hue metrics. MMR-0/1 use regularized closed-form ridge fits, one deterministic spatial train/holdout split, fixed λ `[0.0001, 0.001, 0.01]`, fixed chroma bound ±`0.5`, and a mandatory 5% ΔE2000 gate. Content was rephrased for compliance with licensing restrictions.

## Baseline and representation
`SDR code → BT.1886 → linear BT.709 → BT.2020 → Model E overlap-luminance fit → Y_target → ratio scaling → RGB_base` is unchanged. Additive chroma is not used.

`RGB_base` and HDR reference are converted from linear BT.2020 **absolute nits** to ICtCp. Input chroma is `(Ct_base,Cp_base)`, target is `(Ct_HDR,Cp_HDR)`, and `Y = Y_target_nits/100`; Ct/Cp are unitless ICtCp components. ICtCp I is copied from RGB_base for inverse conversion, but is not treated as linear luminance. After inverse ICtCp, negative RGB is limited to zero and output RGB is ratio-reimposed to exact Model-E `Y_target`. This keeps luminance fixed and makes final scoring include the actual effects of reimposition.

## Exact models and regularization
MMR-0 (8 coefficients): `Ct' = a0+a1Y+a2Ct+a3Cp`; `Cp' = b0+b1Y+b2Ct+b3Cp`.

MMR-1 (12, only conditionally): MMR-0 plus `a4Y·Ct+a5Y·Cp` and `b4Y·Ct+b5Y·Cp`. Identity/no-op is `[Ct'=Ct, Cp'=Cp]`; ridge pulls coefficients to it. The closed form is `(XᵀX/N + λI)⁻¹(XᵀC/N + λΘ_identity)`. Spatially predeclared train/holdout sizes are `4096`/`2048`. Candidate selection uses holdout, not training, objective `0.20*(luma MAE/10)^2 + 0.40*mean ΔE2000² + 0.25*(|chroma|/0.03)^2 + 0.15*(|hue|/20)^2` — never RGB MSE alone.

## Matrix physical results — 73367 / 73348
| Model | chroma params | fit ΔE | holdout ΔE | full ΔE | full ΔE improvement vs H0 |
|---|---:|---:|---:|---:|---:|
| H0 | 0 | 0.907846 | 1.069008 | 0.946417 | baseline |
| MMR-0 | 8 | 0.841684 | 1.016640 | 0.884702 | +6.521% |
| MMR-1 | 12 | 0.838310 | 1.012894 | 0.881528 | +6.856% |

MMR-0: coefficients `[[-0.0018037 -0.0059452]
 [-0.0634307  0.0308061]
 [ 0.6129245 -0.1784321]
 [-0.0304133  0.4980814]]`; λ `0.0001`; regularized Gram condition `2139.4584`; identity deviation `0.662966`; max |coefficient| `0.612924`; chroma-bound pixel fraction `0.00000000`; inverse-negative pixel fraction `0.00000000`.

MMR-1: coefficients `[[-0.00192   -0.0062681]
 [-0.0609     0.0352561]
 [ 0.6104379 -0.1827916]
 [-0.0301095  0.5018661]
 [ 0.0517972  0.0687095]
 [ 0.0213875 -0.0113101]]`; λ `0.0001`; regularized Gram condition `9793.7949`; identity deviation `0.668730`; max |coefficient| `0.610438`; chroma-bound pixel fraction `0.00000000`; inverse-negative pixel fraction `0.00000000`.

The generated scientific float32 nits, PQ16, preview, common crop, luminance-difference, seam and five-way comparison montage are under `results/matrix_73367_73348/`. Preview is inspection-only; scientific `.npy` is authoritative.

## MMR-0 failure / interaction analysis
MMR-0 numerical gate: `False`. Post-MMR-0 residual chroma/hue relative dispersions over reference luminance bins are `0.252240` / `0.538300`. MMR-1 interaction support is `True` under the predeclared ≥0.25 rule. Bin data and all candidate coefficients/errors are retained in `results/matrix_73367_73348/metrics.json`.

## Synthetic hidden Open Matte
Known full synthetic HDR has a fixed nontrivial 12-coefficient ICtCp Y×C grade, but fit sees only central y=72:288. This tests extrapolation into a known hidden region; it is not evidence that real-film unseen regions have ground truth.

| Variant | common mean ΔE2000 | hidden mean ΔE2000 | hidden luma MAE nits |
|---|---:|---:|---:|
| H0 | 12.036356 | 10.710849 | 0.750744 |
| MMR0 | 2.351603 | 2.327899 | 0.750744 |
| MMR1 | 1.924733 | 1.769187 | 0.750744 |

## Frozen cross-scene validation
| Scene | Model | Params | Fit ΔE | Holdout ΔE | Full-overlap ΔE (vs H0) | Hue | Chroma | Seam | numerical gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| The Matrix the_matrix_temporal_stratum_08 | 12 | 0.8383 | 1.0129 | 0.8815 (+6.86%) | 22.596 | 0.01679 | 7.039/5.494 | True |
| The Matrix the_matrix_temporal_stratum_03 | 12 | 1.1838 | 1.1663 | 1.2236 (+37.92%) | 7.939 | 0.01048 | 5.328/33.532 | True |
| The Matrix the_matrix_temporal_stratum_12 | 12 | 0.8506 | 0.6387 | 0.8083 (+22.16%) | 7.846 | 0.01064 | 4.396/5.773 | True |
| BR2049 br2049_temporal_stratum_01 | 12 | 0.0067 | 0.0077 | 0.0077 (+9.66%) | 27.724 | 0.00375 | 0.063/0.055 | True |

All cross-scene numerical gates pass: `True`. Full-overlap ΔE: mean `0.730277`, median `0.844894`, worst `1.223600`, variance `0.198604`. Coefficient maximum deviation across pairs: `0.374084`. BR2049 remains conditional because geometry confidence is 0.9155.

## Failure analysis and remaining trade-off
MMR-1 passes the stated acceptance gate through held-out/full ΔE2000 improvement, finite bounded output, stable coefficients, no severe visual seam degradation, and the reviewed montage. It is **not** a universal hue improvement: on the Matrix development overlap mean absolute hue error changes from `20.804764°` (H0) to `22.596002°` (MMR-1). At the bottom seam it changes from `19.858665°` to `22.841520°`, while chroma seam improves from `0.022206` to `0.011854` and luma seam is invariant by construction. This is a documented mixed chroma/hue trade-off, not evidence that every hue residual is solved. The preview shows it attenuates rather than introduces the previously prominent purple/cyan garment residual.

## Performance, visual artifacts, and final decision
Matrix H0 luminance fit took `0.0676` s CPU; MMR-0 selected fit took `0.0476` s CPU; MMR-1 selected fit took `0.0474` s CPU. Full output finiteness/bounds, seam continuity, coefficient magnitudes, clip fractions, fit/holdout/full metrics and all λ candidates are persisted in metrics JSON. **Visual montage review:** `The five-way montage shows MMR-1 reducing H0's global yellow/green cast and attenuating the localized purple/cyan dark-garment residual; no new H1-like high-contrast blue/purple speckling is visible. This is a display-preview assessment only, not unseen-region HDR ground truth.` Visual gate pass: `True`.

## Final decision — B — MMR-0 fails but MMR-1 passes and is sufficient
Only the predeclared Y×C extension passed development and every frozen cross-scene numerical gate after MMR-0's residual supported luminance-chroma interaction.

**Why this outcome is necessary:** The decision applies the 5% full/holdout ΔE gate together with finite/bounded output, seam and stability gates; it does not select based on RGB MSE.

**Why more complex models are not justified:** The protocol allows exactly MMR-0 and, only when its observed residual supports it, MMR-1. Further polynomial, LUT, spatial, or neural families would be an unapproved open-ended model loop.

STOP: Phase 5 tests only MMR-0 and its one gated MMR-1 extension. No MMR-2, neural method, large LUT, spatial LUT, polynomial expansion, production integration, or full-video run follows this report.
