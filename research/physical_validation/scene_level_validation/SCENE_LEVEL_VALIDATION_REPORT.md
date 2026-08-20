# Phase 6 — Real Scene-Level Validation of Frozen MMR-1

**Final status:** `INSUFFICIENT_DATA_OR_GATE_FAILURE`  
**Controlled full-shot render:** **not ready and not performed**

## Scope and frozen contract

This is a research-only, bounded Phase 6 validation. It did not create MMR-2 or another model family; change MMR-1 equations, features, regularization, objective, constraints, parameterization, sync, or geometry; modify production or P2.32; perform a full-film scan; or render a full shot.

`PHASE6_FROZEN_PROTOCOL.json` was written before source extraction. Startup verified SHA-256 `BAEF76DBFABCEE3FA11656E8BD34B3DFA7E12E948386949910E17D6FC1F076F3` for the frozen Phase-5 MMR implementation, the frozen helper/P2.32 hashes, and the frozen MMR-1 contract: 12 coefficients, features `[1, Y_target_nits/100, Ct, Cp, Y·Ct, Y·Cp]`, λ grid `{1e-4, 1e-3, 1e-2}`, ridge identity prior, chroma bound ±0.5, 4096/2048 spatial diagnostic samples, condition ≤1e6, identity deviation ≤5.0, and final per-frame Model-E luminance reimposition.

For each fixed anchor, shot grouping was limited to a **±5 s HDR proxy window** (320-pixel width). Boundary scoring was unchanged from the bounded P2.34 convention: 0.4 histogram χ² + 0.2 mean luminance difference + 0.4 edge-NCC change, hard threshold 0.30, gradual threshold 0.15, and minimum gradual run 5 frames. A shot had to have preceding and following boundaries inside that exact window. Missing evidence was never widened, inferred, or replaced.

## Bounded shot-grouping outcome

| Fixed P2.32 anchor | Local status | Result |
|---|---|---|
| Matrix 08, HDR/OM 73367/73348 | `LOCAL_SHOT_RESOLVED` | Bounded interval `[73274, 73460)`, 186 frames / 7.758 s; five separated samples available. |
| Matrix 03, 28382/28363 | `ANCHOR_NEAR_CUT` | Rejected by the fixed 12-frame near-cut guard; no fitting or frame extraction followed. |
| Matrix 12, 111978/111959 | `LOCAL_SHOT_UNRESOLVED` | Both local shot boundaries were not demonstrated inside ±5 s; no fitting or frame extraction followed. |
| BR2049 01, 11699/12866 | `ANCHOR_NEAR_CUT` | Rejected by the fixed guard. Its geometry confidence remains 0.9155, so it remains conditional and non-confirmatory. |

This result is intentionally `INSUFFICIENT DATA` for broad scene-level readiness: only one of three primary Matrix anchors met the predeclared bounded-shot rule. The rejected anchors are not evidence of model failure; they are absence of permissible temporal evidence under the frozen protocol.

## Matrix 08: one shared transform across real frames

The resolved Matrix shot used HDR frames `73292`, `73320`, `73366`, and `73413` (10/25/50/75%) for fitting; the OM frame for each was deterministically `HDR−19`. HDR `73440` / OM `73421` (90%) was a whole-frame temporal holdout and did not contribute to either H0 or MMR fitting.

One pooled Model-E H0 parameter set and one pooled MMR-1 coefficient matrix were fit from the four training frames only. The selected frozen candidate used λ `0.0001`, regularized Gram condition `9936.580260`, and coefficient identity deviation `0.779975`—all within the frozen limits:

```text
[[-0.0062128, -0.0009757],
 [-0.0497969,  0.0268100],
 [ 0.7464968, -0.2663913],
 [ 0.2159123,  0.3570090],
 [ 0.0889901,  0.0396592],
 [-0.0057025,  0.0177202]]
```

The exact selected matrix has SHA-256 `63AA8E9A02CA5600A16B5A3B59C3CE8F1AABFF246089AE0178E4D28882B4B50D`. That same hash is recorded on **every one of the five applications**. There was no coefficient averaging, smoothing, refitting, or adaptation at output time.

| Temporal sample | Role | Mean ΔE2000 H0 | Mean ΔE2000 shared MMR-1 | Improvement vs H0 | Phase-5-style bounded/seam gate |
|---|---|---:|---:|---:|---|
| 10% — 73292/73273 | train | 1.484214 | 1.235954 | +16.727% | pass |
| 25% — 73320/73301 | train | 1.515053 | 1.328265 | +12.329% | pass |
| 50% — 73366/73347 | train | 0.928606 | 0.866256 | +6.714% | pass |
| 75% — 73413/73394 | train | recorded in metrics | recorded in metrics | +2.670% | below the 5% improvement sub-gate |
| 90% — 73440/73421 | **whole-frame temporal holdout** | **0.900686** | **0.817071** | **+9.283%** | **pass** |

The 90% holdout is the primary Phase 6 result: its output is finite, has zero chroma-bound pixels, inverse-negative pixel fraction `0.00056424` (<0.005), zero dark fallback pixels, and maximum luminance reimposition error `1.39e-13` nits. The independent 90%-frame MMR fit is explicitly leakage-labelled diagnostic-only; its coefficient distance from the shared transform is `0.671265` L2 and it was never used for output or selection.

## Continuity and visual evidence

Open-Matte-only regions have no HDR ground truth, so their result is evaluated only for coherence with the HDR-center composite—not for objective HDR reconstruction.

On the held-out 90% frame, luminance seam means are invariant by construction: top `85.239758` nits and bottom `12.336482` nits for both H0 and shared MMR-1. The top seam's ICtCp chroma proxy improves from `0.032542` to `0.016385` and its hue proxy from `39.990°` to `7.520°`. The bottom hue proxy improves from `11.360°` to `9.481°`, while bottom chroma proxy changes from `0.019807` to `0.026826`. Thus the output has no severe luminance seam degradation, but it is **not** a universal chroma/hue improvement at every seam.

Inspection of the five four-panel montages and the held-out contact sheet found no newly introduced H1-like high-contrast speckling or obvious luminance discontinuity. MMR-1 visibly adjusts the H0 color rendition in the HDR overlap while retaining the same H0 luminance target. This is an inspection-only finding; it does not establish unseen HDR truth in the Open-Matte extension. The per-frame montages remain the detailed visual evidence.

## Synthetic frozen-model control

The existing frozen known-grade hidden-region diagnostic was rerun without modification. On its hidden region, mean ΔE2000 is `10.710849` for H0, `2.327899` for MMR-0, and `1.769187` for MMR-1. This verifies the retained frozen mechanics under known synthetic truth only; it is not evidence that real-film OM-only pixels have a known HDR target.

## Answers to the required questions

1. **One-transform consistency?** Yes for the only resolved shot: one H0 fit and one exact MMR-1 coefficient matrix were applied to all five samples, proven by the identical coefficient hash.
2. **HDR reproduction versus H0?** The primary whole-frame temporal holdout improves mean ΔE2000 from `0.900686` to `0.817071` (`+9.283%`) and passes finite/bound/seam checks. This confirms transfer into one unseen temporal frame, not all selected Matrix anchors.
3. **SDR Open-Matte → HDR remaster visual improvement?** Matrix 08 shows a visually cleaner chroma rendition in the HDR overlap than H0, without a new visible high-contrast artifact. This is limited to one locally resolved shot and is not a global remaster claim; it is not a DVD-specific test.
4. **OM extension coherence?** The held-out visual/seam evidence supports luminance continuity for Matrix 08. Chroma continuity is mixed at the bottom seam and has no HDR ground truth outside overlap, so the conclusion is limited to coherence rather than fidelity.
5. **Artifacts?** `results/scene_level_metrics.json` retains hashes, commands, byte counts, source-frame hashes, boundaries, coefficients, complete metrics, and synthetic control. Matrix 08 has five previews, difference maps, seam strips, and held-out scientific H0/MMR arrays. `results/phase6_heldout_contact_sheet.png` is the final contact sheet.
6. **Multi-frame stability?** The fixed transform improves ΔE2000 on all five tested frames by `+2.670%` to `+16.727%`; the full temporal holdout passes. The 75% training frame falls below the 5% sub-gate, so uniform Phase-5-style performance across every sampled frame is **not** established.
7. **Near-identity BR2049?** No result. BR2049 01 was `ANCHOR_NEAR_CUT`; no MMR fit or application was permitted. It remains conditional because geometry confidence is `0.9155`, below 0.95.
8. **Readiness for a controlled full-shot render?** **No.** One bounded Matrix shot is encouraging but two of three frozen Matrix anchors did not satisfy defensible local-shot grouping. No controlled render is authorized or performed by this phase.

## Artifacts and stop condition

- Protocol: `research/physical_validation/scene_level_validation/PHASE6_FROZEN_PROTOCOL.json`
- Harness: `research/physical_validation/scene_level_validation/run_scene_level_validation.py`
- Metrics/provenance: `research/physical_validation/scene_level_validation/results/scene_level_metrics.json`
- Final contact sheet: `research/physical_validation/scene_level_validation/results/phase6_heldout_contact_sheet.png`

**STOP.** Phase 6 is complete. No additional model, source-window expansion, geometry adaptation, production change, full-shot render, or full-film processing follows from this report.
