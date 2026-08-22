# Final Scene Validation — Phase 8

**Status:** `COMPLETE`

Research-only bounded validation of the frozen Model E luminance + ratio scaling + MMR-1 transform. No model, sampling-budget, production, synchronization, or geometry change was made.

## Fixed configuration

- Four representative training frames: 10%, 25%, 50%, 75%.
- 5,000 deterministic correctly paired full-resolution common-grid rows per training frame (20,000 per shot).
- 2,048 paired held-mask rows per training frame only for frozen λ selection.
- 90% temporal frame excluded from all fitting and applied only after one shared transform is selected.
- One H0 parameter set and one frozen 12-coefficient MMR-1 matrix per shot; no per-frame fitting.

## Automatic bounded-shot selection

All existing P2.32 temporal strata were considered only through the unchanged ±5-second HDR proxy / fixed boundary machinery. The selection was by predeclared P2.32 index and local-shot eligibility, never by fit quality. No window was widened and no frame/offset/geometry was manually changed.

- Matrix selected: `the_matrix_temporal_stratum_08, the_matrix_temporal_stratum_02, the_matrix_temporal_stratum_05`
- BR2049 selected: `br2049_temporal_stratum_03`
- Desired minimum: 3 Matrix + 1 BR2049; achieved: 3 Matrix + 1 BR2049.

## Per-shot validation

| Shot | Material | Status | λ | Fit s | 90% H0 → MMR mean ΔE2000 | 90% improvement | Conditional geometry |
|---|---|---|---:|---:|---:|---:|---|
| the_matrix_temporal_stratum_08 | The Matrix | VALIDATED | 0.0001 | 26.0926 | 0.917301 → 0.822596 | +10.324% | False |
| the_matrix_temporal_stratum_02 | The Matrix | VALIDATED | 0.0001 | 18.9656 | 1.115842 → 1.081989 | +3.034% | False |
| the_matrix_temporal_stratum_05 | The Matrix | VALIDATION_FAILED | n/a | n/a | n/a | n/a | False |
| br2049_temporal_stratum_03 | BR2049 | VALIDATED | 0.001 | 40.4757 | 0.116680 → 0.088607 | +24.060% | True |

Each JSON shot record contains the complete 12 coefficients, λ candidates, coefficient hash, direct full-overlap ΔE2000/P95, luminance MAE/RMSE, chroma/hue metrics, top/bottom seam metrics, output range/clipping/negative checks, source decode provenance, and the five-frame contact sheet. BR2049 is reported strictly as conditional near-identity/fidelity evidence when available.

## Full-shot renders

| Shot | Frames | Analysis s | Application mean/P95 s per frame | Total render s | Exact stream check | Output |
|---|---:|---:|---:|---:|---|---|
| the_matrix_temporal_stratum_08 | 186 | 26.0926 | 1.1885/1.2867 | 330.573 | True | `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\final_scene_validation\results\renders\the_matrix_temporal_stratum_08\MMR1_generated_HDR_OpenMatte_full_shot_ffv1.mkv` |
| the_matrix_temporal_stratum_02 | 135 | 18.9656 | 1.2107/1.4295 | 248.960 | True | `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\final_scene_validation\results\renders\the_matrix_temporal_stratum_02\MMR1_generated_HDR_OpenMatte_full_shot_ffv1.mkv` |

## Recorded failures and delivery caveat

- **Matrix 05 (`the_matrix_temporal_stratum_05`):** `VALIDATION_FAILED` before any render because **all frozen MMR-1 λ candidates were pathological** under the unchanged fixed `4 × 5,000` configuration. No coefficient, sample budget, synchronization, geometry, or model behavior was changed to force it through. It is the reason there are two rendered Matrix shots rather than three.
- **Render source equivalence:** both rendered shots passed first-and-final-frame byte-hash comparison between the sequential render decoder and independent exact RGB48LE extraction. Matrix 08 rendered all `186` frames; Matrix 02 rendered all `135` frames. All render-frame outputs were finite. Matrix 08 had maximum chroma clipping/inverse-negative fractions of `0/0`; Matrix 02 had `0/0.00024306`, below the frozen `0.005` limit.
- **Container metadata caveat:** the generated video payloads are lossless `FFV1`, `1920×1080`, `gbrp16le` files whose code values are generated through PQ OETF. However, independent `ffprobe` verification reports `color_space=gbr` and does **not** retain BT.2020/SMPTE-2084 stream tags. They are valid research render artifacts but must **not** be treated as pipeline-ready tagged HDR deliverables without a separately controlled packaging/metadata step.
- **Visual inspection result:** the generated contact sheets, seam strips, overlap difference maps, and two complete full-shot videos were inspected as diagnostic previews. The Matrix 08 MMR-1 panels, particularly later representative fight frames, show visible non-neutral purple/cyan chroma shifts relative to the H0/HDR comparison panels. This is a visual artifact/fidelity failure, not an accepted merge result. The previews are not a calibrated HDR display review, so they cannot quantify its display-space severity; however, they are sufficient to reject a visual PASS. No artifact was manually corrected or hidden.

## Final decision

A. **Is the frozen MMR-1 configuration stable across multiple real shots?** **NO.** Matrix 05 has no stable frozen λ candidate, and preview review flags visible chroma/hue artifacts in the rendered Matrix shots; three-shot practical stability is therefore not established.
B. **Is 4 × 5,000 samples sufficient in practice?** **NO.** It yielded two numerical Matrix fits but failed on the third automatically selected Matrix shot; the fixed budget cannot be declared sufficient across the required practical shot set.
C. **Does one T_shot remain valid across all frames of a shot?** **Numerically yes for Matrix 08 and Matrix 02 only:** the same persisted coefficient hash was used for every application, the stream-frame hashes matched exact extraction, and all render-frame numerical bounds passed. **Visually, no general validity claim is supported** because the preview review flags chroma/hue artifacts.
D. **Does the generated Open Matte visually merge with the HDR center?** **NO.** The Matrix diagnostic panels show visible non-neutral purple/cyan chroma shifts in MMR-1 output; calibrated HDR display review could refine severity, but cannot reverse this preview-level failure into a PASS.
E. **Does the process remain fast enough for local full-film use?** **NO.** Application averages `1.1885–1.2107 s/frame` (about `28.5–29.1×` slower than 24 fps), projecting roughly `57–58` application hours for a two-hour 24-fps title before decode/encode/I/O.
F. **Is the current algorithm ready for a controlled production-pipeline integration test?** **NO.** Blocking issues are the Matrix 05 pathological frozen fits, visible Matrix chroma/hue artifacts, missing PQ/BT.2020 stream metadata in the current FFV1 outputs, and insufficient cross-scene practical stability. Do not create a new model automatically.

## Hard stop

STOP. This phase does not authorize MMR-2, a new model family, another sampling sweep, retuning, LUT/AI work, production changes, or full-film processing.
