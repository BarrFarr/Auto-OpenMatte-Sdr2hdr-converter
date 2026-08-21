# HCAPM/HPPM test handoff

**Branch:** `feat/full-pipeline-implementation`  
**Test interval:** `[65359,65371)` (12 frames)  
**Profile:** `tools/openmatte_hdr/profiles/br2049_t2714_300f_throughput.json`  
**Segment:** `benchmarks/v05_gpu_native_segmented_test300/segments/segment_000004.mkv`  
**Overlap:** `[0,280,3840,1880]`; all HCAPM diagnostics below use only the first 100 px from the corresponding TOP/BOTTOM boundary.  
**Status:** diagnostic research only; nothing is approved or deployed.

## Frozen baseline and non-goals

Every report uses the existing baseline path:

`NativeV4GpuShotFitter.fit_shot -> v05.v1.predict -> v05.v4.apply_intensity_conditioned`

No test modified fitting, gain field, spatial field, the production renderer, NVDEC/CUDA/NVENC, existing production HCAPM, or source media. No test ran composite, PQ output, encoder, full render, extension correction, production deployment, a matrix, CDF, or residual correction. No result may be extrapolated beyond 100 px.

The commit contains JSON reports plus two compact metric-dashboard PNGs (<300 KB each). It intentionally excludes large source-derived A/B visual PNGs (up to about 20 MB each), rendered video, segments, source media, native bridge binaries, temporary harnesses, bytecode, and unrelated workspace artifacts.

## Report inventory

| Artifact | Test | Result | Key finding |
|---|---|---|---|
| `seam_hppm_ab.json` | Initial 12-sector HPPM seam A/B | **NO-PASS** | TOP chroma mean/p95 and hue mean did not improve. |
| `polar_hue_chroma_ab.json` | Polar ICtCp hue/chroma A/B | **NO-PASS** | TOP chroma/hue failed; BOTTOM hue p95 increased. |
| `seam_hppm_edge_ab.json` | HPPM in 32 px edge bands | **NO-PASS** | BOTTOM medium-chroma mean worsened. |
| `chroma_weighted_hppm_edge_ab.json` | Edge HPPM with `smoothstep(0.08,0.20,C)` | **NO-PASS** | BOTTOM medium-chroma hue p95 worsened. |
| `hppm_narrow_chroma_gate_edge_ab.json` | BOTTOM-only `0.12->0.20` gate; TOP exact | **NO-PASS** | BOTTOM medium hue p95 still worsened. |
| `hcapm_zonal_diagnostic.json` | Six-zone mismatch diagnostic; no correction | **DIAGNOSTIC_ONLY** | TOP and BOTTOM both have systematic Y dependence. |
| `hcapm_zonal_ab.json` | Y-profiled HCAPM: 12 sectors plus Y centers 10/50/90 | **NO-PASS** | 6/18 groups passed; medium and several hue-p95 regressions. |
| `hcapm_high_chroma_anchored_polar_ab.json` + `.png` | 12-sector, non-Y, high-chroma-anchored polar HCAPM | **NO-PASS** | 3/8 groups passed; two fitted hue sectors require broad interpolation. |
| `hcapm_high_chroma_constant_polar_ab.json` + `.png` | Constant polar HCAPM; no Hue sectors and no Y dependency | **NO-PASS** | 3/4 groups passed; BOTTOM high-chroma improves, TOP high-chroma p95 worsens. |

## Baseline zonal diagnostic

`hcapm_zonal_diagnostic.json` measured `C >= 0.10` baseline corrected-OM samples and `HDR C > 0` in bands `0–20`, `40–60`, and `80–100 px`.

- TOP: OM `280:300`, `320:340`, `360:380` vs HDR `0:20`, `40:60`, `80:100`.
- BOTTOM: OM `1860:1880`, `1820:1840`, `1780:1800` vs HDR `1580:1600`, `1540:1560`, `1500:1520`.
- Classification: `B_systematic_Y_dependence` on both sides.
- The trend is diagnostic evidence only. Do **not** interpolate it beyond 100 px, apply it to the extension, or merge it into production.

## HCAPM high-chroma anchored polar result

`hcapm_high_chroma_anchored_polar_ab.json` tested a non-Y model with 12 hue sectors, cyclic gap interpolation/smoothing, and an envelope of `0.03→0.08`.

- Training: baseline corrected RGB OM `C >= 0.10`, HDR `C > 0`.
- Application: Ct/Cp only; ICtCp I copied exactly; correction limited to 0–100 px.
- Result: **NO-PASS, 3/8 groups**.
- Invariants: applied `I_max_abs_change=0`; low-chroma Ct/Cp change `=0`; finite RGB/ICtCp; no nonfinite values; no preclip out-of-range samples; maximum RGB-luminance change `0.02593491 nits`.
- Critical limitation: only hue sectors 10 and 11 had the minimum high-chroma support on both sides; sectors 0–9 were interpolated. This profile therefore does not provide enough directional evidence to justify a wider model.

## HCAPM high-chroma constant polar result

`hcapm_high_chroma_constant_polar_ab.json` is the requested simpler control test. It intentionally has `hue_sector_count=0` and `Y_dependent_correction=false`.

### Contract

- TOP and BOTTOM fit independently over `0–100 px`.
- Training only: `C_OM >= 0.15` and `HDR C > 0`.
- One robust circular-median `delta_H` and one median `delta_log_C` per side; no hue sectors.
- Application: `C < 0.15` has exactly zero correction; `C >= 0.15` receives full constant polar correction.
- Equations: `H_out=H+delta_H_side`, `C_out=C*exp(delta_log_C_side)`, `Ct_out=C_out*cos(H_out)`, `Cp_out=C_out*sin(H_out)`, `I_out=I_exact`.
- Precomputed profile values are recorded: `cos(delta_H)`, `sin(delta_H)`, `exp(delta_log_C)`.
- No extension correction, no Y/spatial correction, and no production renderer modification.

### Fitted constants

| Side | High-chroma anchors | delta-H | delta-log-C | chroma scale |
|---|---:|---:|---:|---:|
| TOP | 2,896,812 | `+0.081474 deg` | `+0.00690104` | `1.00692490` |
| BOTTOM | 1,457,580 | `+0.675630 deg` | `+0.02139639` | `1.02162693` |

### Result: **NO-PASS (3/4 groups)**

| Side/group | Chroma mean A→B | Chroma p95 A→B | Hue p95 A→B | Qualification |
|---|---:|---:|---:|---|
| TOP, `C < 0.15` | `0.00976918→0.00976918` | `0.02086093→0.02086093` | `73.31958→73.31958 deg` | PASS, exact unchanged |
| TOP, `C >= 0.15` | `0.00679133→0.00666729` | `0.01857865→0.01865748` | `2.91412→2.89610 deg` | **NO-PASS**: chroma p95 increases |
| BOTTOM, `C < 0.15` | `0.00954561→0.00954561` | `0.03928655→0.03928655` | `134.72154→134.72154 deg` | PASS, exact unchanged |
| BOTTOM, `C >= 0.15` | `0.00556584→0.00318563` | `0.01147058→0.00761604` | `3.22350→2.95366 deg` | PASS |

Constant-polar invariants:

- Applied ICtCp `I_max_abs_change = 0.0`.
- Round-trip I drift is numerical only: `1.579167e-09`.
- Inactive (`C < 0.15`) Ct/Cp maximum change is `0.0`.
- RGB and ICtCp are finite; both nonfinite counts are zero.
- Preclip out-of-range fraction is zero.
- Maximum RGB-luminance change is `0.01722762 nits`, below the requested `0.05 nits` limit.

The decision gate remains blocked: BOTTOM success does not qualify a correction when TOP high-chroma p95 regresses. Do not proceed to HCAPM + spatial/Y envelope, do not modify production HCAPM, and do not correct the extension.

## HPPM progression

The HPPM variants reduced scope but did not produce a qualifying correction:

1. Full seam HPPM failed mainly at TOP.
2. A 32 px edge-band restriction worsened BOTTOM medium chroma mean.
3. A `smoothstep(0.08,0.20,C)` gate worsened BOTTOM medium hue p95.
4. A narrower BOTTOM-only `0.12→0.20` gate reduced but did not remove the BOTTOM medium hue-p95 regression.

No HPPM calibration or gate in this directory is approved for production.

## Handoff decision and constraints for the next owner

1. Treat every correction result here as a failed diagnostic experiment, not a candidate deployment.
2. Keep the baseline and the exact `[65359,65371)` interval frozen for any direct comparison.
3. Do not alter fitting, gain, spatial field, renderer, NVDEC/CUDA/NVENC, existing production HCAPM, or source media while diagnosing these reports.
4. Do not run a full render, test other sequences/profiles, or apply any correction outside the `0–100 px` overlap region under this test protocol.
5. Do not add spatial/Y behavior, extension correction, matrix, CDF, or residual correction. The prerequisite constant-polar test did not pass.
6. The JSON reports contain exact geometry, trained constants/calibration, group metrics, qualification rules, invariants, and timings. The compact PNGs are summaries only.

## Reproducibility

- GPU bridge used locally: `dev/v05-native/v5_gpu_bridge.dll` (not committed).
- Decoders: `hevc_cuvid` for HDR and `h264_cuvid` for Open Matte.
- Fit scale: `0.5`; fit stride: `1`.
- Frozen fitter split: 9 training and 3 holdout frames.
- Temporary test harnesses and bytecode were deleted after successful report generation.
