# Phase 16 — per-actual-shot low-frequency chroma-field extension

The run uses the verified Matrix-08 temporal partition. Every field, alpha, TRAIN threshold, gain record, and metric accumulator is isolated by actual shot. The final aggregate is descriptive only and contains no averaged parameter.

## Matrix-08-01 — frames [73274, 73295) (21 frames)

- `alpha_t`: raw `+0.0271234`, shrunk `+0.0258319`, applied `+0.0258319`.
- `alpha_p`: raw `+0.0231009`, shrunk `+0.0220009`, applied `+0.0220009`.
- TRAIN-derived HDR base-chroma thresholds: P50 `0.0851758`, P90 `0.1134370`.
- Fit time `23.706s`; application `1.329634s/frame` during evaluation.
- Pre-visual gate: **METRIC_GATE_PASS__VISUAL_REVIEW_REQUIRED**; manual visual review remains required.

### TRAIN metrics
| Metric | A frozen baseline | B + residual |
|---|---:|---:|
| ΔE_ITP mean | 59.776642 | 53.605034 |
| ΔE_ITP P95 | 171.231689 | 170.587509 |
| Low-frequency chroma residual | 0.0370527 | 0.0135916 |
| Chroma absolute error | 0.0354757 | 0.0157134 |
| Low-chroma hue error | 16.3482° | 6.7147° |
| High-chroma hue error | 6.2719° | 4.9732° |
| Luminance MAE | 11.7823 nits | 11.7771 nits |
| Luminance RMSE | 21.5583 nits | 21.5492 nits |
| ΔE2000 auxiliary mean | 12.323904 | 10.590125 |
| ΔE2000 auxiliary P95 | 31.183924 | 30.926493 |

### HOLDOUT metrics
| Metric | A frozen baseline | B + residual |
|---|---:|---:|
| ΔE_ITP mean | 59.502596 | 53.326300 |
| ΔE_ITP P95 | 170.320740 | 169.586472 |
| Low-frequency chroma residual | 0.0370014 | 0.0136184 |
| Chroma absolute error | 0.0355518 | 0.0157872 |
| Low-chroma hue error | 16.3760° | 6.7947° |
| High-chroma hue error | 6.3721° | 4.7384° |
| Luminance MAE | 11.8220 nits | 11.8163 nits |
| Luminance RMSE | 21.6381 nits | 21.6274 nits |
| ΔE2000 auxiliary mean | 12.311378 | 10.553156 |
| ΔE2000 auxiliary P95 | 30.853321 | 30.533184 |

### Seam
| Quantity | A | B |
|---|---:|---:|
| top chroma step | 0.0243684 | 0.0158133 |
| top hue step | 19.2024° | 8.1135° |
| top luminance step | 5.7041 nits | 5.7039 nits |
| bottom chroma step | 0.0330819 | 0.0183351 |
| bottom hue step | 6.8803° | 4.6339° |
| bottom luminance step | 11.5177 nits | 11.5472 nits |

### Visual
- Composite contact sheet: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\contact_sheet.png`
- A/B low-frequency residual maps: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\residual_maps\lowpass_chroma_A_overlap.png`, `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\residual_maps\lowpass_chroma_B_overlap.png`
- Amplified exact-composite B−A residual (HDR centre must be zero): `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\residual_maps\amplified_B_minus_A_composite.png`
- Top/bottom seam context: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\seam_maps\top_bottom_A_B_context.png`
- Deterministic HDR-reference-only highest-chroma window: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\review_windows\highest_chroma_hdr_reference_only.png`
- Visual manifest / manual target checklist: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-01\visual\visual_manifest.json`

## Matrix-08-02 — frames [73295, 73434) (139 frames)

- `alpha_t`: raw `+0.0034111`, shrunk `+0.0032486`, applied `+0.0032486`.
- `alpha_p`: raw `+0.0017670`, shrunk `+0.0016828`, applied `+0.0016828`.
- TRAIN-derived HDR base-chroma thresholds: P50 `0.0523508`, P90 `0.0994213`.
- Fit time `157.787s`; application `1.354854s/frame` during evaluation.
- Pre-visual gate: **NO_VALUE_DEMONSTRATED**; manual visual review remains required.

### TRAIN metrics
| Metric | A frozen baseline | B + residual |
|---|---:|---:|
| ΔE_ITP mean | 49.849232 | 49.832192 |
| ΔE_ITP P95 | 147.959595 | 147.955200 |
| Low-frequency chroma residual | 0.0188765 | 0.0188499 |
| Chroma absolute error | 0.0192002 | 0.0190982 |
| Low-chroma hue error | 33.9781° | 34.5177° |
| High-chroma hue error | 6.2111° | 5.7149° |
| Luminance MAE | 10.5203 nits | 10.5195 nits |
| Luminance RMSE | 21.6520 nits | 21.6498 nits |
| ΔE2000 auxiliary mean | 9.332669 | 9.299946 |
| ΔE2000 auxiliary P95 | 30.757353 | 30.723343 |

### HOLDOUT metrics
| Metric | A frozen baseline | B + residual |
|---|---:|---:|
| ΔE_ITP mean | 50.088046 | 50.074610 |
| ΔE_ITP P95 | 149.619370 | 149.615097 |
| Low-frequency chroma residual | 0.0188319 | 0.0188233 |
| Chroma absolute error | 0.0190916 | 0.0190058 |
| Low-chroma hue error | 34.4734° | 35.0176° |
| High-chroma hue error | 6.0979° | 5.5991° |
| Luminance MAE | 10.5195 nits | 10.5187 nits |
| Luminance RMSE | 21.6448 nits | 21.6426 nits |
| ΔE2000 auxiliary mean | 9.355240 | 9.324047 |
| ΔE2000 auxiliary P95 | 31.047361 | 31.024151 |

### Seam
| Quantity | A | B |
|---|---:|---:|
| top chroma step | 0.0141539 | 0.0138775 |
| top hue step | 12.4456° | 12.4201° |
| top luminance step | 3.3247 nits | 3.3246 nits |
| bottom chroma step | 0.0196395 | 0.0192022 |
| bottom hue step | 25.9662° | 25.2842° |
| bottom luminance step | 9.8314 nits | 9.8319 nits |

### Visual
- Composite contact sheet: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\contact_sheet.png`
- A/B low-frequency residual maps: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\residual_maps\lowpass_chroma_A_overlap.png`, `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\residual_maps\lowpass_chroma_B_overlap.png`
- Amplified exact-composite B−A residual (HDR centre must be zero): `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\residual_maps\amplified_B_minus_A_composite.png`
- Top/bottom seam context: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\seam_maps\top_bottom_A_B_context.png`
- Deterministic HDR-reference-only highest-chroma window: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\review_windows\highest_chroma_hdr_reference_only.png`
- Visual manifest / manual target checklist: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-02\visual\visual_manifest.json`

## Matrix-08-03 — frames [73434, 73460) (26 frames)

- `alpha_t`: raw `+0.0090357`, shrunk `+0.0086055`, applied `+0.0086055`.
- `alpha_p`: raw `+0.0012869`, shrunk `+0.0012256`, applied `+0.0012256`.
- TRAIN-derived HDR base-chroma thresholds: P50 `0.0592733`, P90 `0.0997665`.
- Fit time `30.127s`; application `1.346947s/frame` during evaluation.
- Pre-visual gate: **METRIC_GATE_PASS__VISUAL_REVIEW_REQUIRED**; manual visual review remains required.

### TRAIN metrics
| Metric | A frozen baseline | B + residual |
|---|---:|---:|
| ΔE_ITP mean | 41.595753 | 41.346689 |
| ΔE_ITP P95 | 139.405563 | 139.310455 |
| Low-frequency chroma residual | 0.0164367 | 0.0145629 |
| Chroma absolute error | 0.0166244 | 0.0148974 |
| Low-chroma hue error | 14.8302° | 12.2024° |
| High-chroma hue error | 4.1511° | 3.0080° |
| Luminance MAE | 8.1396 nits | 8.1431 nits |
| Luminance RMSE | 27.2903 nits | 27.2851 nits |
| ΔE2000 auxiliary mean | 7.717134 | 7.699163 |
| ΔE2000 auxiliary P95 | 26.137136 | 25.883976 |

### HOLDOUT metrics
| Metric | A frozen baseline | B + residual |
|---|---:|---:|
| ΔE_ITP mean | 41.455421 | 41.207067 |
| ΔE_ITP P95 | 138.240402 | 138.135223 |
| Low-frequency chroma residual | 0.0162558 | 0.0143518 |
| Chroma absolute error | 0.0163971 | 0.0146656 |
| Low-chroma hue error | 14.9471° | 12.2214° |
| High-chroma hue error | 4.1112° | 2.9819° |
| Luminance MAE | 7.5721 nits | 7.5759 nits |
| Luminance RMSE | 23.3215 nits | 23.3182 nits |
| ΔE2000 auxiliary mean | 7.629024 | 7.610153 |
| ΔE2000 auxiliary P95 | 25.713308 | 25.492693 |

### Seam
| Quantity | A | B |
|---|---:|---:|
| top chroma step | 0.0176376 | 0.0172282 |
| top hue step | 15.0567° | 9.6704° |
| top luminance step | 28.5220 nits | 28.5124 nits |
| bottom chroma step | 0.0173843 | 0.0184829 |
| bottom hue step | 5.1163° | 5.0649° |
| bottom luminance step | 8.4585 nits | 8.4634 nits |

### Visual
- Composite contact sheet: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\contact_sheet.png`
- A/B low-frequency residual maps: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\residual_maps\lowpass_chroma_A_overlap.png`, `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\residual_maps\lowpass_chroma_B_overlap.png`
- Amplified exact-composite B−A residual (HDR centre must be zero): `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\residual_maps\amplified_B_minus_A_composite.png`
- Top/bottom seam context: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\seam_maps\top_bottom_A_B_context.png`
- Deterministic HDR-reference-only highest-chroma window: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\review_windows\highest_chroma_hdr_reference_only.png`
- Visual manifest / manual target checklist: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\shots\Matrix-08-03\visual\visual_manifest.json`

## Aggregate — only after per-shot reports

This is an equal-shot descriptive macro mean. It does not average or select gain, field coefficients, thresholds, `alpha_t`, or `alpha_p`.

| Holdout metric | A macro mean | B macro mean | B-vs-A improvement |
|---|---:|---:|---:|
| deltaE_ITP_mean | 50.3486877 | 48.2026592 | +4.26% |
| deltaE_ITP_P95 | 152.7268372 | 152.4455973 | +0.18% |
| low_frequency_chroma_residual_mean | 0.0240297 | 0.0155978 | +35.09% |
| chroma_absolute_error_mean | 0.0236802 | 0.0164862 | +30.38% |
| hue_low_chroma_P50_degrees | 21.9321451 | 18.0112210 | +17.88% |
| hue_high_chroma_P90_degrees | 5.5270528 | 4.4398162 | +19.67% |
| luminance_MAE_nits | 9.9712364 | 9.9702938 | +0.01% |
| luminance_RMSE_nits | 22.2014384 | 22.1960426 | +0.02% |
| deltaE2000_auxiliary_mean | 9.7652136 | 9.1624519 | +6.17% |
| deltaE2000_auxiliary_P95 | 29.2046636 | 29.0166759 | +0.64% |
| seam_chroma_step_mean | 0.0210443 | 0.0171565 | +18.47% |

## Full Matrix-08 review outputs

- A: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\video\Matrix08_A_baseline_hdr10.mkv`
- B: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\video\Matrix08_B_extended_chroma_hdr10.mkv`
- A vs B: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\chroma_field_extension\results\matrix08_phase16_per_shot\video\Matrix08_A_vs_B_hdr10.mkv`
- Full render: `347.901s`; application `1.376893s/frame`.

No additional material was run after Matrix-08.
