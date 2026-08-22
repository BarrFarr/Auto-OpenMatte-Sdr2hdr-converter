# Model Discrimination Report — Phase 4 (Final)

## Scope and frozen protocol
This is one bounded diagnostic and validation phase. It reads the established Matrix H0/H1 outputs for residual analysis; it does not refit H0/H1 for that analysis, modify P2.32/production, decode source video, use AI/LUTs/spatial transforms, or start another model family. All real-pair fits use the SDR∩HDR overlap only and apply the frozen selected architecture to the entire Open-Matte SDR frame.

Predeclared before scores: Matrix development `08` (73367/73348); Matrix candidates `03` (28382/28363) and `12` (111978/111959); BR2049 candidate `01` (11699/12866). The Matrix candidates are real, temporally distant P2.32 pairs but have unverified shot/scene independence. BR2049 is real but has recorded geometry confidence 0.9155, below the ordinary 0.95 gate; it is a conditional robustness result, not geometry-confirmatory evidence.

## Established H0/H1 residuals — all 1,536,000 Matrix overlap pixels
H0 is Model E luminance-only plus ratio scaling. H1 is the already-established regularized 3×3 H0 transform. Reference bins are computed from HDR reference luminance/chroma/hue, not model predictions. Hue error excludes only reference pixels with ICtCp chroma ≤1e-6; no luminance/chroma group is discarded, and all group counts are reported.

### H0 by HDR reference luminance
| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |
|---|---:|---:|---:|---:|---:|---:|
| 0–5 | 963101 | 3.147223 | 0.400693 | 0.030591 / 0.031956 | 27.678880 | 3.9241/2.7276/1.3874 |
| 5–20 | 229578 | 9.991823 | 1.019728 | 0.004130 / 0.020760 | 8.631682 | 4.8954/4.4502/2.7649 |
| 20–50 | 208305 | 16.451558 | 1.763123 | -0.012785 / 0.020333 | 9.662071 | -6.0481/-3.7880/0.3683 |
| 50–100 | 117517 | 30.063443 | 3.195733 | -0.018464 / 0.024382 | 9.893376 | -32.4892/-23.7921/-7.9736 |
| 100–250 | 17499 | 48.859822 | 5.192374 | -0.031355 / 0.033831 | 8.121614 | -61.4786/-44.0883/-10.5825 |
| 250–500 | 0 | — | — | — / — | — | — |
| 500–1000 | 0 | — | — | — / — | — | — |
| >1000 | 0 | — | — | — / — | — | — |

### H1 by HDR reference luminance
| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |
|---|---:|---:|---:|---:|---:|---:|
| 0–5 | 963101 | 2.770306 | 0.291065 | 0.011388 / 0.016355 | 24.722793 | 3.2392/2.3868/1.5391 |
| 5–20 | 229578 | 8.864844 | 0.915069 | -0.012414 / 0.021887 | 10.194546 | 3.3335/2.6296/2.6058 |
| 20–50 | 208305 | 15.928240 | 1.866793 | -0.025182 / 0.027731 | 10.953691 | -8.8307/-7.9365/-0.8092 |
| 50–100 | 117517 | 32.516545 | 3.588748 | -0.030610 / 0.032573 | 11.619601 | -36.3054/-29.4530/-9.7146 |
| 100–250 | 17499 | 55.892714 | 5.703521 | -0.042195 / 0.042234 | 11.077721 | -68.1978/-54.4235/-14.1735 |
| 250–500 | 0 | — | — | — / — | — | — |
| 500–1000 | 0 | — | — | — / — | — | — |
| >1000 | 0 | — | — | — / — | — | — |

### H0 by reference chroma magnitude
Reference ICtCp tertiles: `0.03125858`, `0.06563958`.
| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |
|---|---:|---:|---:|---:|---:|---:|
| low chroma | 512000 | 1.447030 | 0.210354 | 0.032656 / 0.032714 | 41.097021 | 1.9349/1.2095/0.7309 |
| medium chroma | 512000 | 7.823328 | 0.855254 | 0.021940 / 0.029147 | 13.378504 | 3.5162/2.8792/1.8880 |
| high chroma | 512000 | 16.393513 | 1.773643 | -0.005712 / 0.022582 | 7.941660 | -7.8934/-5.4714/-0.8113 |

### H1 by reference chroma magnitude
| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |
|---|---:|---:|---:|---:|---:|---:|
| low chroma | 512000 | 1.251885 | 0.140514 | 0.016554 / 0.016931 | 35.043777 | 1.5205/1.0592/0.8055 |
| medium chroma | 512000 | 7.457748 | 0.738208 | 0.004986 / 0.016286 | 12.754624 | 2.2074/0.9846/1.1927 |
| high chroma | 512000 | 16.330412 | 1.857239 | -0.024398 / 0.027563 | 10.781290 | -10.3967/-8.2242/-0.9780 |

### H0 by reference hue sector
| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |
|---|---:|---:|---:|---:|---:|---:|
| red | 72053 | 1.035744 | 0.202350 | 0.037194 / 0.037195 | 97.464450 | 1.6140/0.8370/0.7470 |
| yellow | 107646 | 1.133958 | 0.161014 | 0.030806 / 0.030889 | 51.215806 | 1.5817/0.9881/0.5852 |
| green | 748470 | 5.647369 | 0.702521 | 0.024424 / 0.031531 | 13.286588 | 1.2702/1.0535/0.6716 |
| cyan | 575938 | 15.073486 | 1.542016 | -0.001123 / 0.021236 | 10.199389 | -4.4153/-2.9353/0.4791 |
| blue | 1028 | 1.786628 | 0.237313 | 0.057577 / 0.057577 | 106.566022 | 2.4371/1.6148/0.7336 |
| magenta | 30865 | 1.072132 | 0.246827 | 0.043415 / 0.043415 | 113.261476 | 1.7082/0.8362/0.9513 |

### H1 by reference hue sector
| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |
|---|---:|---:|---:|---:|---:|---:|
| red | 72053 | 0.918447 | 0.123175 | 0.019263 / 0.019393 | 77.990536 | 1.2097/0.8100/0.8680 |
| yellow | 107646 | 0.983845 | 0.106701 | 0.015991 / 0.016298 | 42.487995 | 1.2466/0.8887/0.6591 |
| green | 748470 | 5.286140 | 0.606760 | 0.002082 / 0.018700 | 13.076690 | 0.2110/0.6192/0.9862 |
| cyan | 575938 | 15.038239 | 1.599756 | -0.012086 / 0.022801 | 12.377695 | -6.6567/-6.6131/-0.6660 |
| blue | 1028 | 1.619144 | 0.183474 | 0.042320 / 0.042320 | 111.938377 | 2.1085/1.4845/0.8449 |
| magenta | 30865 | 0.944263 | 0.154229 | 0.025499 / 0.025795 | 89.750024 | 1.2292/0.8218/1.0823 |

## Dependency diagnosis and Model J gate
The predeclared statistic is weighted relative standard deviation of group mean ΔE2000. A dependency is strong at ≥0.25; one axis is dominant only if it is also ≥1.5× the other. H0: `C_mixed_dependency` (luminance `0.9858`, chroma `0.6778`, hue `0.5191`). H1: `A_luminance_dominant` (luminance `1.1886`, chroma `0.7802`, hue `0.6098`). **Combined diagnosis: C_mixed_dependency.**

Model J was permitted only if **both** H0 and H1 were luminance-dominant. Permitted: **False**. Not fitted: predeclared residual gate did not establish a dominant luminance-only dependency in both H0 and H1.

## Development comparison and selection gate
| Model | luma MAE nits | mean ΔE2000 | mean |chroma| | mean |hue| ° |
|---|---:|---:|---:|---:|
| H0 | 8.554624 | 0.946417 | 0.028148 | 20.804764 |
| H1 | 8.346682 | 0.911987 | 0.020260 | 19.525826 |

H1's established mean ΔE2000 change is below the mandatory 5% gate and its Phase 3 visual review found localized blue/purple dark-garment artifacts. Thus it fails despite stable numerical coefficients. The mandatory gate is `mean ΔE2000 improvement >=5%; no visible color artifacts; no worsening seam continuity; no pathological coefficients; no significant instability`.

## Stability and performance
The established H1 selected matrix is numerically stable but visually failed: determinant `0.3794291`, condition number `2.97778`, and `||M-I||_F` `0.8534901`. It is rejected because numerical stability does not override its failed 5%/artifact gates. Model J was not fitted, so it supplies no additional stability claim.

## Frozen-architecture four-pair validation
The frozen applied architecture is **H0**. Outside overlap, values are an unverified Open-Matte extrapolation: seam is only a continuity proxy, not HDR ground truth.

| Pair | HDR / SDR-OM frame | luma MAE nits | mean ΔE2000 | mean |hue| ° | top / bottom seam nits | geometry confidence |
|---|---|---:|---:|---:|---:|---:|
| The Matrix the_matrix_temporal_stratum_08 (development_verified_anchor) | 73367 / 73348 | 8.5546 | 0.9464 | 20.805 | 7.039 / 5.494 | 1.0000 |
| The Matrix the_matrix_temporal_stratum_03 (predeclared_temporal_candidate) | 28382 / 28363 | 13.3630 | 1.9710 | 11.354 | 5.328 / 33.532 | 1.0000 |
| The Matrix the_matrix_temporal_stratum_12 (predeclared_temporal_candidate) | 111978 / 111959 | 7.8628 | 1.0383 | 18.147 | 4.396 / 5.773 | 1.0000 |
| BR2049 br2049_temporal_stratum_01 (predeclared_conditional_geometry_candidate) | 11699 / 12866 | 0.0462 | 0.0085 | 37.410 | 0.063 / 0.055 | 0.9155 |

| Pair | H0 fit s | full-SDR apply s | finite | output range (normalized linear) |
|---|---:|---:|---|---:|
| the_matrix_temporal_stratum_08 | 0.0702 | 0.0530 | True | 0.000007…0.022762 |
| the_matrix_temporal_stratum_03 | 0.0719 | 0.0550 | True | 0.000005…0.039716 |
| the_matrix_temporal_stratum_12 | 0.0790 | 0.0619 | True | 0.000021…0.015531 |
| br2049_temporal_stratum_01 | 0.3096 | 0.2718 | True | 0.000000…0.005773 |

## Synthetic hidden-region validation
Existing predeclared synthetic test: fit only central crop, evaluate both crop and known hidden region. It is a controlled global-transform sanity check, not proof of transfer to real film.

| Variant | common mean ΔE2000 | hidden mean ΔE2000 | hidden luma MAE nits |
|---|---:|---:|---:|
| H0 | 2.257705 | 1.893367 | 2.171249 |
| H1 | 0.027912 | 0.062807 | 1.885585 |

## Final decision — C — no compact tested model is sufficiently general
H1 fails the 5% acceptance gate and visible-artifact gate. The diagnostic did not authorize arbitrary architecture expansion beyond at most J; the frozen four-pair validation is H0 baseline evidence, with unverified Matrix shot independence and BR geometry below the confirmatory threshold.

**Why the chosen outcome is necessary:** The observed residual structure and failed H1 gate do not support claiming that a global matrix or an untested extension reconstructs the studio regrade.

**Why more complex models are not justified:** J was only permitted for clear dual H0/H1 luminance dominance. The gate result above controls this decision; adding hue/chroma/polynomial/LUT/spatial/framewise models would violate the one-shot compact-model protocol and would not repair the missing cross-scene/geometry evidence.

Parameter accounting: H0 has 10 effective compact luminance controls (Model E reports 12 including unused chroma controls); H1 adds 9 coefficients for 19 effective scene controls. Model J has 27 matrix coefficients plus the frozen H0 controls and was not fitted. Performance times and numerical ranges per applied real pair are retained in `results/metrics.json`. STOP: no further Phase 5/model iteration is authorized by this result.
