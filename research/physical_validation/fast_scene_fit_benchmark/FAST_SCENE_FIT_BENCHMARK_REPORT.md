# Phase 7 — Fast Scene-Fit Benchmark for Frozen MMR-1

**Status:** `COMPLETE`

Research-only deterministic Matrix 08 benchmark. Frozen MMR-1 equations, ICtCp representation, Model-E mapping, ridge objective, constraints, and final luminance reimposition were not modified. The 90% frame remained a temporal holdout for B1–B4; B5 is diagnostic-only and cannot support the recommendation.

## Reference reproducibility

Phase-6 B4 coefficient SHA-256: `63AA8E9A02CA5600A16B5A3B59C3CE8F1AABFF246089AE0178E4D28882B4B50D`. The re-decode/re-fit control passed before the reduced-budget sweep.

## Sweep

| Frames | Rows/frame | Total rows | Fit median s | Fit worst s | Peak RSS worst MiB | Worst frame mean ΔE vs ref | Worst frame P95 ΔE vs ref | Equivalent |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1 | 500 | 500 | 0.7917 | 0.8215 | 1161.3 | 0.636412 | 1.250427 | False |
| B1 | 1000 | 1000 | 0.8224 | 0.8313 | 1162.5 | 0.607724 | 1.231871 | False |
| B1 | 2500 | 2500 | 0.8385 | 0.8449 | 1163.1 | 0.640632 | 1.325242 | False |
| B1 | 5000 | 5000 | 0.8922 | 0.8978 | 1163.2 | 0.654139 | 1.368666 | False |
| B1 | 10000 | 10000 | 0.8514 | 0.8576 | 1162.8 | 0.614873 | 1.271962 | False |
| B1 | 25000 | 25000 | 0.8702 | 0.9110 | 1163.3 | 0.617174 | 1.277125 | False |
| B1 | 50000 | 50000 | 0.9435 | 0.9445 | 1160.8 | 0.602592 | 1.281204 | False |
| B2 | 500 | 1000 | 1.6806 | 1.7211 | 1315.1 | 0.259840 | 0.387384 | False |
| B2 | 1000 | 2000 | 1.6532 | 1.6729 | 1314.5 | 0.252632 | 0.485245 | False |
| B2 | 2500 | 5000 | 1.6884 | 1.7410 | 1314.2 | 0.238399 | 0.387026 | False |
| B2 | 5000 | 10000 | 1.6879 | 1.7550 | 1315.7 | 0.229827 | 0.389613 | False |
| B2 | 10000 | 20000 | 1.7065 | 1.7070 | 1316.1 | 0.269229 | 0.440156 | False |
| B2 | 25000 | 50000 | 1.7228 | 1.8930 | 1317.7 | 0.221648 | 0.384541 | False |
| B2 | 50000 | 100000 | 1.8785 | 1.9057 | 1317.9 | 0.240541 | 0.395975 | False |
| B3 | 500 | 1500 | 2.4544 | 2.5157 | 1362.6 | 0.181426 | 0.366605 | False |
| B3 | 1000 | 3000 | 2.5534 | 2.5888 | 1363.5 | 0.147240 | 0.330781 | False |
| B3 | 2500 | 7500 | 2.4738 | 2.6473 | 1361.8 | 0.164042 | 0.296060 | False |
| B3 | 5000 | 15000 | 2.5708 | 2.5933 | 1363.3 | 0.103997 | 0.185218 | False |
| B3 | 10000 | 30000 | 2.6372 | 2.6898 | 1363.0 | 0.154402 | 0.327204 | False |
| B3 | 25000 | 75000 | 2.8817 | 3.1011 | 1365.8 | 0.144734 | 0.298196 | False |
| B3 | 50000 | 150000 | 2.8713 | 2.8741 | 1370.5 | 0.126719 | 0.249070 | False |
| B4 | 500 | 2000 | 3.2370 | 3.2537 | 1410.1 | 0.083344 | 0.235910 | False |
| B4 | 1000 | 4000 | 3.3236 | 3.4513 | 1411.0 | 0.068336 | 0.313491 | False |
| B4 | 2500 | 10000 | 3.3739 | 3.4714 | 1411.3 | 0.063507 | 0.227514 | False |
| B4 | 5000 | 20000 | 4.0778 | 4.2167 | 1411.2 | 0.051434 | 0.160642 | False |
| B4 | 10000 | 40000 | 3.5222 | 3.6116 | 1412.5 | 0.055512 | 0.082166 | False |
| B4 | 25000 | 100000 | 3.6872 | 3.7803 | 1416.3 | 0.047588 | 0.080669 | True |
| B4 | 50000 | 200000 | 3.8105 | 3.8729 | 1420.8 | 0.046445 | 0.091586 | True |
| B5 | 500 | 2500 | 5.0684 | 5.1888 | 1457.1 | 0.165696 | 0.921080 | False |
| B5 | 1000 | 5000 | 5.1326 | 5.1564 | 1458.1 | 0.197418 | 1.133030 | False |
| B5 | 2500 | 12500 | 5.1714 | 5.2028 | 1458.4 | 0.196214 | 1.168833 | False |
| B5 | 5000 | 25000 | 5.3580 | 5.3917 | 1458.3 | 0.195801 | 1.039643 | False |
| B5 | 10000 | 50000 | 4.6332 | 4.6858 | 1460.6 | 0.185335 | 1.219369 | False |
| B5 | 25000 | 125000 | 5.4417 | 5.4819 | 1465.0 | 0.188758 | 1.007098 | False |
| B5 | 50000 | 250000 | 4.9864 | 5.1768 | 1470.9 | 0.194382 | 1.164184 | False |

## Required A–H recommendation

A. **Representative frames per shot:** `4` (10%, 25%, 50%, 75%).
B. **Fit rows per representative frame:** `25000`.
C. **Total fit rows per shot:** `100000`.
D. **Fitting time:** median `3.6872 s`, worst `3.7803 s` (warm-cache H0 + pooled MMR fitting only).
E. **Fit-only projection:** 100/500/1000 shots = `368.7/1843.6/3687.2 s` median and `378.0/1890.2/3780.3 s` worst. Decode/application are separately recorded and not included.
F. **Measured peak RSS:** median `1415.2 MiB`, worst `1416.3 MiB`; GPU memory was not measured and is not inferred.
G. **Expected quality difference versus reference:** worst tested frame mean ΔE2000 `0.047588`, P95 `0.080669`, within predeclared limits 0.05/0.15.
H. **RTX 3080 / CPU practicality:** the measured fit runs on CPU and requires no GPU-memory claim. It is practical only insofar as the reported fit-only and measured RSS budgets suit the local CPU workflow; full decode/application remains a separate throughput cost.

## Cross-scene sanity

`EXCLUDED_BY_PHASE6_STATUS`: Matrix 03 was `ANCHOR_NEAR_CUT`, Matrix 12 `LOCAL_SHOT_UNRESOLVED`, and BR2049 01 `ANCHOR_NEAR_CUT` with conditional geometry. No new anchor, sync, geometry, or source window was used.

## Stop

STOP after this report. No model modification, production integration, full-shot render, or full-film processing is authorized.
