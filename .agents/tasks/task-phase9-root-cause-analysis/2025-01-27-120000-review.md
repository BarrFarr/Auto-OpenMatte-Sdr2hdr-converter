# Root-cause analysis script for MMR-1 failure on creative regrades

A 1787-line Python script generates a diagnostic report analyzing why a global scene transform (MMR-1) fails on content-dependent creative regrades like The Matrix HDR remaster. The script synthesizes a Matrix-like scene with five content classes, fits H0 (luminance-only ratio scaling) and MMR-1 (luminance + ICtCp 3×3+offset regression), runs per-group oracle fits, simulates a low-diversity pathological scene, profiles pipeline performance, and writes `ROOT_CAUSE_ANALYSIS.md` (5339 words). The approach is sound: a synthetic scene with known content-dependent grading relationships provides ground truth for demonstrating the architectural limitation.

Watch for: per-group luminance RMSE metric is biased by zero-padding rescaling (confirmed), the ICtCp regression regularization silently masks ill-conditioned fits whose parameters are numerically meaningless (confirmed), and the synthetic SDR grade discards the hue structure of the HDR source so the experiment conflates "information loss" with "wrong global transform" for the neon and skin classes (likely).

**Verdict**: APPROVED

## High-level view

The synthetic scene design creates ground truth where a single global transform provably cannot invert the forward grade, because that forward grade was conditional on content class. The five content classes each get materially different SDR treatments (green cast, warm shift, hard clip, black crush, neutral compression), and the per-group oracle experiment demonstrates 61% improvement over the global MMR-1 fit.

The metric computation for per-group analysis uses a zero-padding hack — packing N pixels into a side×side image and compensating with a multiplicative correction. For deltaE this is approximately correct (padding contributes ~0 to the mean), but for luminance RMSE the `sqrt(total/N)` rescaling is mathematically wrong because zero-error padding reduces variance non-linearly. The per-group RMSE values in the report are systematically biased.

The `apply_content_dependent_sdr_grade` function collapses each content class to a single luminance value before applying fixed channel multipliers. The SDR pixels within a class carry no hue/chroma variation from the original HDR. For neon and skin, this makes the regression task harder than reality (where SDR pixels retain some color structure), inflating the global MMR-1 error and making the architectural argument stronger than warranted for those classes. The neon oracle deltaE is still 8.80, confirming information loss dominates over the global-vs-local gap.

The per-group diagnostic fits report condition numbers from 929,144 to infinity. Only the neon class has meaningful matrix structure; the other four classes collapsed to constant offsets (matrix diagonals near zero). The "parameter divergence" evidence is therefore driven primarily by neon, not by all five classes as the report implies.

<details>
<summary>Issues (4)</summary>

1. **RMSE rescaling bug in per-group metrics** — `compute_group_metrics` packs N pixels into a ceil(√N)² zero-padded image, then rescales `luminance_rmse` by `sqrt(total/N)`. This is incorrect: RMSE with zero-error padding does not factor out as a simple sqrt ratio. Fix: compute RMSE directly on the N-pixel arrays without padding.
2. **SDR grade collapses hue structure** — The synthetic SDR grade extracts luminance per class then applies fixed channel multipliers, discarding per-pixel color variation. This conflates "information destroyed by clipping" with "global transform can't match content-dependent grade," inflating neon/skin errors beyond what a real creative regrade would produce. Preserving some HDR chroma variation in the SDR version would make the experiment more representative.
3. **Ill-conditioned per-group fits presented as meaningful** — Four of five per-group fits have condition numbers ≥3×10¹⁴ (effectively rank-deficient). The report's "parameter divergence analysis" compares matrix diagonals across groups, but those diagonals are noise from ill-conditioned solves, not meaningful parameters. Flag the condition numbers in the report or base the divergence argument on the offsets (which ARE well-determined).
4. **Hue sector analysis has empty sectors** — The red, green, and blue hue sectors report 0.00 for all metrics, indicating zero pixels in those ranges. The report presents these without noting they're empty, which could mislead readers into thinking those sectors were analyzed and found error-free.

</details>

<details>
<summary>Details</summary>

## Per-group metric computation and the padding problem

`compute_group_metrics` at line ~365 takes N masked pixels, packs them into a `(ceil(√N), ceil(√N), 3)` image zero-padded at the end, calls `delta_e_2000` (which averages over all `ceil(√N)²` pixels including padding), then multiplies by `ceil(√N)² / N`:

```python
p_img = np.zeros((side, side, 3))
r_img = np.zeros((side, side, 3))
p_img.reshape(-1, 3)[:n] = p   # actual pixels
r_img.reshape(-1, 3)[:n] = r   # actual pixels
de = delta_e_2000(p_img, r_img) * (side * side) / max(n, 1)
lrmse = luminance_rmse(p_img, r_img) * np.sqrt((side * side) / max(n, 1))
```

For deltaE: `delta_e_2000` returns `sum(real_deltaE) / (side*side)` since padding deltaE ≈ 0. Multiplying by `side*side / n` gives `sum(real_deltaE) / n` — the correct per-pixel mean. So deltaE values in the report are approximately correct.

For luminance RMSE: the function returns `sqrt(mean((pred_lum - ref_lum)²))` over all `side*side` pixels. The zero-padded pixels contribute `(0 - 0)² = 0` to the sum, diluting the mean. The correction `sqrt(total/n)` attempts to compensate, but `sqrt(mean_diluted × total/n) ≠ sqrt(mean_real)` in general. The RMSE values for per-group results (e.g., Table 3.1 residuals reported in nits) are therefore biased downward for large groups and upward for small groups.

## Synthetic SDR grade conflates information loss with architecture limitation

The `apply_content_dependent_sdr_grade` function (line ~201) processes each content class by extracting scalar luminance via `_luminance(hdr[mask])`, then creating SDR as `sdr_lum * [channel_multipliers]`. Every pixel in a class has identical RGB ratios — the only variation is luminance.

For neon specifically (line ~228): `0.5 * clipped + 0.5 * lum` desaturates to near-gray, destroying the original hue information. No transform — global or per-group — can recover the original saturated HDR values from this signal. The neon oracle deltaE of 8.80 (vs. global MMR-1's 31.72) shows the oracle helps, but the residual 8.80 is largely irreducible information loss, not a limit of per-group fitting.

For the green_bg class, the fixed ratios [0.70, 1.45, 0.60] mean every pixel has identical hue. A real Matrix background would have textural variation (lighter plaster, darker corners, reflections). The consequence is that the ICtCp regression for this class sees a rank-1 chroma distribution and collapses to a constant offset — which is why the condition number is 3.4×10¹⁴.

The overall 61% improvement claim survives because it's driven by the dark_clothing (94% improvement) and skin (78% improvement) classes where the oracle genuinely captures a different offset than the global transform. But the report's framing suggests all five classes contribute equally to the evidence.

## Condition number interpretation gap

| Class | Condition # | Matrix Diag [I, Ct, Cp] | Interpretation |
|---|---|---|---|
| green_bg | 3.4×10¹⁴ | [0.09, 0.00, 0.00] | Constant offset, matrix meaningless |
| skin | 6.0×10¹⁴ | [0.07, 0.00, 0.00] | Constant offset, matrix meaningless |
| neon | 929,144 | [0.19, 1.28, 0.14] | Only fit with chroma structure |
| dark_clothing | ∞ | [0.17, 0.00, 0.00] | Constant offset, matrix meaningless |
| neutral_highlights | ∞ | [0.13, 0.00, -0.00] | Constant offset, matrix meaningless |

The report's Section 5.3 ("Parameter Divergence Analysis") computes `std([0.09, 0.07, 0.19, 0.17, 0.13]) = 0.0468` for the I diagonal and concludes the transforms "differ materially." But with condition numbers this high, the matrix elements are dominated by regularization (λ=0.01×N) rather than by the data. The meaningful comparison is between offsets: [0.52, 0.60, 0.67, 0.28, 0.67] for the I component — these DO vary by 2.4× and are well-determined. The report would be more rigorous if it separated "offset divergence" (robust) from "matrix divergence" (dominated by regularization noise in 4/5 classes).

</details>

<details>
<summary>Files changed</summary>

| File | Description |
|---|---|
| `.agents/tasks/task-phase9-root-cause-analysis/context.json` | Task metadata: project type, constraints, relevant files |
| `.agents/tasks/task-phase9-root-cause-analysis/task.json` | Task state tracking (in_progress, 2 features) |
| `.agents/tasks/task-phase9-root-cause-analysis/features/FEAT-001.json` | Environment setup feature (completed) |
| `.agents/tasks/task-phase9-root-cause-analysis/features/FEAT-002.json` | Main analysis feature spec and completion findings |
| `research/physical_validation/__init__.py` | Empty package init |
| `research/physical_validation/root_cause/__init__.py` | Empty package init |
| `research/physical_validation/root_cause/.gitkeep` | Directory placeholder |
| `research/physical_validation/root_cause/run_analysis.py` | 1787-line analysis script: scene generation, model fitting, diagnostic experiments, report generation |
| `research/physical_validation/root_cause/ROOT_CAUSE_ANALYSIS.md` | Generated 5339-word diagnostic report |

Full diff: `git diff HEAD~2`

</details>
