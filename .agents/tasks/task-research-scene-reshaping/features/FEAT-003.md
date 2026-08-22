# FEAT-003: Full Experimental Comparison Pipeline

## Status: completed

## Description
Implement the full experimental comparison pipeline: run all 7 models on all 6 synthetic scenes, compute all metrics, perform additional experiments, and generate the comprehensive research report.

## Files Created
- research/src/reshaping_research/comparison/runner.py
- research/src/reshaping_research/comparison/experiments.py
- research/src/reshaping_research/comparison/report_generator.py
- research/src/reshaping_research/run_comparison.py
- research/src/reshaping_research/__main__.py
- research/tests/test_comparison.py
- research/reports/RESEARCH_REPORT.md (generated)

## Acceptance Criteria (all met)
- All 7 models tested on all 6 scenes with full metrics
- Additional experiments complete (strategy, hue, temporal, robustness, parameter reduction)
- Comprehensive report generated with all sections A-L
- Tests pass for runner, experiments, and report generation (18 new tests, 211 total)
- Total runtime 6.7s on CPU with 256x256 images (well under 5 minutes)

## Findings
- Best overall quality score: Model D: CDF Matching (0.7113)
- Lowest RMSE: Model C: Monotonic Polynomial (111.48 nits avg)
- All models cluster relatively close in quality (overall score 0.60-0.71)
- Temporal stability excellent across all models (CV < 0.01)
- Report answers key question with YES at HIGH confidence
- Strategy B (representative subsampling) works nearly as well as full pixel fitting
- Hue correction provides marginal improvement (< 2% average)
