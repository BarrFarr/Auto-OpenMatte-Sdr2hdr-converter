# FEAT-001: Research Module Scaffold

## Status: completed

## Description
Create the research/ experimental module scaffold with all infrastructure for
the scene-based SDR-to-HDR reshaping research project.

## Acceptance Criteria
- [x] research/pyproject.toml with Python >=3.9, all required dependencies
- [x] utils/transfer_functions.py with PQ, HLG, BT.1886 (adapted from production)
- [x] utils/color_spaces.py with BT.709/2020 matrices, ICtCp, Lab conversions
- [x] metrics/luminance_metrics.py (MAE, RMSE, percentile, segmented)
- [x] metrics/color_metrics.py (deltaE2000, deltaE ICtCp, chroma, hue)
- [x] metrics/spatial_metrics.py (gradient continuity, edge preservation, seam)
- [x] metrics/temporal_metrics.py (parameter variance, flicker risk)
- [x] synthetic/scene_generator.py (6 scene types)
- [x] synthetic/sdr_simulator.py (controlled HDR->SDR)
- [x] synthetic/open_matte_simulator.py (wider SDR with HDR center crop)
- [x] models/ and comparison/ placeholder packages
- [x] Full test suite with 84 passing tests
- [x] Module installs and all tests pass

## Findings
- numpy 2.0.2 installed (compatible with Python 3.9)
- scipy 1.13.1 installed
- 4 HLG warnings (RuntimeWarning: invalid value in log) are expected - HLG log branch
  handles edge case via np.where but numpy evaluates both branches. Does not affect
  correctness as np.where selects the correct result.
- Used Tuple from typing module instead of tuple[] for Python 3.9 compatibility
  in luminance_metrics.py type hints
