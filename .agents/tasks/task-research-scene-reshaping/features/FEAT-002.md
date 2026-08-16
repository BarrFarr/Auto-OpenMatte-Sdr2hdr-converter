# FEAT-002: Candidate Reshaping Models

## Status: completed

## Description
Implement 7 candidate mathematical models (A through G) for scene-based SDR-to-HDR
reshaping, all following a common interface (fit/apply/param_count/name).

## Acceptance Criteria
- [x] Model A: Linear Gain (model_a_linear.py)
- [x] Model B: Piecewise Linear (model_b_piecewise.py)
- [x] Model C: Monotonic Polynomial (model_c_polynomial.py)
- [x] Model D: CDF Matching (model_d_cdf.py)
- [x] Model E: CDF + Regularization (model_e_cdf_regularized.py)
- [x] Model F: Luma + Chroma Regression (model_f_luma_chroma_regression.py)
- [x] Model G: Hybrid (model_g_hybrid.py)
- [x] models/__init__.py with get_all_models()
- [x] Comprehensive test_models.py (109 tests)
- [x] All tests pass (193 total: 84 existing + 109 new)

## Findings
- All model files must be written with bash (cat > file) since fs_write resolves to 
  /projects/sandbox/ which differs from the actual repo location at /root/
- scipy.optimize.minimize with L-BFGS-B works well for monotonic polynomial and
  CDF correction fitting
- Model G highlight shoulder uses exponential soft compression which naturally ensures
  monotonicity in the highlight region
- Regularized 3x3 matrix fitting uses (H@S^T + lambda*I*N) @ (S@S^T + lambda*I*N)^-1
  for numerical stability
- All models handle edge cases (zero input, minimal data < 10 samples, very dark/bright
  scenes) gracefully by returning identity-like defaults
- HLG warnings (4 RuntimeWarning: invalid value in log) are pre-existing from FEAT-001
