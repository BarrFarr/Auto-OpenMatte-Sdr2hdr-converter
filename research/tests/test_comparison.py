"""Tests for the comparison pipeline: runner, experiments, report generation."""

from __future__ import annotations

import numpy as np
import pytest

from reshaping_research.comparison.runner import (
    run_comparison,
    run_single,
    ModelResult,
    ComparisonResults,
)
from reshaping_research.comparison.experiments import (
    run_all_experiments,
    strategy_comparison,
    luma_vs_full_rgb,
    hue_necessity_test,
    temporal_stability_test,
    robustness_test,
    parameter_reduction_sweep,
    ExperimentResults,
)
from reshaping_research.comparison.report_generator import generate_report
from reshaping_research.synthetic.scene_generator import SceneType


class TestRunner:
    """Tests for comparison runner."""

    def test_run_single_produces_result(self):
        """Test that run_single returns a valid ModelResult."""
        result = run_single(
            model_name="linear",
            scene_type=SceneType.NEUTRAL,
            width=64,
            height=64,
            peak_nits=4000.0,
            seed=42,
        )
        assert isinstance(result, ModelResult)
        assert result.model_name != ""
        assert result.scene_type == "neutral"
        assert result.param_count > 0
        assert result.fit_time_ms >= 0
        assert result.apply_time_ms >= 0
        assert result.center_lum_rmse_nits >= 0
        assert result.center_lum_mae_nits >= 0

    def test_run_single_all_models(self):
        """Test that each model can complete on at least one scene."""
        model_names = ["linear", "piecewise", "polynomial", "cdf",
                       "regularized", "regression", "hybrid"]
        for name in model_names:
            result = run_single(
                model_name=name,
                scene_type=SceneType.NEUTRAL,
                width=64,
                height=64,
                seed=42,
            )
            assert isinstance(result, ModelResult)
            assert result.center_lum_rmse_nits >= 0

    def test_run_comparison_small(self):
        """Test full comparison with minimal image size."""
        results = run_comparison(width=64, height=64, seed=42)
        assert isinstance(results, ComparisonResults)
        # 7 models x 6 scenes = 42 results
        assert len(results.results) == 42
        assert results.total_time_s > 0

    def test_overall_quality_bounded(self):
        """Test that overall quality score is bounded."""
        result = run_single(
            model_name="hybrid",
            scene_type=SceneType.MIXED,
            width=64,
            height=64,
            seed=42,
        )
        assert 0.0 <= result.overall_quality_score <= 1.0

    def test_seam_metrics_present(self):
        """Test that seam metrics are computed."""
        result = run_single(
            model_name="hybrid",
            scene_type=SceneType.NEUTRAL,
            width=64,
            height=64,
            seed=42,
        )
        assert 0.0 <= result.seam_top_continuity <= 1.0
        assert 0.0 <= result.seam_bottom_continuity <= 1.0

    def test_extension_metrics_present(self):
        """Test that extension metrics are computed."""
        result = run_single(
            model_name="hybrid",
            scene_type=SceneType.NEUTRAL,
            width=64,
            height=64,
            seed=42,
        )
        assert 0.0 <= result.extension_smoothness <= 1.0
        assert 0.0 <= result.extension_clipping_ratio <= 1.0


class TestExperiments:
    """Tests for additional experiments."""

    def test_strategy_comparison_runs(self):
        """Test strategy comparison completes without error."""
        results = strategy_comparison(width=64, height=64, seed=42)
        assert len(results) == 6
        for r in results:
            assert r.strategy_a_rmse >= 0
            assert r.strategy_b_rmse >= 0
            assert r.speedup_ratio > 0

    def test_luma_vs_rgb_runs(self):
        """Test luma vs RGB comparison completes."""
        results = luma_vs_full_rgb(width=64, height=64, seed=42)
        assert len(results) == 6
        for r in results:
            assert r.model_b_rmse >= 0
            assert r.model_f_rmse >= 0
            assert r.quality_ratio > 0

    def test_hue_necessity_runs(self):
        """Test hue necessity experiment completes."""
        results = hue_necessity_test(width=64, height=64, seed=42)
        assert len(results) == 6
        for r in results:
            assert r.hue_shift_mean_degrees >= 0

    def test_temporal_stability_runs(self):
        """Test temporal stability experiment completes."""
        results = temporal_stability_test(
            width=64, height=64, seed=42, n_frames=3
        )
        assert len(results) > 0
        for r in results:
            assert r.param_cv_mean >= 0
            assert r.output_rmse_std >= 0

    def test_robustness_runs(self):
        """Test robustness experiment completes."""
        results = robustness_test(width=64, height=64, seed=42)
        assert len(results) > 0
        for r in results:
            assert len(r.sample_counts) > 0
            assert len(r.rmse_values) == len(r.sample_counts)

    def test_parameter_reduction_runs(self):
        """Test parameter reduction sweep completes."""
        results = parameter_reduction_sweep(width=64, height=64, seed=42)
        assert len(results) > 0
        for r in results:
            assert len(r.configs) == 7
            assert len(r.param_counts) == 7
            assert len(r.rmse_values) == 7
            assert r.best_config != ""

    def test_run_all_experiments(self):
        """Test that run_all_experiments completes."""
        results = run_all_experiments(width=64, height=64, seed=42)
        assert isinstance(results, ExperimentResults)
        assert len(results.strategy_results) > 0
        assert len(results.luma_vs_rgb_results) > 0
        assert len(results.hue_results) > 0
        assert len(results.temporal_results) > 0
        assert len(results.robustness_results) > 0
        assert len(results.param_reduction_results) > 0
        assert results.total_time_s > 0


class TestReportGenerator:
    """Tests for report generation."""

    def test_generates_valid_markdown(self):
        """Test that report generator produces valid non-empty markdown."""
        comparison = run_comparison(width=64, height=64, seed=42)
        experiments = run_all_experiments(width=64, height=64, seed=42)
        report = generate_report(comparison, experiments)

        assert isinstance(report, str)
        assert len(report) > 1000
        assert "# Scene-Based SDR-to-HDR Reshaping" in report

    def test_report_contains_all_sections(self):
        """Test that all required sections A-L are present."""
        comparison = run_comparison(width=64, height=64, seed=42)
        experiments = run_all_experiments(width=64, height=64, seed=42)
        report = generate_report(comparison, experiments)

        required_sections = [
            "## A. Current Project Audit",
            "## B. Conceptual Redesign",
            "## C. Research: Techniques Analyzed",
            "## D. Proposed Architecture",
            "## E. Candidate Models",
            "## F. Synthetic Test Results",
            "## G. Parameter Count vs. Quality",
            "## H. Quality Metrics Detail",
            "## I. Performance",
            "## J. Experiment Results",
            "## K. Recommendation",
            "## L. Answer to Key Research Question",
        ]

        for section in required_sections:
            assert section in report, f"Missing section: {section}"

    def test_report_contains_numerical_data(self):
        """Test that report has actual numerical results."""
        comparison = run_comparison(width=64, height=64, seed=42)
        experiments = run_all_experiments(width=64, height=64, seed=42)
        report = generate_report(comparison, experiments)

        assert "Model A" in report
        assert "Model G" in report
        assert "neutral" in report
        assert "difficult" in report

        import re
        numbers = re.findall(r"\d+\.\d+", report)
        assert len(numbers) > 50

    def test_report_has_recommendation(self):
        """Test that report contains a clear recommendation."""
        comparison = run_comparison(width=64, height=64, seed=42)
        experiments = run_all_experiments(width=64, height=64, seed=42)
        report = generate_report(comparison, experiments)

        assert "Recommended for production" in report or "Best absolute quality" in report

    def test_report_answers_key_question(self):
        """Test that report answers YES or NO to the key question."""
        comparison = run_comparison(width=64, height=64, seed=42)
        experiments = run_all_experiments(width=64, height=64, seed=42)
        report = generate_report(comparison, experiments)

        assert "### Answer:" in report
        assert ("**YES**" in report or "**NO**" in report or "**CONDITIONAL**" in report)
