from __future__ import annotations

import math

import pytest

from control_dashboard import AnalysisOptions, TransferFunctionSpec, analyze
from control_dashboard.domain.models import StabilityStatus

FAST = AnalysisOptions(frequency_points=400, nyquist_points=1000)


def test_first_order_closed_loop_step_metrics() -> None:
    bundle = analyze(TransferFunctionSpec((1,), (1, 1)), FAST)
    metrics = bundle.step.metrics
    assert bundle.closed_loop_stability is StabilityStatus.STABLE
    assert metrics.steady_state_value == pytest.approx(0.5)
    assert metrics.steady_state_error == pytest.approx(0.5)
    assert metrics.overshoot_percent == pytest.approx(0.0, abs=1e-10)


def test_second_order_reference_margins_and_overshoot() -> None:
    bundle = analyze(TransferFunctionSpec((1,), (1, 1, 0)), FAST)
    assert math.isinf(bundle.frequency.gain_margin)
    assert bundle.frequency.phase_margin_deg == pytest.approx(51.82729237, rel=1e-6)
    assert bundle.step.metrics.steady_state_error == pytest.approx(0.0, abs=1e-12)
    assert bundle.step.metrics.overshoot_percent == pytest.approx(16.3028, rel=2e-4)
    assert bundle.nyquist.conclusion is StabilityStatus.STABLE


def test_nyquist_sign_convention_for_stabilized_unstable_plant() -> None:
    bundle = analyze(TransferFunctionSpec((3,), (1, 1, -2)), FAST)
    assert bundle.nyquist.open_rhp_poles == 1
    assert bundle.nyquist.encirclements == -1
    assert bundle.nyquist.predicted_closed_rhp_poles == 0
    assert bundle.nyquist.actual_closed_rhp_poles == 0
    assert bundle.nyquist.criterion_consistent
    assert bundle.closed_loop_stability is StabilityStatus.STABLE


def test_nyquist_detects_unstable_and_marginal_closed_loops() -> None:
    unstable = analyze(TransferFunctionSpec((10,), (1, 3, 3, 1)), FAST)
    assert unstable.nyquist.encirclements == 2
    assert unstable.nyquist.actual_closed_rhp_poles == 2
    assert unstable.closed_loop_stability is StabilityStatus.UNSTABLE
    assert unstable.step.metrics.overshoot_percent is None

    marginal = analyze(TransferFunctionSpec((8,), (1, 3, 3, 1)), FAST)
    assert marginal.closed_loop_stability is StabilityStatus.MARGINAL
    assert marginal.nyquist.conclusion is StabilityStatus.MARGINAL
    assert marginal.step.metrics.settling_time_s is None


def test_root_locus_selected_gain_and_unity_consistency() -> None:
    options = AnalysisOptions(frequency_points=300, nyquist_points=800, root_gain_selected=2.0)
    bundle = analyze(TransferFunctionSpec((1,), (1, 2, 0)), options)
    assert bundle.root_locus.unity_consistent
    assert sorted(bundle.root_locus.unity_poles, key=lambda value: value.real) == pytest.approx((-1 + 0j, -1 + 0j))
    assert sorted(bundle.root_locus.selected_poles, key=lambda value: value.imag) == pytest.approx((-1 - 1j, -1 + 1j))


def test_static_gain_is_valid_and_root_locus_is_not_applicable() -> None:
    bundle = analyze(TransferFunctionSpec((2,), (1,)), FAST)
    assert bundle.closed_loop_stability is StabilityStatus.STABLE
    assert not bundle.root_locus.applicable
    assert bundle.step.metrics.steady_state_value == pytest.approx(2 / 3)
    assert bundle.step.metrics.steady_state_error == pytest.approx(1 / 3)


def test_buck_reference_model_preserves_expected_poles_and_zero() -> None:
    bundle = analyze(
        TransferFunctionSpec((0.001144, 10.4), (22e-9, 2e-4, 1)),
        FAST,
    )
    assert len(bundle.pole_zero.open_poles) == 2
    assert len(bundle.pole_zero.open_zeros) == 1
    assert bundle.pole_zero.open_zeros[0].real == pytest.approx(-9090.9090909, rel=1e-7)
