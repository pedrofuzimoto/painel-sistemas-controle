from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from control_dashboard import AnalysisOptions, TransferFunctionSpec, analyze
from control_dashboard.domain.models import StabilityStatus

REFERENCE_PATH = Path(__file__).with_name("nise_6e_cases.json")
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in REFERENCE["cases"]}


def _analyze_case(case_id: str, **option_overrides: float | int) -> object:
    case = CASES[case_id]
    model = case["application_input"]
    option_values: dict[str, float | int] = {"frequency_points": 700, "nyquist_points": 2000}
    option_values.update(option_overrides)
    options = AnalysisOptions(**option_values)
    return analyze(
        TransferFunctionSpec(
            tuple(model["numerator"]),
            tuple(model["denominator"]),
            name=f"Nise {case['example']}",
        ),
        options,
    )


def test_nise_example_4_3_natural_frequency_and_damping() -> None:
    case = CASES["nise_4_3"]
    expected = case["expected"]
    bundle = _analyze_case(case["id"])

    assert bundle.closed_loop_stability is StabilityStatus.STABLE
    assert len(bundle.pole_zero.closed_poles) == 2
    for pole in bundle.pole_zero.closed_poles:
        assert pole.magnitude == pytest.approx(expected["natural_frequency_rad_s"], rel=1e-12)
        assert pole.damping_ratio == pytest.approx(expected["damping_ratio"], rel=1e-12)


def test_nise_example_4_5_second_order_step_metrics() -> None:
    case = CASES["nise_4_5"]
    expected = case["expected"]
    bundle = _analyze_case(case["id"])
    metrics = bundle.step.metrics

    assert metrics.peak_time_s == pytest.approx(expected["peak_time_s"], abs=0.002)
    assert metrics.overshoot_percent == pytest.approx(expected["overshoot_percent"], abs=0.01)
    assert metrics.rise_time_s == pytest.approx(expected["rise_time_approx_s"], abs=0.01)

    # O livro usa Ts ~= 4/(zeta*wn); o painel mede a permanência real na banda de 2%.
    assert metrics.settling_time_s is not None
    assert abs(metrics.settling_time_s - expected["settling_time_approx_s"]) < 0.05

    time = np.asarray(bundle.step.time_s)
    output = np.asarray(bundle.step.output)
    settling_index = int(np.searchsorted(time, metrics.settling_time_s))
    tolerance = 0.02 * abs(metrics.steady_state_value)
    assert np.all(np.abs(output[settling_index:] - metrics.steady_state_value) <= tolerance + 1e-6)


def test_nise_example_7_1_steady_state_error() -> None:
    case = CASES["nise_7_1"]
    expected = case["expected"]
    bundle = _analyze_case(case["id"])
    metrics = bundle.step.metrics

    assert metrics.steady_state_value == pytest.approx(expected["steady_state_value"], rel=1e-12)
    assert metrics.steady_state_error == pytest.approx(expected["steady_state_error"], rel=1e-12)


def test_nise_example_8_5_root_locus_imaginary_axis_crossing() -> None:
    case = CASES["nise_8_5"]
    expected = case["expected"]
    bundle = _analyze_case(
        case["id"],
        root_gain_selected=expected["critical_gain"],
        root_gain_max=12.0,
    )

    assert bundle.root_locus.applicable
    assert bundle.root_locus.unity_consistent
    assert expected["critical_gain"] in bundle.root_locus.gains
    assert bundle.frequency.gain_margin == pytest.approx(expected["critical_gain"], abs=0.01)
    assert bundle.frequency.phase_crossover_rad_s == pytest.approx(
        expected["imaginary_axis_frequency_rad_s"], abs=0.01
    )

    crossing_pair = sorted(bundle.root_locus.selected_poles, key=lambda pole: abs(pole.real))[:2]
    assert all(abs(pole.real) < 0.001 for pole in crossing_pair)
    assert sorted(abs(pole.imag) for pole in crossing_pair) == pytest.approx(
        [expected["imaginary_axis_frequency_rad_s"]] * 2,
        abs=0.01,
    )


def test_nise_example_10_7_nyquist_stability_boundary() -> None:
    case = CASES["nise_10_7"]
    expected = case["expected"]
    base = _analyze_case(case["id"], frequency_points=1000, nyquist_points=4000)

    assert base.closed_loop_stability is StabilityStatus.STABLE
    assert base.frequency.gain_margin == pytest.approx(expected["critical_gain"], rel=1e-12)
    assert base.frequency.phase_crossover_rad_s == pytest.approx(
        expected["imaginary_axis_frequency_rad_s"], rel=1e-12
    )
    assert base.nyquist.open_rhp_poles == 0
    assert base.nyquist.encirclements == 0
    assert base.nyquist.criterion_consistent

    model = case["application_input"]
    denominator = tuple(model["denominator"])
    critical = analyze(
        TransferFunctionSpec((expected["critical_gain"],), denominator, "Nise 10.7 K crítico"),
        AnalysisOptions(frequency_points=700, nyquist_points=2000),
    )
    above = analyze(
        TransferFunctionSpec((expected["critical_gain"] + 1.0,), denominator, "Nise 10.7 acima do limite"),
        AnalysisOptions(frequency_points=500, nyquist_points=1500),
    )

    assert critical.closed_loop_stability is StabilityStatus.MARGINAL
    assert critical.nyquist.conclusion is StabilityStatus.MARGINAL
    assert critical.step.metrics.settling_time_s is None
    axis_poles = [pole for pole in critical.pole_zero.closed_poles if abs(pole.real) < 1e-7]
    assert sorted(abs(pole.imag) for pole in axis_poles) == pytest.approx(
        [expected["imaginary_axis_frequency_rad_s"]] * 2,
        rel=1e-10,
    )
    assert above.closed_loop_stability is StabilityStatus.UNSTABLE


def test_nise_example_10_9_exact_bode_gain_margin() -> None:
    case = CASES["nise_10_9"]
    expected = case["expected"]
    bundle = _analyze_case(case["id"], frequency_points=1000, nyquist_points=3000)

    critical_gain = expected["initial_gain"] * bundle.frequency.gain_margin
    assert bundle.closed_loop_stability is StabilityStatus.STABLE
    assert critical_gain == pytest.approx(expected["critical_gain_exact_response"], rel=1e-12)
    assert bundle.frequency.phase_crossover_rad_s == pytest.approx(expected["phase_crossover_rad_s"], abs=0.01)
    assert bundle.frequency.gain_margin_db == pytest.approx(expected["gain_margin_db_rounded"], abs=0.6)
    assert bundle.nyquist.criterion_consistent
    assert math.isfinite(bundle.frequency.gain_margin_db)
