from __future__ import annotations

import pytest

from control_dashboard.domain.diagnostics import AnalysisValidationError
from control_dashboard.domain.models import TransferFunctionSpec
from control_dashboard.domain.validation import parse_transfer_function_text, validate_spec


def test_parser_accepts_decimal_comma_semicolon_spaces_and_scientific_notation() -> None:
    spec = parse_transfer_function_text("0,001144; 10,4", "22e-9 2e-4 1")
    assert spec.numerator == pytest.approx((0.001144, 10.4))
    assert spec.denominator == pytest.approx((22e-9, 2e-4, 1.0))


def test_validation_removes_only_leading_zeros_and_normalizes_denominator() -> None:
    result = validate_spec(TransferFunctionSpec((0, 2, 0), (0, 2, 4, 0)))
    assert result.normalized_spec.numerator == pytest.approx((1.0, 0.0))
    assert result.normalized_spec.denominator == pytest.approx((1.0, 2.0, 0.0))


@pytest.mark.parametrize(
    ("numerator", "denominator", "code"),
    [
        ((0,), (1,), "zero_numerator"),
        ((1,), (0,), "zero_denominator"),
        ((1, 2), (1,), "improper_system"),
        ((1,), tuple([1.0] + [0.0] * 21), "order_too_high"),
    ],
)
def test_invalid_models_raise_structured_diagnostics(numerator, denominator, code) -> None:
    with pytest.raises(AnalysisValidationError) as caught:
        validate_spec(TransferFunctionSpec(numerator, denominator))
    assert code in {item.code for item in caught.value.diagnostics}


def test_non_numeric_and_non_finite_inputs_are_rejected() -> None:
    with pytest.raises(AnalysisValidationError) as non_numeric:
        parse_transfer_function_text("1; s", "1; 2")
    assert non_numeric.value.diagnostics[0].code == "numerador_not_numeric"

    with pytest.raises(AnalysisValidationError) as non_finite:
        parse_transfer_function_text("NaN", "1")
    assert non_finite.value.diagnostics[0].code == "numerador_not_finite"


def test_cancellation_and_conditioning_generate_warnings_without_reduction() -> None:
    cancellation = validate_spec(TransferFunctionSpec((1, 1), (1, 3, 2)))
    assert "pole_zero_cancellation" in {item.code for item in cancellation.diagnostics}
    assert cancellation.normalized_spec.numerator == (1.0, 1.0)

    conditioning = validate_spec(TransferFunctionSpec((1e-15, 1), (1, 1)))
    assert "ill_conditioned_coefficients" in {item.code for item in conditioning.diagnostics}
