from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from control_dashboard.domain.diagnostics import AnalysisValidationError
from control_dashboard.domain.models import (
    AnalysisOptions,
    Diagnostic,
    Severity,
    TransferFunctionSpec,
)

_TOKEN_SPLIT = re.compile(r"[;\s]+")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    normalized_spec: TransferFunctionSpec
    diagnostics: tuple[Diagnostic, ...]


def parse_coefficients(text: str, field: str) -> tuple[float, ...]:
    cleaned = text.strip().strip("[]()")
    if not cleaned:
        raise AnalysisValidationError(
            [
                Diagnostic(
                    f"{field}_empty",
                    Severity.ERROR,
                    "entrada",
                    f"Informe os coeficientes do {field}.",
                )
            ]
        )

    raw_tokens = [token for token in _TOKEN_SPLIT.split(cleaned) if token]
    values: list[float] = []
    for token in raw_tokens:
        normalized_token = token.replace(",", ".")
        try:
            value = float(normalized_token)
        except ValueError as exc:
            raise AnalysisValidationError(
                [
                    Diagnostic(
                        f"{field}_not_numeric",
                        Severity.ERROR,
                        "entrada",
                        f"Coeficiente inválido no {field}: “{token}”. Use ponto e vírgula ou espaço entre valores.",
                    )
                ]
            ) from exc
        if not math.isfinite(value):
            raise AnalysisValidationError(
                [
                    Diagnostic(
                        f"{field}_not_finite",
                        Severity.ERROR,
                        "entrada",
                        f"O {field} contém NaN ou infinito.",
                    )
                ]
            )
        values.append(value)
    return tuple(values)


def parse_transfer_function_text(
    numerator_text: str,
    denominator_text: str,
    name: str = "G",
) -> TransferFunctionSpec:
    diagnostics: list[Diagnostic] = []
    parsed: dict[str, tuple[float, ...]] = {}
    for field, text in (
        ("numerador", numerator_text),
        ("denominador", denominator_text),
    ):
        try:
            parsed[field] = parse_coefficients(text, field)
        except AnalysisValidationError as exc:
            diagnostics.extend(exc.diagnostics)
    if diagnostics:
        raise AnalysisValidationError(diagnostics)
    return TransferFunctionSpec(parsed["numerador"], parsed["denominador"], name)


def _trim_leading_zeros(values: tuple[float, ...]) -> tuple[float, ...]:
    first = 0
    while first < len(values) and values[first] == 0.0:
        first += 1
    return values[first:]


def _conditioning_ratio(values: tuple[float, ...]) -> float:
    nonzero = [abs(value) for value in values if value != 0.0]
    if not nonzero:
        return math.inf
    return max(nonzero) / min(nonzero)


def _cancellation_count(num: tuple[float, ...], den: tuple[float, ...]) -> int:
    if len(num) <= 1 or len(den) <= 1:
        return 0
    zeros = np.roots(num)
    poles = np.roots(den)
    count = 0
    remaining = list(poles)
    for zero in zeros:
        if not remaining:
            break
        distances = [abs(zero - pole) / max(1.0, abs(zero), abs(pole)) for pole in remaining]
        index = int(np.argmin(distances))
        if distances[index] < 1e-6:
            count += 1
            remaining.pop(index)
    return count


def validate_spec(spec: TransferFunctionSpec) -> ValidationResult:
    diagnostics: list[Diagnostic] = []
    numerator = _trim_leading_zeros(tuple(float(value) for value in spec.numerator))
    denominator = _trim_leading_zeros(tuple(float(value) for value in spec.denominator))

    if not numerator:
        diagnostics.append(
            Diagnostic(
                "zero_numerator",
                Severity.ERROR,
                "entrada",
                "O numerador não pode ser totalmente nulo.",
            )
        )
    if not denominator:
        diagnostics.append(
            Diagnostic(
                "zero_denominator",
                Severity.ERROR,
                "entrada",
                "O denominador não pode ser totalmente nulo.",
            )
        )
    if diagnostics:
        raise AnalysisValidationError(diagnostics)

    if any(not math.isfinite(value) for value in numerator + denominator):
        raise AnalysisValidationError(
            [Diagnostic("not_finite", Severity.ERROR, "entrada", "Todos os coeficientes devem ser reais e finitos.")]
        )

    numerator_degree = len(numerator) - 1
    denominator_degree = len(denominator) - 1
    if numerator_degree > denominator_degree:
        diagnostics.append(
            Diagnostic(
                "improper_system",
                Severity.ERROR,
                "entrada",
                "O sistema é impróprio: o grau do numerador excede o grau do denominador.",
            )
        )
    if denominator_degree > 20:
        diagnostics.append(
            Diagnostic(
                "order_too_high",
                Severity.ERROR,
                "entrada",
                "A ordem máxima aceita na v1 é 20.",
            )
        )
    elif denominator_degree > 10:
        diagnostics.append(
            Diagnostic(
                "high_order",
                Severity.WARNING,
                "entrada",
                "Sistemas acima da ordem 10 podem apresentar condicionamento numérico ruim.",
            )
        )
    if any(item.severity is Severity.ERROR for item in diagnostics):
        raise AnalysisValidationError(diagnostics)

    lead = denominator[0]
    normalized_num = tuple(value / lead for value in numerator)
    normalized_den = tuple(value / lead for value in denominator)

    if max(_conditioning_ratio(normalized_num), _conditioning_ratio(normalized_den)) > 1e12:
        diagnostics.append(
            Diagnostic(
                "ill_conditioned_coefficients",
                Severity.WARNING,
                "entrada",
                "A razão entre coeficientes não nulos excede 10¹²; interprete raízes e margens com cautela.",
            )
        )

    cancellations = _cancellation_count(normalized_num, normalized_den)
    if cancellations:
        diagnostics.append(
            Diagnostic(
                "pole_zero_cancellation",
                Severity.WARNING,
                "entrada",
                f"Foram detectados {cancellations} possível(is) cancelamento(s) polo-zero. O modelo não será reduzido automaticamente.",
            )
        )

    normalized = TransferFunctionSpec(normalized_num, normalized_den, spec.name)
    return ValidationResult(normalized, tuple(diagnostics))


def validate_options(options: AnalysisOptions) -> tuple[Diagnostic, ...]:
    errors: list[Diagnostic] = []
    if options.frequency_unit not in {"Hz", "rad/s"}:
        errors.append(Diagnostic("frequency_unit", Severity.ERROR, "opções", "Unidade de frequência inválida."))
    if (options.frequency_min_rad_s is None) != (options.frequency_max_rad_s is None):
        errors.append(
            Diagnostic(
                "frequency_pair",
                Severity.ERROR,
                "opções",
                "Informe os dois limites de frequência ou use a faixa automática.",
            )
        )
    if options.frequency_min_rad_s is not None and options.frequency_max_rad_s is not None:
        if options.frequency_min_rad_s <= 0 or options.frequency_min_rad_s >= options.frequency_max_rad_s:
            errors.append(
                Diagnostic(
                    "frequency_range", Severity.ERROR, "opções", "A faixa de frequência deve ser positiva e crescente."
                )
            )
    if not 200 <= options.frequency_points <= 5000:
        errors.append(Diagnostic("frequency_points", Severity.ERROR, "opções", "Use entre 200 e 5.000 pontos no Bode."))
    if not 500 <= options.nyquist_points <= 10000:
        errors.append(
            Diagnostic("nyquist_points", Severity.ERROR, "opções", "Use entre 500 e 10.000 pontos no Nyquist.")
        )
    if not 0 < options.settling_threshold < 1:
        errors.append(
            Diagnostic(
                "settling_threshold", Severity.ERROR, "opções", "A banda de acomodação deve ficar entre 0 e 100%."
            )
        )
    if not 0 <= options.rise_lower < options.rise_upper <= 1:
        errors.append(
            Diagnostic("rise_limits", Severity.ERROR, "opções", "Os limites do tempo de subida são inválidos.")
        )
    if options.time_end_s is not None and options.time_end_s <= 0:
        errors.append(Diagnostic("time_end", Severity.ERROR, "opções", "O horizonte temporal deve ser positivo."))
    if options.time_points is not None and not 100 <= options.time_points <= 10000:
        errors.append(Diagnostic("time_points", Severity.ERROR, "opções", "Use entre 100 e 10.000 pontos no tempo."))
    if options.root_gain_selected < 0:
        errors.append(Diagnostic("root_gain", Severity.ERROR, "opções", "O ganho selecionado deve ser não negativo."))
    if options.root_gain_max is not None and options.root_gain_max <= 0:
        errors.append(Diagnostic("root_gain_max", Severity.ERROR, "opções", "O ganho máximo deve ser positivo."))
    if errors:
        raise AnalysisValidationError(errors)
    return ()
