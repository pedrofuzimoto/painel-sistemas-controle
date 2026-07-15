from __future__ import annotations

import math

from control_dashboard.domain.models import AnalysisBundle, PoleRecord
from control_dashboard.visualization.theme import convert_frequency


def _coefficient(value: float) -> str:
    magnitude = abs(value)
    if math.isclose(magnitude, round(magnitude), rel_tol=0, abs_tol=1e-12):
        return str(int(round(magnitude)))
    return f"{magnitude:.6g}"


def polynomial_latex(coefficients: tuple[float, ...]) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, value in enumerate(coefficients):
        if value == 0:
            continue
        power = degree - index
        magnitude = _coefficient(value)
        if power == 0:
            body = magnitude
        elif power == 1:
            body = "s" if math.isclose(abs(value), 1.0) else f"{magnitude}s"
        else:
            body = f"s^{{{power}}}" if math.isclose(abs(value), 1.0) else f"{magnitude}s^{{{power}}}"
        if not terms:
            terms.append(f"-{body}" if value < 0 else body)
        else:
            terms.append((" - " if value < 0 else " + ") + body)
    return "".join(terms) if terms else "0"


def transfer_function_latex(bundle: AnalysisBundle) -> str:
    numerator = polynomial_latex(bundle.normalized_spec.numerator)
    denominator = polynomial_latex(bundle.normalized_spec.denominator)
    return rf"G(s)=\frac{{{numerator}}}{{{denominator}}}"


def format_number(value: float | None, unit: str = "", precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "∞"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{precision}g}{suffix}"


def format_frequency(omega: float | None, unit: str) -> str:
    return format_number(convert_frequency(omega, unit), unit)


def pole_table(records: tuple[PoleRecord, ...]) -> list[dict[str, str | int]]:
    return [
        {
            "Real": f"{record.real:.7g}",
            "Imaginária": f"{record.imag:+.7g}",
            "ωn (rad/s)": f"{record.magnitude:.7g}",
            "ζ": "N/A" if record.damping_ratio is None else f"{record.damping_ratio:.5g}",
            "Multiplicidade": record.multiplicity,
        }
        for record in records
    ]


def selected_pole_table(poles: tuple[complex, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pole in sorted(poles, key=lambda value: (value.real, value.imag)):
        magnitude = abs(pole)
        damping = -pole.real / magnitude if magnitude > 1e-12 else None
        rows.append(
            {
                "Polo": f"{pole.real:.7g} {pole.imag:+.7g}j",
                "ωn (rad/s)": f"{magnitude:.7g}",
                "ζ": "N/A" if damping is None else f"{damping:.5g}",
            }
        )
    return rows
