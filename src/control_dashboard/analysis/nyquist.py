from __future__ import annotations

import math
import warnings

import control as ct
import numpy as np

from control_dashboard.analysis.systems import SystemContext
from control_dashboard.domain.models import (
    AnalysisOptions,
    Diagnostic,
    FrequencyResult,
    NyquistResult,
    Severity,
    StabilityStatus,
)


def analyze_nyquist(
    context: SystemContext,
    frequency: FrequencyResult,
    options: AnalysisOptions,
) -> tuple[NyquistResult, tuple[Diagnostic, ...]]:
    diagnostics: list[Diagnostic] = []
    if frequency.omega_rad_s:
        omega_limits = (frequency.omega_rad_s[0], frequency.omega_rad_s[-1])
    else:
        omega_limits = (1e-2, 1e2)

    caught_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            omega = np.concatenate(
                ([0.0], np.logspace(math.log10(omega_limits[0]), math.log10(omega_limits[1]), options.nyquist_points))
            )
            response = ct.nyquist_response(
                context.open_loop,
                omega=omega,
                indent_direction="right",
                warn_encirclements=True,
            )
        caught_messages = [str(item.message) for item in caught]
        for message in caught_messages:
            diagnostics.append(Diagnostic("nyquist_warning", Severity.WARNING, "Nyquist", message))

        primary_array = np.asarray(response.response, dtype=complex)
        primary = tuple(complex(value) for value in primary_array)
        mirror = tuple(complex(value) for value in np.conjugate(primary_array))
        contour = np.asarray(response.contour, dtype=complex)
        contour_frequency = tuple(float(abs(value.imag)) for value in contour)
        encirclements = int(response.count)
        min_distance = float(np.min(np.abs(primary_array + 1.0))) if primary_array.size else None
        available = True
    except Exception as exc:
        diagnostics.append(
            Diagnostic("nyquist_failed", Severity.ERROR, "Nyquist", f"Falha ao calcular o contorno: {exc}")
        )
        primary = mirror = ()
        contour_frequency = ()
        encirclements = 0
        min_distance = None
        available = False

    open_rhp = int(np.count_nonzero(context.open_poles.real > context.tolerance))
    actual_closed = int(np.count_nonzero(context.closed_poles.real > context.tolerance))
    predicted_closed = open_rhp + encirclements
    consistent = available and predicted_closed == actual_closed
    critical_warning = any("encirclement" in message.lower() for message in caught_messages)

    if context.closed_stability is StabilityStatus.MARGINAL:
        conclusion = StabilityStatus.MARGINAL
    elif not consistent or critical_warning:
        conclusion = StabilityStatus.INCONCLUSIVE
    elif actual_closed == 0:
        conclusion = StabilityStatus.STABLE
    else:
        conclusion = StabilityStatus.UNSTABLE

    if min_distance is not None and math.isfinite(min_distance) and min_distance <= 1e-6:
        conclusion = StabilityStatus.MARGINAL
        diagnostics.append(
            Diagnostic(
                "nyquist_critical_point",
                Severity.WARNING,
                "Nyquist",
                "O contorno passa numericamente pelo ponto crítico −1+j0.",
            )
        )
    if context.open_poles.size and np.any(np.abs(context.open_poles.real) <= context.tolerance):
        diagnostics.append(
            Diagnostic(
                "nyquist_axis_pole",
                Severity.INFO,
                "Nyquist",
                "Polos da malha aberta sobre o eixo imaginário foram tratados com indentação à direita.",
            )
        )

    result = NyquistResult(
        available=available,
        contour_frequency_rad_s=contour_frequency,
        primary=primary,
        mirror=mirror,
        encirclements=encirclements,
        open_rhp_poles=open_rhp,
        predicted_closed_rhp_poles=predicted_closed,
        actual_closed_rhp_poles=actual_closed,
        minimum_distance_to_minus_one=min_distance,
        criterion_consistent=consistent,
        conclusion=conclusion,
    )
    return result, tuple(diagnostics)
