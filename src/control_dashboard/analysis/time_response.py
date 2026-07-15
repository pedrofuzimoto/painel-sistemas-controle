from __future__ import annotations

import math
import warnings

import control as ct
import numpy as np

from control_dashboard.analysis.systems import SystemContext
from control_dashboard.domain.models import (
    AnalysisOptions,
    Diagnostic,
    Severity,
    StabilityStatus,
    StepMetrics,
    StepResult,
)


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(np.asarray(value).reshape(-1)[0])
    except (ValueError, TypeError, IndexError):
        return None
    return result if math.isfinite(result) else None


def _time_grid(context: SystemContext, options: AnalysisOptions) -> tuple[np.ndarray, bool]:
    poles = context.closed_poles
    hit_cap = False
    if options.time_end_s is not None:
        time_end = float(options.time_end_s)
    elif not poles.size:
        time_end = 1.0
    elif context.closed_stability is StabilityStatus.STABLE:
        decay_rates = -poles.real[poles.real < -context.tolerance]
        alpha = float(np.min(decay_rates)) if decay_rates.size else 1.0
        time_end = 8.0 / alpha
    elif context.closed_stability is StabilityStatus.UNSTABLE:
        growth_rates = poles.real[poles.real > context.tolerance]
        rate = float(np.max(growth_rates)) if growth_rates.size else max(float(np.max(np.abs(poles))), 1.0)
        time_end = 5.0 / rate
    else:
        frequencies = np.abs(poles.imag[np.abs(poles.imag) > context.tolerance])
        time_end = 10.0 * math.pi / float(np.min(frequencies)) if frequencies.size else 10.0

    if options.time_points is not None:
        points = int(options.time_points)
    elif not poles.size:
        points = 1000
    else:
        beta = max(float(np.max(np.abs(poles))), 1.0 / max(time_end, 1e-12))
        desired = int(math.ceil(50.0 * beta * time_end))
        points = min(max(desired, 1000), 10000)
        hit_cap = desired > 10000
    return np.linspace(0.0, time_end, points), hit_cap


def analyze_step(
    context: SystemContext,
    options: AnalysisOptions,
) -> tuple[StepResult, tuple[Diagnostic, ...]]:
    diagnostics: list[Diagnostic] = []
    time, hit_cap = _time_grid(context, options)
    if hit_cap:
        diagnostics.append(
            Diagnostic(
                "time_resolution_cap",
                Severity.WARNING,
                "Resposta temporal",
                "A malha possui escalas de tempo muito separadas; a amostragem automática atingiu 10.000 pontos.",
            )
        )

    steady = _finite_or_none(ct.dcgain(context.closed_loop))
    try:
        if not context.closed_poles.size:
            output = np.full_like(time, steady if steady is not None else 0.0)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                response = ct.step_response(context.closed_loop, T=time)
            output = np.squeeze(np.asarray(response.outputs, dtype=float))
            for warning in caught:
                diagnostics.append(
                    Diagnostic("step_warning", Severity.WARNING, "Resposta temporal", str(warning.message))
                )
        available = True
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                "step_failed",
                Severity.ERROR,
                "Resposta temporal",
                f"Falha ao simular a resposta ao degrau: {exc}",
            )
        )
        time = np.asarray([], dtype=float)
        output = np.asarray([], dtype=float)
        available = False

    metrics = StepMetrics(None, None, None, None, None, steady, None)
    lower = upper = None
    if available and context.closed_stability is StabilityStatus.STABLE and steady is not None:
        steady_error = 1.0 - steady
        if not context.closed_poles.size:
            metrics = StepMetrics(0.0, 0.0, 0.0, abs(steady), 0.0, steady, steady_error)
        elif abs(steady) <= 1e-12:
            metrics = StepMetrics(None, None, None, None, None, steady, steady_error)
            diagnostics.append(
                Diagnostic(
                    "zero_steady_state",
                    Severity.INFO,
                    "Resposta temporal",
                    "O valor final é zero; subida, acomodação e overshoot relativos não são aplicáveis.",
                )
            )
        else:
            try:
                info = ct.step_info(
                    context.closed_loop,
                    T=time,
                    yfinal=steady,
                    SettlingTimeThreshold=options.settling_threshold,
                    RiseTimeLimits=(options.rise_lower, options.rise_upper),
                )
                metrics = StepMetrics(
                    rise_time_s=_finite_or_none(info.get("RiseTime")),
                    settling_time_s=_finite_or_none(info.get("SettlingTime")),
                    overshoot_percent=_finite_or_none(info.get("Overshoot")),
                    peak=_finite_or_none(info.get("Peak")),
                    peak_time_s=_finite_or_none(info.get("PeakTime")),
                    steady_state_value=steady,
                    steady_state_error=steady_error,
                )
            except Exception as exc:
                diagnostics.append(
                    Diagnostic(
                        "step_metrics_failed",
                        Severity.WARNING,
                        "Resposta temporal",
                        f"A curva foi calculada, mas as métricas não puderam ser extraídas: {exc}",
                    )
                )
                metrics = StepMetrics(None, None, None, None, None, steady, steady_error)
        lower = steady - options.settling_threshold * abs(steady)
        upper = steady + options.settling_threshold * abs(steady)
    elif context.closed_stability is not StabilityStatus.STABLE:
        diagnostics.append(
            Diagnostic(
                "step_metrics_not_applicable",
                Severity.WARNING,
                "Resposta temporal",
                "Métricas de desempenho não são aplicáveis a uma malha não assintoticamente estável.",
            )
        )

    result = StepResult(
        available=available,
        time_s=tuple(float(value) for value in time),
        output=tuple(float(value) for value in output),
        stability=context.closed_stability,
        metrics=metrics,
        settling_lower=lower,
        settling_upper=upper,
    )
    return result, tuple(diagnostics)
