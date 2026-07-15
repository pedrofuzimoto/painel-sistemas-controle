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
    GainMarginEntry,
    PhaseMarginEntry,
    Severity,
)


def _number_or_none(value: object) -> float | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    result = float(array.reshape(-1)[0])
    return None if math.isnan(result) else result


def _automatic_limits(context: SystemContext, crossover_frequencies: list[float]) -> tuple[float, float]:
    roots = np.concatenate((context.open_poles, context.open_zeros))
    breaks = [float(abs(root)) for root in roots if abs(root) > context.tolerance]
    if breaks:
        omega_min = 10.0 ** (math.floor(math.log10(min(breaks))) - 2)
        omega_max = 10.0 ** (math.ceil(math.log10(max(breaks))) + 2)
    else:
        omega_min, omega_max = 1e-2, 1e2
    finite_crossings = [value for value in crossover_frequencies if math.isfinite(value) and value > 0]
    if finite_crossings:
        omega_min = min(omega_min, min(finite_crossings) / 10.0)
        omega_max = max(omega_max, max(finite_crossings) * 10.0)
    omega_min = max(omega_min, 1e-12)
    omega_max = max(omega_max, omega_min * 100.0)
    return omega_min, omega_max


def analyze_frequency(
    context: SystemContext,
    options: AnalysisOptions,
) -> tuple[FrequencyResult, tuple[Diagnostic, ...]]:
    diagnostics: list[Diagnostic] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            gm, pm, w_pc, w_gc = ct.margin(context.open_loop)
            gm_all, pm_all, _, w_pc_all, w_gc_all, _ = ct.stability_margins(context.open_loop, returnall=True)
        except Exception as exc:
            gm = pm = w_pc = w_gc = math.nan
            gm_all = pm_all = w_pc_all = w_gc_all = np.asarray([])
            diagnostics.append(
                Diagnostic(
                    "margin_unavailable",
                    Severity.WARNING,
                    "Bode",
                    f"As margens não puderam ser determinadas: {exc}",
                )
            )

    for warning in caught:
        diagnostics.append(Diagnostic("frequency_warning", Severity.WARNING, "Bode", str(warning.message)))

    crossover_values = [
        float(value)
        for values in (np.atleast_1d(w_pc_all), np.atleast_1d(w_gc_all))
        for value in values
        if np.isfinite(value) and value > 0
    ]
    if options.frequency_min_rad_s is None:
        omega_min, omega_max = _automatic_limits(context, crossover_values)
    else:
        omega_min = float(options.frequency_min_rad_s)
        omega_max = float(options.frequency_max_rad_s)

    omega = np.logspace(math.log10(omega_min), math.log10(omega_max), options.frequency_points)
    try:
        response = ct.frequency_response(context.open_loop, omega)
        magnitude = np.squeeze(np.asarray(response.magnitude, dtype=float))
        phase = np.squeeze(np.asarray(response.phase, dtype=float))
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(magnitude), np.finfo(float).tiny))
        phase_deg = np.rad2deg(np.unwrap(phase))
        available = True
    except Exception as exc:
        diagnostics.append(
            Diagnostic("bode_failed", Severity.ERROR, "Bode", f"Falha ao calcular a resposta em frequência: {exc}")
        )
        omega = magnitude_db = phase_deg = np.asarray([], dtype=float)
        available = False

    gain_entries: list[GainMarginEntry] = []
    for gain, frequency in zip(np.atleast_1d(gm_all), np.atleast_1d(w_pc_all), strict=False):
        gain_float = float(gain)
        frequency_float = float(frequency)
        if math.isnan(gain_float) or math.isnan(frequency_float):
            continue
        gain_db = math.inf if math.isinf(gain_float) else 20.0 * math.log10(gain_float)
        gain_entries.append(GainMarginEntry(gain_float, gain_db, frequency_float))

    phase_entries = tuple(
        PhaseMarginEntry(float(margin), float(frequency))
        for margin, frequency in zip(np.atleast_1d(pm_all), np.atleast_1d(w_gc_all), strict=False)
        if not math.isnan(float(margin)) and not math.isnan(float(frequency))
    )
    gm_value = _number_or_none(gm)
    gm_db = None
    if gm_value is not None:
        gm_db = math.inf if math.isinf(gm_value) else 20.0 * math.log10(gm_value)

    result = FrequencyResult(
        available=available,
        omega_rad_s=tuple(float(value) for value in omega),
        magnitude_db=tuple(float(value) for value in magnitude_db),
        phase_deg=tuple(float(value) for value in phase_deg),
        gain_margin=gm_value,
        gain_margin_db=gm_db,
        phase_margin_deg=_number_or_none(pm),
        phase_crossover_rad_s=_number_or_none(w_pc),
        gain_crossover_rad_s=_number_or_none(w_gc),
        gain_margins=tuple(gain_entries),
        phase_margins=phase_entries,
    )
    return result, tuple(diagnostics)
