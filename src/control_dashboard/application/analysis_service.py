from __future__ import annotations

import importlib.metadata

from control_dashboard.analysis.frequency import analyze_frequency
from control_dashboard.analysis.nyquist import analyze_nyquist
from control_dashboard.analysis.pole_zero import analyze_pole_zero
from control_dashboard.analysis.root_locus import analyze_root_locus
from control_dashboard.analysis.systems import build_system_context
from control_dashboard.analysis.time_response import analyze_step
from control_dashboard.domain.models import AnalysisBundle, AnalysisOptions, TransferFunctionSpec
from control_dashboard.domain.validation import validate_options, validate_spec


def _versions() -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for package in ("control", "numpy", "scipy", "matplotlib", "plotly", "streamlit", "reportlab"):
        try:
            values.append((package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            values.append((package, "não instalado"))
    return tuple(values)


def analyze(
    spec: TransferFunctionSpec,
    options: AnalysisOptions | None = None,
) -> AnalysisBundle:
    """Executa uma análise completa sem depender da interface ou dos renderizadores."""

    selected_options = options or AnalysisOptions()
    validate_options(selected_options)
    validation = validate_spec(spec)
    normalized = validation.normalized_spec
    context = build_system_context(normalized)

    pole_zero = analyze_pole_zero(context)
    frequency, frequency_diagnostics = analyze_frequency(context, selected_options)
    nyquist, nyquist_diagnostics = analyze_nyquist(context, frequency, selected_options)
    root_locus, root_diagnostics = analyze_root_locus(context, normalized, selected_options)
    step, step_diagnostics = analyze_step(context, selected_options)

    diagnostics = (
        validation.diagnostics + frequency_diagnostics + nyquist_diagnostics + root_diagnostics + step_diagnostics
    )
    return AnalysisBundle(
        original_spec=spec,
        normalized_spec=normalized,
        options=selected_options,
        open_loop_stability=context.open_stability,
        closed_loop_stability=context.closed_stability,
        pole_zero=pole_zero,
        frequency=frequency,
        nyquist=nyquist,
        root_locus=root_locus,
        step=step,
        diagnostics=diagnostics,
        versions=_versions(),
    )
