from __future__ import annotations

import warnings

import control as ct
import numpy as np

from control_dashboard.analysis.systems import SystemContext, closed_loop_roots
from control_dashboard.domain.models import (
    AnalysisOptions,
    Diagnostic,
    RootLocusResult,
    Severity,
    TransferFunctionSpec,
)


def _same_roots(first: np.ndarray, second: np.ndarray) -> bool:
    if first.size != second.size:
        return False
    return bool(
        np.allclose(
            np.sort_complex(first),
            np.sort_complex(second),
            rtol=1e-5,
            atol=1e-7,
        )
    )


def analyze_root_locus(
    context: SystemContext,
    spec: TransferFunctionSpec,
    options: AnalysisOptions,
) -> tuple[RootLocusResult, tuple[Diagnostic, ...]]:
    diagnostics: list[Diagnostic] = []
    selected = float(options.root_gain_selected)
    selected_poles = closed_loop_roots(spec, selected)
    unity_poles = closed_loop_roots(spec, 1.0)
    unity_consistent = _same_roots(unity_poles, context.closed_poles)

    if not context.open_poles.size:
        return (
            RootLocusResult(False, (), (), selected, tuple(selected_poles), tuple(unity_poles), unity_consistent),
            (
                Diagnostic(
                    "root_locus_static",
                    Severity.INFO,
                    "Lugar das raízes",
                    "O ganho estático não possui polos finitos para formar um lugar das raízes.",
                ),
            ),
        )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            automatic = ct.root_locus_map(context.open_loop)
            automatic_gains = np.asarray(automatic.gains, dtype=float)
            if options.root_gain_max is not None:
                maximum = max(float(options.root_gain_max), selected, 1.0)
                positive = np.geomspace(max(maximum * 1e-6, 1e-12), maximum, 400)
                linear = np.linspace(0.0, maximum, 400)
                gains = np.unique(np.concatenate((linear, positive, [0.0, 1.0, selected])))
            else:
                gains = np.unique(np.concatenate((automatic_gains, [0.0, 1.0, selected])))
            data = ct.root_locus_map(context.open_loop, gains=gains)
        for warning in caught:
            diagnostics.append(
                Diagnostic("root_locus_warning", Severity.WARNING, "Lugar das raízes", str(warning.message))
            )
        loci_array = np.asarray(data.loci, dtype=complex)
        loci = tuple(tuple(complex(value) for value in row) for row in loci_array)
        applicable = True
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                "root_locus_failed",
                Severity.ERROR,
                "Lugar das raízes",
                f"Falha ao calcular o lugar das raízes: {exc}",
            )
        )
        gains = np.asarray([], dtype=float)
        loci = ()
        applicable = False

    if not unity_consistent:
        diagnostics.append(
            Diagnostic(
                "root_locus_unity_mismatch",
                Severity.ERROR,
                "Lugar das raízes",
                "Os polos para K=1 não coincidem com os polos da malha fechada.",
            )
        )

    return (
        RootLocusResult(
            applicable=applicable,
            gains=tuple(float(value) for value in gains),
            loci=loci,
            selected_gain=selected,
            selected_poles=tuple(complex(value) for value in selected_poles),
            unity_poles=tuple(complex(value) for value in unity_poles),
            unity_consistent=unity_consistent,
        ),
        tuple(diagnostics),
    )
