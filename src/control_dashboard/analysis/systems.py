from __future__ import annotations

from dataclasses import dataclass

import control as ct
import numpy as np

from control_dashboard.domain.diagnostics import AnalysisComputationError
from control_dashboard.domain.models import Diagnostic, Severity, StabilityStatus, TransferFunctionSpec


@dataclass(slots=True)
class SystemContext:
    open_loop: ct.TransferFunction
    closed_loop: ct.TransferFunction
    sensitivity: ct.TransferFunction
    open_poles: np.ndarray
    open_zeros: np.ndarray
    closed_poles: np.ndarray
    tolerance: float
    open_stability: StabilityStatus
    closed_stability: StabilityStatus


def root_tolerance(roots: np.ndarray | tuple[complex, ...]) -> float:
    array = np.asarray(roots, dtype=complex)
    scale = float(np.max(np.abs(array))) if array.size else 1.0
    return 1e-7 * max(1.0, scale)


def classify_stability(roots: np.ndarray | tuple[complex, ...]) -> StabilityStatus:
    array = np.asarray(roots, dtype=complex)
    if not array.size:
        return StabilityStatus.STABLE
    tolerance = root_tolerance(array)
    if np.any(array.real > tolerance):
        return StabilityStatus.UNSTABLE
    if np.any(np.abs(array.real) <= tolerance):
        return StabilityStatus.MARGINAL
    return StabilityStatus.STABLE


def _closed_denominator(spec: TransferFunctionSpec, gain: float = 1.0) -> np.ndarray:
    numerator = np.asarray(spec.numerator, dtype=float)
    denominator = np.asarray(spec.denominator, dtype=float)
    size = max(numerator.size, denominator.size)
    return np.pad(denominator, (size - denominator.size, 0)) + gain * np.pad(numerator, (size - numerator.size, 0))


def closed_loop_roots(spec: TransferFunctionSpec, gain: float = 1.0) -> np.ndarray:
    characteristic = np.trim_zeros(_closed_denominator(spec, gain), trim="f")
    if not characteristic.size:
        return np.asarray([], dtype=complex)
    if characteristic.size == 1:
        return np.asarray([], dtype=complex)
    return np.roots(characteristic)


def build_system_context(spec: TransferFunctionSpec) -> SystemContext:
    characteristic = np.trim_zeros(_closed_denominator(spec), trim="f")
    if not characteristic.size:
        raise AnalysisComputationError(
            [
                Diagnostic(
                    "singular_closed_loop",
                    Severity.ERROR,
                    "sistema",
                    "A malha fechada é indefinida porque D(s) + N(s) é identicamente nulo.",
                )
            ]
        )
    try:
        open_loop = ct.tf(list(spec.numerator), list(spec.denominator))
        closed_loop = ct.feedback(open_loop, 1.0)
        sensitivity = ct.feedback(1.0, open_loop)
        open_poles = np.asarray(ct.poles(open_loop), dtype=complex)
        open_zeros = np.asarray(ct.zeros(open_loop), dtype=complex)
        closed_poles = np.asarray(ct.poles(closed_loop), dtype=complex)
    except Exception as exc:
        raise AnalysisComputationError(
            [
                Diagnostic(
                    "system_construction",
                    Severity.ERROR,
                    "sistema",
                    f"Não foi possível construir a malha: {exc}",
                )
            ]
        ) from exc

    tolerance = root_tolerance(np.concatenate((open_poles, closed_poles)))
    return SystemContext(
        open_loop=open_loop,
        closed_loop=closed_loop,
        sensitivity=sensitivity,
        open_poles=open_poles,
        open_zeros=open_zeros,
        closed_poles=closed_poles,
        tolerance=tolerance,
        open_stability=classify_stability(open_poles),
        closed_stability=classify_stability(closed_poles),
    )
