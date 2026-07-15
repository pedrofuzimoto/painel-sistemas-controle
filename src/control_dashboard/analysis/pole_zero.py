from __future__ import annotations

import numpy as np

from control_dashboard.analysis.systems import SystemContext
from control_dashboard.domain.models import PoleRecord, PoleZeroResult


def _records(roots: np.ndarray, tolerance: float) -> tuple[PoleRecord, ...]:
    remaining = [complex(value) for value in np.sort_complex(roots)]
    records: list[PoleRecord] = []
    while remaining:
        root = remaining.pop(0)
        group = [root]
        unmatched: list[complex] = []
        for candidate in remaining:
            distance = abs(candidate - root) / max(1.0, abs(candidate), abs(root))
            if distance <= max(tolerance, 1e-6):
                group.append(candidate)
            else:
                unmatched.append(candidate)
        remaining = unmatched
        representative = sum(group) / len(group)
        magnitude = abs(representative)
        damping = -representative.real / magnitude if magnitude > tolerance else None
        records.append(
            PoleRecord(
                real=float(representative.real),
                imag=float(representative.imag),
                magnitude=float(magnitude),
                damping_ratio=float(damping) if damping is not None else None,
                multiplicity=len(group),
            )
        )
    return tuple(records)


def analyze_pole_zero(context: SystemContext) -> PoleZeroResult:
    tolerance = context.tolerance
    return PoleZeroResult(
        open_poles=_records(context.open_poles, tolerance),
        open_zeros=_records(context.open_zeros, tolerance),
        closed_poles=_records(context.closed_poles, tolerance),
        open_rhp_poles=int(np.count_nonzero(context.open_poles.real > tolerance)),
        open_axis_poles=int(np.count_nonzero(np.abs(context.open_poles.real) <= tolerance)),
        closed_rhp_poles=int(np.count_nonzero(context.closed_poles.real > tolerance)),
        closed_axis_poles=int(np.count_nonzero(np.abs(context.closed_poles.real) <= tolerance)),
    )
