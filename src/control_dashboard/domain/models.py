from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class StabilityStatus(StrEnum):
    STABLE = "estável"
    UNSTABLE = "instável"
    MARGINAL = "limítrofe"
    INCONCLUSIVE = "inconclusiva"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    severity: Severity
    analysis: str
    message: str


@dataclass(frozen=True, slots=True)
class TransferFunctionSpec:
    numerator: tuple[float, ...]
    denominator: tuple[float, ...]
    name: str = "G"


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    frequency_unit: Literal["Hz", "rad/s"] = "Hz"
    frequency_min_rad_s: float | None = None
    frequency_max_rad_s: float | None = None
    frequency_points: int = 1500
    nyquist_points: int = 4000
    settling_threshold: float = 0.02
    rise_lower: float = 0.10
    rise_upper: float = 0.90
    time_end_s: float | None = None
    time_points: int | None = None
    root_gain_selected: float = 1.0
    root_gain_max: float | None = None


@dataclass(frozen=True, slots=True)
class PoleRecord:
    real: float
    imag: float
    magnitude: float
    damping_ratio: float | None
    multiplicity: int = 1

    @property
    def value(self) -> complex:
        return complex(self.real, self.imag)


@dataclass(frozen=True, slots=True)
class PoleZeroResult:
    open_poles: tuple[PoleRecord, ...]
    open_zeros: tuple[PoleRecord, ...]
    closed_poles: tuple[PoleRecord, ...]
    open_rhp_poles: int
    open_axis_poles: int
    closed_rhp_poles: int
    closed_axis_poles: int


@dataclass(frozen=True, slots=True)
class GainMarginEntry:
    gain: float
    gain_db: float
    omega_rad_s: float


@dataclass(frozen=True, slots=True)
class PhaseMarginEntry:
    phase_deg: float
    omega_rad_s: float


@dataclass(frozen=True, slots=True)
class FrequencyResult:
    available: bool
    omega_rad_s: tuple[float, ...]
    magnitude_db: tuple[float, ...]
    phase_deg: tuple[float, ...]
    gain_margin: float | None
    gain_margin_db: float | None
    phase_margin_deg: float | None
    phase_crossover_rad_s: float | None
    gain_crossover_rad_s: float | None
    gain_margins: tuple[GainMarginEntry, ...]
    phase_margins: tuple[PhaseMarginEntry, ...]


@dataclass(frozen=True, slots=True)
class NyquistResult:
    available: bool
    contour_frequency_rad_s: tuple[float, ...]
    primary: tuple[complex, ...]
    mirror: tuple[complex, ...]
    encirclements: int
    open_rhp_poles: int
    predicted_closed_rhp_poles: int
    actual_closed_rhp_poles: int
    minimum_distance_to_minus_one: float | None
    criterion_consistent: bool
    conclusion: StabilityStatus


@dataclass(frozen=True, slots=True)
class RootLocusResult:
    applicable: bool
    gains: tuple[float, ...]
    loci: tuple[tuple[complex, ...], ...]
    selected_gain: float
    selected_poles: tuple[complex, ...]
    unity_poles: tuple[complex, ...]
    unity_consistent: bool


@dataclass(frozen=True, slots=True)
class StepMetrics:
    rise_time_s: float | None
    settling_time_s: float | None
    overshoot_percent: float | None
    peak: float | None
    peak_time_s: float | None
    steady_state_value: float | None
    steady_state_error: float | None


@dataclass(frozen=True, slots=True)
class StepResult:
    available: bool
    time_s: tuple[float, ...]
    output: tuple[float, ...]
    stability: StabilityStatus
    metrics: StepMetrics
    settling_lower: float | None
    settling_upper: float | None


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    original_spec: TransferFunctionSpec
    normalized_spec: TransferFunctionSpec
    options: AnalysisOptions
    open_loop_stability: StabilityStatus
    closed_loop_stability: StabilityStatus
    pole_zero: PoleZeroResult
    frequency: FrequencyResult
    nyquist: NyquistResult
    root_locus: RootLocusResult
    step: StepResult
    diagnostics: tuple[Diagnostic, ...]
    versions: tuple[tuple[str, str], ...]
