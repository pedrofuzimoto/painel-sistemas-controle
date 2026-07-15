from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from control_dashboard.domain.models import AnalysisBundle
from control_dashboard.visualization.theme import BLUE, DARK, GREEN, ORANGE, RED, convert_frequency, frequency_values


def _new_axes(title: str, figsize: tuple[float, float] = (7.2, 4.8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return fig, ax


def _pole_zero(bundle: AnalysisBundle):
    fig, ax = _new_axes("Mapa de polos e zeros")
    poles = bundle.pole_zero.open_poles
    zeros = bundle.pole_zero.open_zeros
    if poles:
        ax.scatter(
            [p.real for p in poles],
            [p.imag for p in poles],
            marker="x",
            s=70,
            linewidth=2,
            color=RED,
            label="Polos de G",
        )
    if zeros:
        ax.scatter(
            [z.real for z in zeros],
            [z.imag for z in zeros],
            marker="o",
            s=65,
            facecolors="none",
            edgecolors=BLUE,
            linewidth=2,
            label="Zeros de G",
        )
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.axvspan(ax.get_xlim()[0], 0, color=GREEN, alpha=0.04)
    ax.set_xlabel("Parte real (rad/s)")
    ax.set_ylabel("Parte imaginária (rad/s)")
    ax.set_aspect("equal", adjustable="datalim")
    if poles or zeros:
        ax.legend()
    fig.tight_layout()
    return fig


def _bode(bundle: AnalysisBundle):
    result = bundle.frequency
    unit = bundle.options.frequency_unit
    x = frequency_values(result.omega_rad_s, unit)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.3, 6.0), sharex=True)
    ax1.semilogx(x, result.magnitude_db, color=BLUE)
    ax2.semilogx(x, result.phase_deg, color=ORANGE)
    ax1.axhline(0, color=DARK, linewidth=0.8, linestyle=":")
    ax2.axhline(-180, color=DARK, linewidth=0.8, linestyle=":")
    for omega, color in ((result.phase_crossover_rad_s, RED), (result.gain_crossover_rad_s, GREEN)):
        value = convert_frequency(omega, unit)
        if value is not None and math.isfinite(value) and value > 0:
            ax1.axvline(value, color=color, linewidth=1, linestyle="--")
            ax2.axvline(value, color=color, linewidth=1, linestyle="--")
    ax1.set_title("Diagrama de Bode")
    ax1.set_ylabel("Magnitude (dB)")
    ax2.set_ylabel("Fase (graus)")
    ax2.set_xlabel("Frequência (Hz)" if unit == "Hz" else "Frequência angular (rad/s)")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def _nyquist(bundle: AnalysisBundle):
    fig, ax = _new_axes("Diagrama de Nyquist", (6.5, 5.7))
    primary = np.asarray(bundle.nyquist.primary, dtype=complex)
    mirror = np.asarray(bundle.nyquist.mirror, dtype=complex)
    if primary.size:
        ax.plot(primary.real, primary.imag, color=BLUE, label="ω ≥ 0")
        ax.plot(mirror.real, mirror.imag, color=BLUE, linestyle="--", label="ω ≤ 0")
    ax.scatter([-1], [0], marker="x", s=80, linewidth=2, color=RED, label="−1+j0")
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_xlim(-3, 1.5)
    ax.set_ylim(-2.25, 2.25)
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    return fig


def _root_locus(bundle: AnalysisBundle):
    fig, ax = _new_axes("Lugar das raízes", (6.8, 5.5))
    result = bundle.root_locus
    if result.applicable and result.loci:
        loci = np.asarray(result.loci, dtype=complex)
        for branch in range(loci.shape[1]):
            ax.plot(loci[:, branch].real, loci[:, branch].imag, color=BLUE)
    else:
        ax.text(0.5, 0.5, "Não aplicável ao ganho estático", ha="center", va="center", transform=ax.transAxes)
    poles = bundle.pole_zero.open_poles
    zeros = bundle.pole_zero.open_zeros
    if poles:
        ax.scatter([p.real for p in poles], [p.imag for p in poles], marker="x", s=70, color=RED, label="Polos de G")
    if zeros:
        ax.scatter(
            [z.real for z in zeros],
            [z.imag for z in zeros],
            marker="o",
            s=65,
            facecolors="none",
            edgecolors=BLUE,
            label="Zeros de G",
        )
    unity = np.asarray(result.unity_poles, dtype=complex)
    if unity.size:
        ax.scatter(unity.real, unity.imag, marker="D", s=45, color=GREEN, label="K=1")
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_xlabel("Parte real (rad/s)")
    ax.set_ylabel("Parte imaginária (rad/s)")
    ax.set_aspect("equal", adjustable="datalim")
    if poles or zeros or unity.size:
        ax.legend()
    fig.tight_layout()
    return fig


def _step(bundle: AnalysisBundle):
    fig, ax = _new_axes("Resposta ao degrau — malha fechada")
    time = np.asarray(bundle.step.time_s, dtype=float)
    output = np.asarray(bundle.step.output, dtype=float)
    ax.plot(time, np.ones_like(time), color=DARK, linestyle="--", label="Degrau unitário")
    ax.plot(time, output, color=BLUE, label="Saída y(t)")
    metrics = bundle.step.metrics
    if metrics.steady_state_value is not None:
        ax.axhline(metrics.steady_state_value, color=GREEN, linestyle=":", label="Valor final")
    if bundle.step.settling_lower is not None and bundle.step.settling_upper is not None:
        ax.axhspan(bundle.step.settling_lower, bundle.step.settling_upper, color=GREEN, alpha=0.08, label="Banda")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    return fig


def make_static_figures(bundle: AnalysisBundle) -> dict[str, plt.Figure]:
    return {
        "polos_zeros": _pole_zero(bundle),
        "bode": _bode(bundle),
        "nyquist": _nyquist(bundle),
        "lugar_raizes": _root_locus(bundle),
        "resposta_degrau": _step(bundle),
    }
