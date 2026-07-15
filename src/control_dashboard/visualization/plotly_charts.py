from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from control_dashboard.domain.models import AnalysisBundle, PoleRecord
from control_dashboard.visualization.theme import (
    BLUE,
    DARK,
    GREEN,
    GRID,
    ORANGE,
    RED,
    STABLE_FILL,
    convert_frequency,
    frequency_label,
    frequency_values,
)


def _base_layout(title: str, height: int = 520) -> dict[str, object]:
    return {
        "title": {"text": title, "x": 0.02},
        "height": height,
        "margin": {"l": 62, "r": 28, "t": 62, "b": 55},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "color": DARK},
        "legend": {"orientation": "h", "y": 1.03, "x": 1, "xanchor": "right"},
    }


def _record_arrays(records: tuple[PoleRecord, ...]) -> tuple[list[float], list[float], list[str]]:
    x = [record.real for record in records]
    y = [record.imag for record in records]
    hover = [
        f"{record.real:.6g} {record.imag:+.6g}j<br>ωn={record.magnitude:.6g} rad/s"
        + (f"<br>ζ={record.damping_ratio:.4g}" if record.damping_ratio is not None else "")
        + (f"<br>multiplicidade={record.multiplicity}" if record.multiplicity > 1 else "")
        for record in records
    ]
    return x, y, hover


def pole_zero_figure(bundle: AnalysisBundle, show_closed: bool = False) -> go.Figure:
    result = bundle.pole_zero
    fig = go.Figure()
    poles_x, poles_y, poles_hover = _record_arrays(result.open_poles)
    zeros_x, zeros_y, zeros_hover = _record_arrays(result.open_zeros)
    fig.add_trace(
        go.Scatter(
            x=poles_x,
            y=poles_y,
            mode="markers",
            marker={"symbol": "x", "size": 13, "color": RED, "line": {"width": 2}},
            text=poles_hover,
            hovertemplate="%{text}<extra>Polos de G</extra>",
            name="Polos de G",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=zeros_x,
            y=zeros_y,
            mode="markers",
            marker={"symbol": "circle-open", "size": 13, "color": BLUE, "line": {"width": 2}},
            text=zeros_hover,
            hovertemplate="%{text}<extra>Zeros de G</extra>",
            name="Zeros de G",
        )
    )
    if show_closed:
        closed_x, closed_y, closed_hover = _record_arrays(result.closed_poles)
        fig.add_trace(
            go.Scatter(
                x=closed_x,
                y=closed_y,
                mode="markers",
                marker={"symbol": "diamond-open", "size": 12, "color": GREEN, "line": {"width": 2}},
                text=closed_hover,
                hovertemplate="%{text}<extra>Polos de T</extra>",
                name="Polos de T (K=1)",
            )
        )
    fig.add_vrect(x0=-1e300, x1=0, fillcolor=STABLE_FILL, line_width=0, layer="below")
    fig.add_hline(y=0, line={"color": DARK, "width": 1})
    fig.add_vline(x=0, line={"color": DARK, "width": 1})
    fig.update_layout(**_base_layout("Mapa de polos e zeros"))
    fig.update_xaxes(title="Parte real (rad/s)", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Parte imaginária (rad/s)", gridcolor=GRID, scaleanchor="x", scaleratio=1, zeroline=False)
    return fig


def bode_figure(bundle: AnalysisBundle) -> go.Figure:
    result = bundle.frequency
    unit = bundle.options.frequency_unit
    x = frequency_values(result.omega_rad_s, unit)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=x, y=result.magnitude_db, mode="lines", line={"color": BLUE, "width": 2}, name="Magnitude"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=result.phase_deg, mode="lines", line={"color": ORANGE, "width": 2}, name="Fase"),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line={"color": DARK, "width": 1, "dash": "dot"}, row=1, col=1)
    fig.add_hline(y=-180, line={"color": DARK, "width": 1, "dash": "dot"}, row=2, col=1)
    for omega, color in (
        (result.phase_crossover_rad_s, RED),
        (result.gain_crossover_rad_s, GREEN),
    ):
        value = convert_frequency(omega, unit)
        if value is not None and math.isfinite(value) and value > 0:
            fig.add_vline(x=value, line={"color": color, "width": 1.2, "dash": "dash"}, row="all", col=1)
    fig.update_layout(**_base_layout("Diagrama de Bode", height=650), hovermode="x unified")
    fig.update_xaxes(type="log", title=frequency_label(unit), gridcolor=GRID, row=2, col=1)
    fig.update_xaxes(type="log", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title="Magnitude (dB)", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title="Fase (graus)", gridcolor=GRID, row=2, col=1)
    return fig


def nyquist_figure(bundle: AnalysisBundle, focus_critical: bool = True) -> go.Figure:
    result = bundle.nyquist
    unit = bundle.options.frequency_unit
    frequencies = frequency_values(result.contour_frequency_rad_s, unit)
    primary = np.asarray(result.primary, dtype=complex)
    mirror = np.asarray(result.mirror, dtype=complex)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=primary.real,
            y=primary.imag,
            mode="lines",
            line={"color": BLUE, "width": 2},
            customdata=frequencies,
            hovertemplate="Re=%{x:.5g}<br>Im=%{y:.5g}<br>f=%{customdata:.5g}<extra>ω≥0</extra>",
            name="Frequências positivas",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mirror.real,
            y=mirror.imag,
            mode="lines",
            line={"color": BLUE, "width": 1.5, "dash": "dash"},
            customdata=frequencies,
            hovertemplate="Re=%{x:.5g}<br>Im=%{y:.5g}<br>f=%{customdata:.5g}<extra>ω≤0</extra>",
            name="Frequências negativas",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-1],
            y=[0],
            mode="markers+text",
            marker={"symbol": "x", "size": 15, "color": RED, "line": {"width": 2}},
            text=["−1+j0"],
            textposition="top left",
            name="Ponto crítico",
        )
    )
    fig.add_hline(y=0, line={"color": DARK, "width": 1})
    fig.add_vline(x=0, line={"color": DARK, "width": 1})
    fig.update_layout(**_base_layout("Diagrama de Nyquist", height=600))
    fig.update_xaxes(title="Parte real", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Parte imaginária", gridcolor=GRID, scaleanchor="x", scaleratio=1, zeroline=False)
    if focus_critical:
        fig.update_xaxes(range=[-3.0, 1.5])
        fig.update_yaxes(range=[-2.25, 2.25])
    return fig


def _damping_grid(fig: go.Figure, radius: float) -> None:
    for damping in (0.2, 0.4, 0.6, 0.8):
        imaginary = radius * math.sqrt(max(0.0, 1.0 - damping**2))
        real = -radius * damping
        for sign in (-1.0, 1.0):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=real,
                y1=sign * imaginary,
                line={"color": "rgba(80,80,80,0.22)", "width": 1, "dash": "dot"},
                layer="below",
            )


def root_locus_figure(bundle: AnalysisBundle) -> go.Figure:
    result = bundle.root_locus
    fig = go.Figure()
    if result.applicable and result.loci:
        loci = np.asarray(result.loci, dtype=complex)
        gains = np.asarray(result.gains, dtype=float)
        for branch in range(loci.shape[1]):
            values = loci[:, branch]
            fig.add_trace(
                go.Scatter(
                    x=values.real,
                    y=values.imag,
                    mode="lines",
                    line={"color": BLUE, "width": 2},
                    customdata=gains,
                    hovertemplate="K=%{customdata:.6g}<br>p=%{x:.6g}%{y:+.6g}j<extra></extra>",
                    showlegend=branch == 0,
                    name="Lugar das raízes",
                )
            )
        radius = max(float(np.nanpercentile(np.abs(loci), 90)), 1.0)
        _damping_grid(fig, radius)
    poles_x, poles_y, _ = _record_arrays(bundle.pole_zero.open_poles)
    zeros_x, zeros_y, _ = _record_arrays(bundle.pole_zero.open_zeros)
    fig.add_trace(
        go.Scatter(
            x=poles_x, y=poles_y, mode="markers", marker={"symbol": "x", "size": 13, "color": RED}, name="Polos de G"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=zeros_x,
            y=zeros_y,
            mode="markers",
            marker={"symbol": "circle-open", "size": 13, "color": BLUE},
            name="Zeros de G",
        )
    )
    unity = np.asarray(result.unity_poles, dtype=complex)
    fig.add_trace(
        go.Scatter(
            x=unity.real,
            y=unity.imag,
            mode="markers",
            marker={"symbol": "diamond", "size": 10, "color": GREEN},
            name="K=1",
        )
    )
    if not math.isclose(result.selected_gain, 1.0):
        selected = np.asarray(result.selected_poles, dtype=complex)
        fig.add_trace(
            go.Scatter(
                x=selected.real,
                y=selected.imag,
                mode="markers",
                marker={"symbol": "star", "size": 12, "color": ORANGE},
                name=f"K={result.selected_gain:g}",
            )
        )
    fig.add_hline(y=0, line={"color": DARK, "width": 1})
    fig.add_vline(x=0, line={"color": DARK, "width": 1})
    fig.update_layout(**_base_layout("Lugar das raízes", height=600))
    fig.update_xaxes(title="Parte real (rad/s)", gridcolor=GRID)
    fig.update_yaxes(title="Parte imaginária (rad/s)", gridcolor=GRID, scaleanchor="x", scaleratio=1)
    return fig


def step_figure(bundle: AnalysisBundle) -> go.Figure:
    result = bundle.step
    time = np.asarray(result.time_s, dtype=float)
    output = np.asarray(result.output, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time, y=np.ones_like(time), mode="lines", line={"color": DARK, "dash": "dash"}, name="Degrau unitário"
        )
    )
    fig.add_trace(go.Scatter(x=time, y=output, mode="lines", line={"color": BLUE, "width": 2.2}, name="Saída y(t)"))
    metrics = result.metrics
    if metrics.steady_state_value is not None:
        fig.add_hline(
            y=metrics.steady_state_value,
            line={"color": GREEN, "width": 1.3, "dash": "dot"},
            annotation_text=f"y∞={metrics.steady_state_value:.5g}",
        )
    if result.settling_lower is not None and result.settling_upper is not None:
        fig.add_hrect(
            y0=result.settling_lower,
            y1=result.settling_upper,
            fillcolor="rgba(43,138,62,0.09)",
            line_width=0,
            annotation_text="banda de acomodação",
        )
    if metrics.peak_time_s is not None and metrics.peak is not None and output.size:
        index = int(np.argmin(np.abs(time - metrics.peak_time_s)))
        fig.add_trace(
            go.Scatter(
                x=[time[index]],
                y=[output[index]],
                mode="markers",
                marker={"size": 10, "color": RED},
                name="Pico",
            )
        )
    fig.update_layout(**_base_layout("Resposta ao degrau — malha fechada", height=520), hovermode="x unified")
    fig.update_xaxes(title="Tempo (s)", gridcolor=GRID)
    fig.update_yaxes(title="Amplitude", gridcolor=GRID)
    return fig
