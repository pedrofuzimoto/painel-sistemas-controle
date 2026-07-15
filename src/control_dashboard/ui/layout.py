from __future__ import annotations

import json
import math
from pathlib import Path

import streamlit as st

from control_dashboard.application.analysis_service import analyze
from control_dashboard.domain.diagnostics import AnalysisComputationError, AnalysisValidationError
from control_dashboard.domain.models import AnalysisBundle, AnalysisOptions, Diagnostic, Severity, TransferFunctionSpec
from control_dashboard.domain.validation import parse_transfer_function_text
from control_dashboard.reporting import build_export
from control_dashboard.ui.components import (
    format_frequency,
    format_number,
    pole_table,
    selected_pole_table,
    transfer_function_latex,
)
from control_dashboard.visualization.plotly_charts import (
    bode_figure,
    nyquist_figure,
    pole_zero_figure,
    root_locus_figure,
    step_figure,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_FILE = ROOT / "examples" / "reference_systems.json"
LOGO_FILE = ROOT / "assets" / "pedro-control-logo.png"


@st.cache_data(show_spinner=False)
def _load_examples() -> dict[str, dict[str, str]]:
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _cached_analyze(spec: TransferFunctionSpec, options: AnalysisOptions) -> AnalysisBundle:
    return analyze(spec, options)


def _apply_example() -> None:
    examples = _load_examples()
    selected = examples[st.session_state.example_name]
    st.session_state.numerator_input = selected["numerator"]
    st.session_state.denominator_input = selected["denominator"]


def _initialize_state() -> None:
    examples = _load_examples()
    first_name = next(iter(examples))
    defaults = examples[first_name]
    st.session_state.setdefault("example_name", first_name)
    st.session_state.setdefault("numerator_input", defaults["numerator"])
    st.session_state.setdefault("denominator_input", defaults["denominator"])
    st.session_state.setdefault("validation_diagnostics", ())
    st.session_state.setdefault("report_bytes", None)
    if "analysis_bundle" not in st.session_state:
        try:
            spec = parse_transfer_function_text(defaults["numerator"], defaults["denominator"])
            st.session_state.analysis_bundle = _cached_analyze(spec, AnalysisOptions())
        except (AnalysisValidationError, AnalysisComputationError) as exc:
            st.session_state.analysis_bundle = None
            st.session_state.validation_diagnostics = exc.diagnostics


def _sidebar_form() -> None:
    examples = _load_examples()
    st.sidebar.caption("Mascote oficial em malha fechada 😄")
    st.sidebar.header("Modelo de malha aberta")
    st.sidebar.selectbox(
        "Exemplo",
        options=list(examples),
        key="example_name",
        on_change=_apply_example,
        help="Carrega coeficientes de referência; você pode editá-los livremente.",
    )
    st.sidebar.caption(examples[st.session_state.example_name]["description"])

    with st.sidebar.form("transfer_function_form"):
        st.text_input(
            "Numerador N(s)",
            key="numerator_input",
            help="Coeficientes em potências decrescentes, separados por ; ou espaço.",
        )
        st.text_input(
            "Denominador D(s)",
            key="denominator_input",
            help="Exemplo: 1; 3; 2 representa s² + 3s + 2.",
        )
        st.caption("Aceita ponto ou vírgula decimal e notação científica. Use ‘;’ ou espaço entre coeficientes.")

        with st.expander("Opções avançadas"):
            unit = st.selectbox("Unidade de frequência", ["Hz", "rad/s"], index=0)
            manual_frequency = st.checkbox("Faixa de frequência manual", value=False)
            frequency_min = st.number_input(
                "Frequência mínima",
                min_value=1e-12,
                value=0.1,
                format="%.6g",
                disabled=not manual_frequency,
            )
            frequency_max = st.number_input(
                "Frequência máxima",
                min_value=1e-11,
                value=100000.0,
                format="%.6g",
                disabled=not manual_frequency,
            )
            frequency_points = st.number_input("Pontos no Bode", min_value=200, max_value=5000, value=1500, step=100)
            nyquist_points = st.number_input("Pontos no Nyquist", min_value=500, max_value=10000, value=4000, step=500)
            manual_time = st.checkbox("Horizonte temporal manual", value=False)
            time_end = st.number_input(
                "Tempo final (s)",
                min_value=1e-9,
                value=1.0,
                format="%.6g",
                disabled=not manual_time,
            )
            settling_percent = st.slider("Banda de acomodação (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
            selected_gain = st.number_input("Ganho K selecionado", min_value=0.0, value=1.0, format="%.6g")
            manual_gain_max = st.checkbox("Limitar faixa de K", value=False)
            gain_max = st.number_input(
                "K máximo",
                min_value=1e-9,
                value=100.0,
                format="%.6g",
                disabled=not manual_gain_max,
            )

        submitted = st.form_submit_button("Analisar sistema", type="primary", width="stretch")

    if not submitted:
        return
    try:
        spec = parse_transfer_function_text(
            st.session_state.numerator_input,
            st.session_state.denominator_input,
        )
        factor = 2.0 * math.pi if unit == "Hz" else 1.0
        options = AnalysisOptions(
            frequency_unit=unit,
            frequency_min_rad_s=float(frequency_min) * factor if manual_frequency else None,
            frequency_max_rad_s=float(frequency_max) * factor if manual_frequency else None,
            frequency_points=int(frequency_points),
            nyquist_points=int(nyquist_points),
            settling_threshold=float(settling_percent) / 100.0,
            time_end_s=float(time_end) if manual_time else None,
            root_gain_selected=float(selected_gain),
            root_gain_max=float(gain_max) if manual_gain_max else None,
        )
        with st.spinner("Calculando respostas e critérios de estabilidade…"):
            bundle = _cached_analyze(spec, options)
        st.session_state.analysis_bundle = bundle
        st.session_state.validation_diagnostics = ()
        st.session_state.report_bytes = None
    except (AnalysisValidationError, AnalysisComputationError) as exc:
        st.session_state.validation_diagnostics = exc.diagnostics
    except Exception as exc:  # barreira de segurança da interface
        st.session_state.validation_diagnostics = (
            Diagnostic("unexpected_ui_error", Severity.ERROR, "interface", f"Falha inesperada: {exc}"),
        )


def _show_submission_errors() -> None:
    diagnostics = st.session_state.validation_diagnostics
    if not diagnostics:
        return
    st.error("A nova entrada não foi aplicada. O último resultado válido permanece no painel.")
    for diagnostic in diagnostics:
        st.error(diagnostic.message)


def _status_banner(bundle: AnalysisBundle) -> None:
    status = bundle.closed_loop_stability
    message = f"Malha fechada: {status.value}"
    if status.value == "estável":
        st.success("✓ " + message)
    elif status.value == "instável":
        st.error("✕ " + message)
    else:
        st.warning("⚠ " + message)


def _summary_metrics(bundle: AnalysisBundle) -> None:
    frequency = bundle.frequency
    step = bundle.step.metrics
    columns = st.columns(6)
    columns[0].metric("Margem de ganho", format_number(frequency.gain_margin_db, "dB"))
    columns[1].metric("Margem de fase", format_number(frequency.phase_margin_deg, "°"))
    columns[2].metric("Overshoot", format_number(step.overshoot_percent, "%"))
    columns[3].metric("Tempo de subida", format_number(step.rise_time_s, "s"))
    columns[4].metric("Acomodação", format_number(step.settling_time_s, "s"))
    columns[5].metric("Erro permanente", format_number(step.steady_state_error))


def _diagnostics_panel(bundle: AnalysisBundle) -> None:
    if not bundle.diagnostics:
        st.success("Nenhum aviso numérico ou de modelagem para esta análise.")
        return
    with st.expander(f"Avisos e observações ({len(bundle.diagnostics)})", expanded=True):
        for diagnostic in bundle.diagnostics:
            message = f"**{diagnostic.analysis}:** {diagnostic.message}"
            if diagnostic.severity is Severity.ERROR:
                st.error(message)
            elif diagnostic.severity is Severity.WARNING:
                st.warning(message)
            else:
                st.info(message)


def _overview_tab(bundle: AnalysisBundle) -> None:
    show_closed = st.toggle("Sobrepor polos de T(s) para K=1", value=False)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(pole_zero_figure(bundle, show_closed), width="stretch", key="overview_pole_zero")
    with right:
        st.plotly_chart(step_figure(bundle), width="stretch", key="overview_step")
    pole_col, closed_col = st.columns(2)
    with pole_col:
        st.subheader("Polos e zeros de G(s)")
        rows = pole_table(bundle.pole_zero.open_poles)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("G(s) não possui polos finitos.")
    with closed_col:
        st.subheader("Polos de T(s)")
        rows = pole_table(bundle.pole_zero.closed_poles)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("T(s) não possui polos finitos.")
    _diagnostics_panel(bundle)


def _bode_tab(bundle: AnalysisBundle) -> None:
    st.plotly_chart(bode_figure(bundle), width="stretch", key="bode_chart")
    frequency = bundle.frequency
    columns = st.columns(4)
    columns[0].metric("MG (fator)", format_number(frequency.gain_margin))
    columns[1].metric("MG (dB)", format_number(frequency.gain_margin_db, "dB"))
    columns[2].metric("MF", format_number(frequency.phase_margin_deg, "°"))
    columns[3].metric(
        "Cruzamento de ganho", format_frequency(frequency.gain_crossover_rad_s, bundle.options.frequency_unit)
    )
    with st.expander("Todos os cruzamentos"):
        rows = [
            {
                "Tipo": "Margem de ganho",
                "Margem": format_number(item.gain_db, "dB"),
                "Frequência": format_frequency(item.omega_rad_s, bundle.options.frequency_unit),
            }
            for item in frequency.gain_margins
        ]
        rows.extend(
            {
                "Tipo": "Margem de fase",
                "Margem": format_number(item.phase_deg, "°"),
                "Frequência": format_frequency(item.omega_rad_s, bundle.options.frequency_unit),
            }
            for item in frequency.phase_margins
        )
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("Não há cruzamentos finitos adicionais.")
    if bundle.pole_zero.open_rhp_poles or len(frequency.gain_margins) > 1 or len(frequency.phase_margins) > 1:
        st.warning("Margens isoladas não substituem o critério de Nyquist e a verificação dos polos fechados.")


def _nyquist_tab(bundle: AnalysisBundle) -> None:
    focus = st.toggle("Focar a região do ponto crítico −1+j0", value=True)
    chart, criterion = st.columns([7, 3])
    with chart:
        st.plotly_chart(nyquist_figure(bundle, focus), width="stretch", key="nyquist_chart")
    with criterion:
        st.subheader("Critério de estabilidade")
        st.metric("P — polos abertos no SPD", bundle.nyquist.open_rhp_poles)
        st.metric("N — envolvimentos horários", bundle.nyquist.encirclements)
        st.metric("Z previsto = P + N", bundle.nyquist.predicted_closed_rhp_poles)
        st.metric("Z pelos polos fechados", bundle.nyquist.actual_closed_rhp_poles)
        if bundle.nyquist.criterion_consistent:
            st.success(f"Critério consistente: conclusão {bundle.nyquist.conclusion.value}.")
        else:
            st.warning("Conclusão inconclusiva: revise faixa, amostragem e avisos numéricos.")
        st.caption("Convenção adotada: envolvimentos no sentido horário são positivos e N = Z − P.")


def _root_locus_tab(bundle: AnalysisBundle) -> None:
    st.plotly_chart(root_locus_figure(bundle), width="stretch", key="root_locus_chart")
    result = bundle.root_locus
    st.caption("O ganho selecionado é exploratório. Bode, Nyquist e degrau continuam usando K=1.")
    left, right = st.columns(2)
    with left:
        st.subheader(f"Polos para K={result.selected_gain:g}")
        rows = selected_pole_table(result.selected_poles)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("Não há polos finitos para o ganho selecionado.")
    with right:
        st.subheader("Verificação de K=1")
        if result.unity_consistent:
            st.success("Os polos do lugar das raízes coincidem com os polos de T(s).")
        else:
            st.error("Foi detectada inconsistência nos polos para K=1.")


def _time_tab(bundle: AnalysisBundle) -> None:
    st.plotly_chart(step_figure(bundle), width="stretch", key="time_step_chart")
    metrics = bundle.step.metrics
    rows = [
        {"Métrica": "Tempo de subida (10–90%)", "Valor": format_number(metrics.rise_time_s, "s")},
        {
            "Métrica": f"Tempo de acomodação ({100 * bundle.options.settling_threshold:g}%)",
            "Valor": format_number(metrics.settling_time_s, "s"),
        },
        {"Métrica": "Overshoot", "Valor": format_number(metrics.overshoot_percent, "%")},
        {"Métrica": "Pico", "Valor": format_number(metrics.peak)},
        {"Métrica": "Instante do pico", "Valor": format_number(metrics.peak_time_s, "s")},
        {"Métrica": "Valor em regime permanente", "Valor": format_number(metrics.steady_state_value)},
        {"Métrica": "Erro em regime permanente", "Valor": format_number(metrics.steady_state_error)},
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    if bundle.closed_loop_stability.value != "estável":
        st.warning("As métricas de desempenho são N/A porque a malha não é assintoticamente estável.")


def _report_tab(bundle: AnalysisBundle) -> None:
    st.write(
        "O pacote contém relatório PDF A4, cinco figuras em PNG/SVG e um manifesto JSON com os resultados e versões das bibliotecas."
    )
    if st.button("Preparar relatório", type="primary"):
        try:
            with st.spinner("Gerando figuras e relatório…"):
                st.session_state.report_bytes = build_export(bundle)
        except Exception as exc:
            st.session_state.report_bytes = None
            st.error(f"Não foi possível gerar o relatório: {exc}")
    if st.session_state.report_bytes:
        st.download_button(
            "Baixar relatório e figuras (.zip)",
            data=st.session_state.report_bytes,
            file_name="analise_sistema_controle.zip",
            mime="application/zip",
            type="primary",
        )


def run_app() -> None:
    st.set_page_config(
        page_title="Análise de Sistemas de Controle",
        page_icon=str(LOGO_FILE),
        layout="wide",
    )
    st.logo(str(LOGO_FILE), size="large", icon_image=str(LOGO_FILE))
    _initialize_state()
    _sidebar_form()

    logo_column, title_column = st.columns([1, 7], vertical_alignment="center")
    with logo_column:
        st.image(str(LOGO_FILE), width=112)
    with title_column:
        st.title("Análise de Sistemas de Controle")
        st.caption("Sistemas SISO contínuos · realimentação negativa unitária · H(s)=1")
    _show_submission_errors()
    bundle = st.session_state.analysis_bundle
    if bundle is None:
        st.info("Informe uma função de transferência válida para iniciar a análise.")
        return

    st.latex(transfer_function_latex(bundle))
    _status_banner(bundle)
    _summary_metrics(bundle)
    tabs = st.tabs(
        [
            "Visão geral",
            "Bode e margens",
            "Nyquist",
            "Lugar das raízes",
            "Resposta temporal",
            "Relatório",
        ]
    )
    with tabs[0]:
        _overview_tab(bundle)
    with tabs[1]:
        _bode_tab(bundle)
    with tabs[2]:
        _nyquist_tab(bundle)
    with tabs[3]:
        _root_locus_tab(bundle)
    with tabs[4]:
        _time_tab(bundle)
    with tabs[5]:
        _report_tab(bundle)
