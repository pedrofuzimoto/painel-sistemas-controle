from __future__ import annotations

import dataclasses
import io
import json
import math
import zipfile
from datetime import datetime
from enum import Enum
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from control_dashboard.domain.models import AnalysisBundle, PoleRecord
from control_dashboard.visualization.static_charts import make_static_figures
from control_dashboard.visualization.theme import convert_frequency


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "infinito" if value > 0 else "-infinito"
    return value


def _fmt(value: float | None, unit: str = "", precision: int = 5) -> str:
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "infinita"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{precision}g}{suffix}"


def _coefficients(values: tuple[float, ...]) -> str:
    return "[" + "; ".join(f"{value:.9g}" for value in values) + "]"


def _pole_rows(records: tuple[PoleRecord, ...], kind: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in records:
        damping = "N/A" if record.damping_ratio is None else f"{record.damping_ratio:.5g}"
        rows.append(
            [
                kind,
                f"{record.real:.7g}",
                f"{record.imag:+.7g}",
                f"{record.magnitude:.7g}",
                damping,
                str(record.multiplicity),
            ]
        )
    return rows


def _table(data, widths=None) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1559A6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB9C5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#536273"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Página {document.page}")
    canvas.restoreState()


def _build_pdf(
    bundle: AnalysisBundle,
    png_assets: dict[str, bytes],
    generated_at: datetime,
) -> bytes:
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Relatório de análise de sistema de controle",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=colors.HexColor("#243447"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1559A6"),
            spaceBefore=10,
            spaceAfter=7,
        )
    )
    normal = styles["BodyText"]
    normal.leading = 14
    story = [
        Paragraph("Painel de Análise de Sistemas de Controle", styles["ReportTitle"]),
        Paragraph(
            f"Relatório gerado em {generated_at:%d/%m/%Y às %H:%M:%S} — realimentação negativa unitária, H(s)=1.",
            normal,
        ),
        Spacer(1, 10),
        Paragraph("Modelo analisado", styles["Section"]),
        _table(
            [
                ["Item", "Valor"],
                ["Nome", escape(bundle.normalized_spec.name)],
                ["Numerador original", _coefficients(bundle.original_spec.numerator)],
                ["Denominador original", _coefficients(bundle.original_spec.denominator)],
                ["Numerador normalizado", _coefficients(bundle.normalized_spec.numerator)],
                ["Denominador normalizado", _coefficients(bundle.normalized_spec.denominator)],
                ["Estabilidade de G(s)", bundle.open_loop_stability.value],
                ["Estabilidade de T(s)", bundle.closed_loop_stability.value],
            ],
            [5.0 * cm, 11.2 * cm],
        ),
        Paragraph("Indicadores principais", styles["Section"]),
    ]
    frequency_unit = bundle.options.frequency_unit
    frequency = bundle.frequency
    step = bundle.step.metrics
    story.append(
        _table(
            [
                ["Indicador", "Valor"],
                ["Margem de ganho", _fmt(frequency.gain_margin)],
                ["Margem de ganho", _fmt(frequency.gain_margin_db, "dB")],
                ["Margem de fase", _fmt(frequency.phase_margin_deg, "graus")],
                [
                    "Cruzamento de fase",
                    _fmt(convert_frequency(frequency.phase_crossover_rad_s, frequency_unit), frequency_unit),
                ],
                [
                    "Cruzamento de ganho",
                    _fmt(convert_frequency(frequency.gain_crossover_rad_s, frequency_unit), frequency_unit),
                ],
                ["Tempo de subida (10–90%)", _fmt(step.rise_time_s, "s")],
                ["Tempo de acomodação", _fmt(step.settling_time_s, "s")],
                ["Overshoot", _fmt(step.overshoot_percent, "%")],
                ["Valor final", _fmt(step.steady_state_value)],
                ["Erro em regime permanente", _fmt(step.steady_state_error)],
            ],
            [8.0 * cm, 8.2 * cm],
        )
    )
    story.extend(
        [
            Paragraph("Polos e zeros", styles["Section"]),
            _table(
                [["Tipo", "Real", "Imaginária", "Módulo", "Amortecimento", "Mult."]]
                + _pole_rows(bundle.pole_zero.open_poles, "Polo de G")
                + _pole_rows(bundle.pole_zero.open_zeros, "Zero de G")
                + _pole_rows(bundle.pole_zero.closed_poles, "Polo de T"),
                [3.1 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3.1 * cm, 1.3 * cm],
            ),
            Paragraph("Critério de Nyquist", styles["Section"]),
            _table(
                [
                    ["Grandeza", "Valor"],
                    ["P — polos abertos no SPD", str(bundle.nyquist.open_rhp_poles)],
                    ["N — envolvimentos horários", str(bundle.nyquist.encirclements)],
                    ["Z previsto = P + N", str(bundle.nyquist.predicted_closed_rhp_poles)],
                    ["Z obtido dos polos fechados", str(bundle.nyquist.actual_closed_rhp_poles)],
                    ["Conclusão", bundle.nyquist.conclusion.value],
                ],
                [10.0 * cm, 6.2 * cm],
            ),
        ]
    )

    if bundle.diagnostics:
        story.append(Paragraph("Avisos e observações", styles["Section"]))
        for diagnostic in bundle.diagnostics:
            story.append(
                Paragraph(
                    f"<b>{escape(diagnostic.analysis)} — {escape(diagnostic.severity.value)}:</b> {escape(diagnostic.message)}",
                    normal,
                )
            )
            story.append(Spacer(1, 3))

    story.append(PageBreak())
    image_buffers: list[io.BytesIO] = []
    for index, (name, title) in enumerate(
        (
            ("polos_zeros", "Mapa de polos e zeros"),
            ("bode", "Diagrama de Bode"),
            ("nyquist", "Diagrama de Nyquist"),
            ("lugar_raizes", "Lugar das raízes"),
            ("resposta_degrau", "Resposta ao degrau"),
        )
    ):
        story.append(Paragraph(title, styles["Section"]))
        buffer = io.BytesIO(png_assets[name])
        image_buffers.append(buffer)
        height = 11.8 * cm if name == "bode" else 10.5 * cm
        story.append(Image(buffer, width=16.3 * cm, height=height, kind="proportional"))
        if index in {1, 3}:
            story.append(PageBreak())

    story.append(Paragraph("Ambiente de cálculo", styles["Section"]))
    story.append(_table([["Biblioteca", "Versão"], *[list(item) for item in bundle.versions]], [8.0 * cm, 8.2 * cm]))
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return output.getvalue()


def build_export(bundle: AnalysisBundle) -> bytes:
    """Produz um ZIP com PDF, PNG, SVG e manifesto JSON sem persistir arquivos."""

    generated_at = datetime.now(ZoneInfo("America/Sao_Paulo"))
    figures = make_static_figures(bundle)
    png_assets: dict[str, bytes] = {}
    svg_assets: dict[str, bytes] = {}
    try:
        for name, figure in figures.items():
            png = io.BytesIO()
            svg = io.BytesIO()
            figure.savefig(png, format="png", dpi=300, bbox_inches="tight")
            figure.savefig(svg, format="svg", bbox_inches="tight")
            png_assets[name] = png.getvalue()
            svg_assets[name] = svg.getvalue()
    finally:
        for figure in figures.values():
            plt.close(figure)

    pdf = _build_pdf(bundle, png_assets, generated_at)
    manifest = {
        "generated_at": generated_at.isoformat(),
        "analysis": _jsonable(bundle),
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("relatorio_analise_controle.pdf", pdf)
        zip_file.writestr("analysis.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, data in png_assets.items():
            zip_file.writestr(f"figuras/{name}.png", data)
        for name, data in svg_assets.items():
            zip_file.writestr(f"figuras/{name}.svg", data)
    return archive.getvalue()
