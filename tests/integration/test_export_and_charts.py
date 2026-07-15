from __future__ import annotations

import io
import json
import zipfile

from control_dashboard import AnalysisOptions, TransferFunctionSpec, analyze
from control_dashboard.reporting import build_export
from control_dashboard.visualization.plotly_charts import (
    bode_figure,
    nyquist_figure,
    pole_zero_figure,
    root_locus_figure,
    step_figure,
)


def _bundle():
    return analyze(
        TransferFunctionSpec((1,), (1, 1, 0)),
        AnalysisOptions(frequency_points=300, nyquist_points=800),
    )


def test_plotly_renderers_use_the_analysis_bundle_without_recalculation_errors() -> None:
    bundle = _bundle()
    figures = [
        pole_zero_figure(bundle),
        bode_figure(bundle),
        nyquist_figure(bundle),
        root_locus_figure(bundle),
        step_figure(bundle),
    ]
    assert all(figure.data for figure in figures)
    assert len(bode_figure(bundle).data[0].x) == len(bundle.frequency.omega_rad_s)


def test_export_contains_valid_pdf_figures_and_manifest() -> None:
    archive_bytes = build_export(_bundle())
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        assert "relatorio_analise_controle.pdf" in names
        assert "analysis.json" in names
        for stem in ("polos_zeros", "bode", "nyquist", "lugar_raizes", "resposta_degrau"):
            assert f"figuras/{stem}.png" in names
            assert f"figuras/{stem}.svg" in names
        assert archive.read("relatorio_analise_controle.pdf").startswith(b"%PDF")
        manifest = json.loads(archive.read("analysis.json"))
        assert manifest["analysis"]["closed_loop_stability"] == "estável"
        assert manifest["analysis"]["nyquist"]["criterion_consistent"] is True
