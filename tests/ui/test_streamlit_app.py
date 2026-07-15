from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"


def test_app_starts_with_complete_dashboard() -> None:
    app = AppTest.from_file(str(APP)).run(timeout=45)
    assert not app.exception
    assert app.title[0].value == "Análise de Sistemas de Controle"
    assert [tab.label for tab in app.tabs] == [
        "Visão geral",
        "Bode e margens",
        "Nyquist",
        "Lugar das raízes",
        "Resposta temporal",
        "Relatório",
    ]
    assert len(app.metric) >= 6
    assert len(app.get("plotly_chart")) == 6


def test_invalid_submission_keeps_last_valid_analysis() -> None:
    app = AppTest.from_file(str(APP)).run(timeout=45)
    initial_value = app.metric[0].value
    app.text_input(key="numerator_input").set_value("")
    app.button(key="FormSubmitter:transfer_function_form-Analisar sistema").click()
    app.run(timeout=45)
    assert not app.exception
    assert app.metric[0].value == initial_value
    assert any("último resultado válido" in error.value for error in app.error)
    assert any("numerador" in error.value for error in app.error)
