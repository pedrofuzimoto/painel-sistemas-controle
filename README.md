# Painel de Análise de Sistemas de Controle

[![Abrir no Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://painel-sistemas-controle.streamlit.app/)

**Aplicação online:** https://painel-sistemas-controle.streamlit.app/

Aplicação web em português para análise de sistemas SISO contínuos descritos por
uma função de transferência de malha aberta.

## Funcionalidades

- validação e normalização dos coeficientes de `G(s)`;
- mapa de polos e zeros;
- diagramas de Bode e margens de estabilidade;
- diagrama de Nyquist com verificação de `P`, `N` e `Z`;
- lugar das raízes para ganhos não negativos;
- resposta ao degrau em malha fechada com métricas temporais;
- exportação de relatório técnico em PDF, PNG, SVG e JSON.

A aplicação considera sistemas SISO contínuos, racionais e próprios, com
realimentação negativa unitária.

## Executar localmente

Requer Python 3.12 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Abra `http://localhost:8501` no navegador.

## Desenvolvimento e testes

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests streamlit_app.py
```

O conjunto de regressão acadêmica compara polos, resposta temporal, erro em regime
permanente, lugar das raízes, Nyquist e margens com exemplos da 6ª edição do livro de
Norman S. Nise. Consulte [a validação e as tolerâncias adotadas](docs/validacao-nise.md).

## Arquitetura

O núcleo matemático em `src/control_dashboard/analysis` não depende da interface.
Os resultados são consolidados em DTOs imutáveis e consumidos pelos renderizadores
Plotly, Matplotlib e pelo gerador de relatórios ReportLab.
