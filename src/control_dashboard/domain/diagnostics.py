from __future__ import annotations

from collections.abc import Iterable

from control_dashboard.domain.models import Diagnostic


class AnalysisValidationError(ValueError):
    """Erro de entrada apresentado de forma controlada pela interface."""

    def __init__(self, diagnostics: Iterable[Diagnostic]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(item.message for item in self.diagnostics))


class AnalysisComputationError(RuntimeError):
    """Falha que impede a construção coerente do sistema analisado."""

    def __init__(self, diagnostics: Iterable[Diagnostic]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(item.message for item in self.diagnostics))
