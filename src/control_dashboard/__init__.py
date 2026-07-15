"""Painel local para análise de sistemas de controle SISO contínuos."""

from control_dashboard.application.analysis_service import analyze
from control_dashboard.domain.models import AnalysisOptions, TransferFunctionSpec

__all__ = ["AnalysisOptions", "TransferFunctionSpec", "analyze"]
