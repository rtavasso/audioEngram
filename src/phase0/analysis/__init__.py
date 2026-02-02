"""Analysis and reporting utilities."""

from .run_phase0 import run_full_analysis
from .report import generate_report, make_decision
from .plots import (
    plot_cluster_sizes,
    plot_variance_vs_size,
    plot_pca_scatter,
    plot_confidence_curve,
    create_all_plots,
)

__all__ = [
    "run_full_analysis",
    "generate_report",
    "make_decision",
    "plot_cluster_sizes",
    "plot_variance_vs_size",
    "plot_pca_scatter",
    "plot_confidence_curve",
    "create_all_plots",
]
