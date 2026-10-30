"""PubFig: reproducible, publication-oriented plotting helpers.

Research prototype. It offers technical preflight checks, not official IEEE/ACM certification.
"""
from .core import (
    PALETTE, PublicationConfig, publication_style, validate_figure,
    save_publication_figure, line_plot, bar_plot, scatter_plot, heatmap,
)
__all__ = [
    "PALETTE", "PublicationConfig", "publication_style", "validate_figure",
    "save_publication_figure", "line_plot", "bar_plot", "scatter_plot", "heatmap",
]
__version__ = "0.1.0"
