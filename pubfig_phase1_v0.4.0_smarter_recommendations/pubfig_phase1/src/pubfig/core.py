"""Core plotting API for PubFig."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings as pywarnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes

PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "black": "#222222", "grey": "#777777",
}


def _format_financial_axis(ax: Axes, values, axis: str = "y") -> None:
    """Format monetary values with compact k/M/B labels (e.g. ₹2.5M)."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return
    maximum = float(numeric.abs().max())
    if maximum >= 1_000_000_000:
        divisor, suffix = 1_000_000_000, "B"
    elif maximum >= 1_000_000:
        divisor, suffix = 1_000_000, "M"
    elif maximum >= 1_000:
        divisor, suffix = 1_000, "k"
    else:
        divisor, suffix = 1, ""

    def _format(value, _position):
        scaled = value / divisor
        return f"₹{scaled:.1f}{suffix}" if suffix else f"₹{value:,.0f}"

    formatter = FuncFormatter(_format)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        raise ValueError("axis must be 'x' or 'y'")

def _prepare_line_x(data: pd.DataFrame, x: str):
    """Parse datetime columns and common YYYY-MM month strings when possible."""
    values = data[x]
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.notna().all():
            return parsed, True
    if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
        sample = values.dropna().astype(str).head(100)
        if not sample.empty:
            import re
            pattern = re.compile(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[ T].*)?$")
            if sample.map(lambda v: bool(pattern.match(v.strip()))).mean() >= 0.8:
                parsed = pd.to_datetime(values, errors="coerce")
                if parsed.notna().all():
                    return parsed, True
    return values, False

def _format_date_axis(fig: Figure, ax: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=8)


@dataclass(frozen=True)
class PublicationConfig:
    """Figure defaults; values are defaults, not a guarantee of venue compliance."""
    width_in: float = 6.5
    height_in: float = 3.6
    dpi: int = 600
    min_width_in: float = 3.25
    min_height_in: float = 2.0
    min_dpi: int = 300
    font_family: str = "DejaVu Sans"
    font_size: float = 9.0


def publication_style(config: PublicationConfig | None = None) -> None:
    """Apply consistent defaults with non-color visual encodings available to authors."""
    config = config or PublicationConfig()
    plt.rcParams.update({
        "font.family": config.font_family, "font.size": config.font_size,
        "axes.labelsize": config.font_size, "axes.titlesize": config.font_size + 1,
        "legend.fontsize": max(7, config.font_size - 1),
        "xtick.labelsize": max(7, config.font_size - 1),
        "ytick.labelsize": max(7, config.font_size - 1),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.7,
        "figure.dpi": 120, "savefig.dpi": config.dpi,
        "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _check_columns(data: pd.DataFrame, columns: list[str]) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if data.empty:
        raise ValueError("data must contain at least one row")


def _numeric_values(data: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        vals = pd.to_numeric(data[col], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"Column '{col}' contains missing or non-numeric values")
        if not np.isfinite(vals.to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains non-finite values")


def _axes(ax: Axes | None, figsize: tuple[float, float] = (6.5, 3.6)) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def line_plot(data: pd.DataFrame, x: str, y: str, *, label: str | None = None,
              title: str = "", xlabel: str | None = None, ylabel: str | None = None,
              marker: str = "o", color: str = PALETTE["blue"],
              linestyle: str = "-", ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Create a line chart. Markers and line style supplement color for distinction."""
    _check_columns(data, [x, y]); _numeric_values(data, [y])
    publication_style(); fig, ax = _axes(ax)
    plot_data = data.copy()
    plot_x, is_date = _prepare_line_x(plot_data, x)
    plot_data["_pubfig_x"] = plot_x
    if is_date:
        plot_data = plot_data.sort_values("_pubfig_x")
    ax.plot(plot_data["_pubfig_x"], plot_data[y], color=color, marker=marker,
            markersize=3.5, markerfacecolor="white", linestyle=linestyle, label=label)
    if is_date:
        _format_date_axis(fig, ax)
    elif plot_data["_pubfig_x"].nunique() > 12:
        step = int(np.ceil(plot_data["_pubfig_x"].nunique() / 8))
        ax.set_xticks(ax.get_xticks()[::step])
        ax.tick_params(axis="x", labelrotation=25)
    ax.set(title=title, xlabel=xlabel or x, ylabel=ylabel or y)
    if "inr" in (ylabel or y).lower():
        _format_financial_axis(ax, plot_data[y], axis="y")
    if label: ax.legend()
    fig.tight_layout()
    return fig, ax


def bar_plot(data: pd.DataFrame, x: str, y: str, *, title: str = "",
             xlabel: str | None = None, ylabel: str | None = None,
             color: str = PALETTE["sky"], ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Create a single-series categorical bar chart."""
    _check_columns(data, [x, y]); _numeric_values(data, [y])
    publication_style(); fig, ax = _axes(ax)
    ax.bar(data[x].astype(str), data[y], color=color, edgecolor=PALETTE["black"], linewidth=.5)
    ax.set(title=title, xlabel=xlabel or x, ylabel=ylabel or y)
    if "inr" in (ylabel or y).lower():
        _format_financial_axis(ax, data[y], axis="y")
    if len(data) > 12:
        step = int(np.ceil(len(data) / 12))
        for i, tick in enumerate(ax.get_xticklabels()):
            tick.set_visible(i % step == 0)
            tick.set_rotation(25)
            tick.set_ha("right")
    fig.tight_layout()
    return fig, ax


def scatter_plot(data: pd.DataFrame, x: str, y: str, *, title: str = "",
                 xlabel: str | None = None, ylabel: str | None = None,
                 label: str | None = None, color: str = PALETTE["vermillion"],
                 marker: str = "o", alpha: float = 0.35, marker_size: float = 24,
                 ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Create a scatter plot with a high-contrast default marker."""
    _check_columns(data, [x, y]); _numeric_values(data, [x, y])
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")
    if marker_size <= 0:
        raise ValueError("marker_size must be positive")
    publication_style(); fig, ax = _axes(ax)
    ax.scatter(data[x], data[y], label=label, color=color, marker=marker, s=marker_size,
               edgecolors=PALETTE["black"], linewidths=.25, alpha=alpha)
    ax.set(title=title, xlabel=xlabel or x, ylabel=ylabel or y)
    if "inr" in (ylabel or y).lower():
        _format_financial_axis(ax, data[y], axis="y")
    if "inr" in (xlabel or x).lower():
        _format_financial_axis(ax, data[x], axis="x")
    if label: ax.legend()
    fig.tight_layout()
    return fig, ax


def heatmap(data: pd.DataFrame, *, title: str = "", xlabel: str = "", ylabel: str = "",
            cmap: str = "viridis", annot: bool = False, fmt: str = ".2g",
            ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Create a labeled numeric matrix heatmap without requiring seaborn."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one row and column")
    arr = data.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("heatmap data must be numeric, finite, and non-missing")
    publication_style(); fig, ax = _axes(ax, (6.0, 4.5))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(data.shape[1]), labels=[str(c) for c in data.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(data.shape[0]), labels=[str(i) for i in data.index])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Value")
    if annot and arr.size <= 400:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center", fontsize=7)
    elif annot and arr.size > 400:
        pywarnings.warn("Annotation disabled for matrices with more than 400 cells", UserWarning)
    fig.tight_layout()
    return fig, ax


def validate_figure(fig: Figure, *, min_width_in: float = 3.25,
                    min_height_in: float = 2.0, min_dpi: int = 300) -> list[str]:
    """Return technical preflight warnings; not a venue-specific compliance decision."""
    if min_width_in < 0 or min_height_in < 0 or min_dpi < 0:
        raise ValueError("minimum size and DPI thresholds must be non-negative")
    problems: list[str] = []
    w, h = fig.get_size_inches()
    if w < min_width_in: problems.append(f"Figure width {w:.2f} in is below {min_width_in:.2f} in.")
    if h < min_height_in: problems.append(f"Figure height {h:.2f} in is below {min_height_in:.2f} in.")
    if fig.dpi < min_dpi: problems.append(f"Canvas DPI {fig.dpi:g} is below {min_dpi}.")
    if not fig.axes: problems.append("Figure contains no axes.")
    return problems


def save_publication_figure(fig: Figure, stem: str | Path, outdir: str | Path = "figures",
                            *, dpi: int = 600, formats: tuple[str, ...] = ("png", "pdf", "svg"),
                            config: PublicationConfig | None = None) -> dict[str, Any]:
    """Save requested formats and return paths plus technical preflight warnings."""
    config = config or PublicationConfig()
    if dpi < 1: raise ValueError("dpi must be a positive integer")
    allowed = {"png", "pdf", "svg"}
    if not formats or any(f.lower().lstrip(".") not in allowed for f in formats):
        raise ValueError("formats must be a non-empty subset of png, pdf, svg")
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    name = Path(stem).stem
    paths: dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = out / f"{name}.{fmt}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt == "png": kwargs["dpi"] = dpi
        fig.savefig(path, format=fmt, **kwargs)
        paths[fmt] = str(path.resolve())
    warnings = validate_figure(fig, min_width_in=config.min_width_in,
                               min_height_in=config.min_height_in, min_dpi=0)
    if "png" in paths and dpi < config.min_dpi:
        warnings.append(f"Requested PNG export DPI {dpi} is below {config.min_dpi}.")
    return {"files": paths, "warnings": warnings, "export_dpi": dpi,
            "vector_formats": [f for f in paths if f in {"pdf", "svg"}]}
