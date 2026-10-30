from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
import matplotlib.pyplot as plt
from pubfig import (PublicationConfig, publication_style, validate_figure,
                    save_publication_figure, line_plot, bar_plot, scatter_plot, heatmap)

@pytest.fixture
def df():
    return pd.DataFrame({'x':[1,2,3,4], 'y':[2.0,3.0,2.5,5.0], 'group':['A','B','A','B']})

def test_all_four_chart_types(df):
    makers = [
        lambda: line_plot(df, 'x', 'y'),
        lambda: bar_plot(df, 'group', 'y'),
        lambda: scatter_plot(df, 'x', 'y'),
        lambda: heatmap(df[['x','y']].corr()),
    ]
    for make in makers:
        fig, ax = make()
        assert fig.axes and ax in fig.axes
        plt.close(fig)

def test_missing_column_rejected(df):
    with pytest.raises(ValueError, match='Missing required column'):
        line_plot(df, 'not_x', 'y')

def test_non_numeric_and_nan_rejected(df):
    broken = df.copy(); broken.loc[0, 'y'] = np.nan
    with pytest.raises(ValueError, match='missing or non-numeric'):
        scatter_plot(broken, 'x', 'y')

def test_empty_data_rejected():
    with pytest.raises(ValueError, match='at least one row'):
        bar_plot(pd.DataFrame({'x':[], 'y':[]}), 'x', 'y')

def test_preflight_flags_small_figure():
    fig, _ = plt.subplots(figsize=(2,1), dpi=100)
    notes = validate_figure(fig)
    assert any('width' in x.lower() for x in notes)
    assert any('height' in x.lower() for x in notes)
    assert any('dpi' in x.lower() for x in notes)
    plt.close(fig)

def test_exports_png_pdf_svg_and_raster_dpi(df, tmp_path):
    fig, _ = line_plot(df, 'x', 'y')
    result = save_publication_figure(fig, 'example', tmp_path, dpi=600)
    assert set(result['files']) == {'png','pdf','svg'}
    for path in result['files'].values(): assert Path(path).is_file() and Path(path).stat().st_size > 100
    with Image.open(result['files']['png']) as im:
        assert im.info['dpi'][0] >= 590
        assert im.width > 1000
    assert 'pdf' in result['vector_formats'] and 'svg' in result['vector_formats']
    plt.close(fig)

def test_heatmap_annotation_limit_warns():
    data = pd.DataFrame(np.ones((21,20)))
    with pytest.warns(UserWarning, match='more than 400 cells'):
        fig, _ = heatmap(data, annot=True)
    plt.close(fig)

def test_bad_export_format_rejected(df, tmp_path):
    fig, _ = line_plot(df, 'x', 'y')
    with pytest.raises(ValueError, match='formats'):
        save_publication_figure(fig, 'x', tmp_path, formats=('jpg',))
    plt.close(fig)

def test_scatter_plot_uses_collection_without_lines(df):
    fig, ax = scatter_plot(df, 'x', 'y', alpha=0.25, marker_size=18)
    assert len(ax.collections) == 1
    assert len(ax.lines) == 0
    assert ax.collections[0].get_alpha() == 0.25
    plt.close(fig)

def test_scatter_rejects_invalid_alpha(df):
    with pytest.raises(ValueError, match='alpha'):
        scatter_plot(df, 'x', 'y', alpha=0)

def test_month_strings_use_compact_date_ticks_and_inr_format():
    data = pd.DataFrame({
        "Month": pd.date_range("2023-01-01", periods=36, freq="MS").strftime("%Y-%m"),
        "Revenue_INR": np.linspace(1_900_000, 3_200_000, 36),
    })
    fig, ax = line_plot(data, "Month", "Revenue_INR")
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert any("₹" in label and "M" in label for label in labels)
    assert len(ax.get_xticklabels()) <= 10
    plt.close(fig)


def test_scatter_inr_axis_is_formatted(df):
    data = pd.DataFrame({"Revenue_INR": [1_000_000, 2_000_000, 3_000_000], "Profit_INR": [200_000, 500_000, 800_000]})
    fig, ax = scatter_plot(data, "Revenue_INR", "Profit_INR")
    assert any("₹" in tick.get_text() for tick in ax.get_xticklabels())
    assert any("₹" in tick.get_text() for tick in ax.get_yticklabels())
    plt.close(fig)
