"""PubFig interactive dashboard — expanded prototype."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

from pubfig import (
    line_plot, bar_plot, scatter_plot, heatmap,
    save_publication_figure, validate_figure,
)

st.set_page_config(
    page_title="PubFig Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
.pubfig-hero {padding: 1.35rem 1.5rem; border-radius: 18px;
  background: linear-gradient(115deg, #10243a 0%, #153d57 55%, #176b70 100%);
  color: #f5fbff; margin-bottom: 1rem;}
.pubfig-hero h1 {font-size: 2.15rem; margin: 0 0 .25rem 0; color: white;}
.pubfig-hero p {margin: 0; color: #d5e8f1; font-size: 1rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
  padding: .8rem 1rem; border-radius: 12px;}
div[data-testid="stTabs"] button {font-weight: 600;}
.small-muted {font-size: .86rem; opacity: .78;}
</style>
<div class="pubfig-hero">
  <h1>PubFig Studio</h1>
  <p>Build, inspect, and export publication-oriented figures — reproducibly.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("Import → Configure → Inspect → Export")
    upload = st.file_uploader("Upload dataset", type=["csv"])
    st.markdown("---")
    st.markdown("**Figure defaults**")
    theme = st.selectbox("Visual theme", ["Publication light", "Minimal", "Presentation"], index=0)
    width = st.selectbox("Figure width", ["Single column (3.5 in)", "Wide (6.5 in)", "Presentation (9 in)"], index=1)
    export_dpi = st.select_slider("Raster export resolution", options=[150, 300, 450, 600, 900], value=600)
    st.caption("PDF and SVG exports are vector formats; DPI primarily affects raster output.")

if upload is None:
    st.info("Upload a CSV to start. You can use the 5,000-row synthetic test dataset to exercise the chart workflow.")
    st.markdown("### Included chart types")
    cols = st.columns(4)
    for col, title, desc in zip(cols, ["Line & area", "Bar", "Scatter", "Distribution"], [
        "Ordered trends and time series", "Category comparisons with aggregation",
        "Relationships between numeric variables", "Histogram, box and violin plots"]):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.markdown("Also includes heatmaps, correlation analysis, dataset checks, and PNG/PDF/SVG export.")
    st.stop()

try:
    data = pd.read_csv(upload)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

numeric_cols = list(data.select_dtypes(include="number").columns)
categorical_cols = [c for c in data.columns if c not in numeric_cols]
date_cols = []
for col in data.columns:
    if col in numeric_cols:
        continue
    if any(token in col.lower() for token in ["date", "time", "month", "year"]):
        parsed = pd.to_datetime(data[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            date_cols.append(col)


def _is_identifier_column(series: pd.Series, name: str) -> bool:
    """Heuristically identify columns that are IDs rather than analytic variables."""
    label = name.lower().replace(" ", "_")
    id_tokens = ("id", "uuid", "guid", "transaction", "customer_number", "row_number", "index")
    named_like_id = any(token in label for token in id_tokens)
    non_null = series.dropna()
    if non_null.empty:
        return True
    unique_ratio = non_null.nunique() / len(non_null)
    return named_like_id or (len(non_null) >= 20 and unique_ratio >= 0.98)


def _preferred_aggregation(frame: pd.DataFrame, date_col: str) -> tuple[str, str]:
    """Choose a readable time bucket based on the number of unique timestamps."""
    parsed = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if parsed.empty:
        return "D", "daily"
    span_days = max(1, (parsed.max() - parsed.min()).days)
    unique_days = parsed.dt.floor("D").nunique()
    if span_days >= 730 or unique_days > 500:
        return "MS", "monthly"
    if span_days >= 120 or unique_days > 120:
        return "W", "weekly"
    return "D", "daily"


def _make_recommendations(frame: pd.DataFrame, numeric: list[str], categorical: list[str], dates: list[str]):
    """Create transparent chart candidates using data quality and variable-role heuristics."""
    usable_numeric = []
    for col in numeric:
        if _is_identifier_column(frame[col], col):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(vals) >= 3 and vals.nunique() >= 2:
            usable_numeric.append(col)

    usable_categories = []
    for col in categorical:
        if col in dates or _is_identifier_column(frame[col], col):
            continue
        nunique = frame[col].nunique(dropna=True)
        if 2 <= nunique <= min(30, max(10, int(len(frame) * 0.2))):
            usable_categories.append(col)

    # Variable names are weak clues, not proof of meaning. Use them only to rank candidates.
    def semantic_score(col: str, role: str) -> int:
        name = col.lower().replace(" ", "_")
        tokens = {
            "outcome": ("revenue", "sales", "profit", "cost", "price", "units", "quantity", "satisfaction", "score", "duration", "days", "amount", "total"),
            "discount": ("discount", "promotion", "rate", "percent", "percentage"),
            "time": ("date", "time", "month", "year", "week", "day"),
            "category": ("region", "category", "segment", "channel", "type", "group", "department", "product"),
        }
        return 1 if any(t in name for t in tokens.get(role, ())) else 0

    recs = []
    if dates and usable_numeric:
        # Choose a likely business outcome where possible, not simply the first numeric column.
        y = max(usable_numeric, key=lambda c: (semantic_score(c, "outcome"), frame[c].notna().mean(), -usable_numeric.index(c)))
        x = max(dates, key=lambda c: (semantic_score(c, "time"), -dates.index(c)))
        freq, freq_label = _preferred_aggregation(frame, x)
        recs.append({"kind": "Line", "title": f"{freq_label.title()} trend in {y}", "x": x, "y": y,
                     "reason": f"{x} is date-like and {y} is numeric. Aggregating observations {freq_label} reduces timestamp-level noise and makes the time trend easier to interpret.",
                     "score": 88 + 4 * semantic_score(y, "outcome"), "agg": "Mean", "time_freq": freq, "time_label": freq_label})

    # Rank category/measure pairs by semantic fit and usable data, not column order alone.
    category_pairs = []
    for x in usable_categories:
        groups = frame[x].nunique(dropna=True)
        for y in usable_numeric:
            score = 70 + 5 * semantic_score(y, "outcome") + 4 * semantic_score(x, "category")
            if groups <= 12:
                score += 3
            if frame[[x, y]].dropna().shape[0] < 10:
                score -= 12
            category_pairs.append((score, x, y))
    category_pairs.sort(key=lambda item: item[0], reverse=True)
    if category_pairs:
        score, x, y = category_pairs[0]
        agg = "Sum" if any(t in y.lower() for t in ("revenue", "sales", "profit", "cost", "amount", "total", "units", "quantity")) else "Mean"
        action = "total" if agg == "Sum" else "mean"
        recs.append({"kind": "Bar", "title": f"{action.title()} {y} by {x}", "x": x, "y": y,
                     "reason": f"Compare the {action} {y} across {frame[x].nunique(dropna=True)} groups in {x}. {agg} is a suggested aggregation; choose Sum for additive quantities and Mean for typical per-record values.",
                     "score": min(96, score), "agg": agg})
        # Pick a second useful group/measure pair for distribution comparison if enough data.
        box_candidates = [(70 + 4 * semantic_score(y2, "outcome") + (3 if frame[x].nunique() <= 8 else 0), x, y2)
                          for x in usable_categories for y2 in usable_numeric if frame[x].nunique() <= 12 and len(frame[[x, y2]].dropna()) >= 10]
        box_candidates.sort(key=lambda item: item[0], reverse=True)
        if box_candidates:
            bscore, bx, by = box_candidates[0]
            if (bx, by) != (x, y):
                recs.append({"kind": "Box plot", "title": f"Distribution of {by} across {bx}", "x": bx, "y": by,
                             "reason": f"A grouped box plot compares medians, spread, and potential outliers of {by} across {bx}. Check group sample sizes before interpreting differences.",
                             "score": min(88, bscore), "agg": "None"})

    # Prefer analytically meaningful numeric pairs; penalize constant/near-constant columns.
    scatter_candidates = []
    for i, x in enumerate(usable_numeric):
        for y in usable_numeric[i + 1:]:
            pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 10 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
                continue
            score = 72 + 5 * semantic_score(x, "discount") + 5 * semantic_score(y, "outcome") + 3 * semantic_score(x, "outcome") + 3 * semantic_score(y, "discount")
            # Prefer pairs with less missing data; avoid implying a relationship is meaningful just because it is numeric.
            score += int(3 * len(pair) / max(1, len(frame)))
            scatter_candidates.append((score, x, y))
    scatter_candidates.sort(key=lambda item: item[0], reverse=True)
    if scatter_candidates:
        score, x, y = scatter_candidates[0]
        recs.append({"kind": "Scatter", "title": f"Relationship between {x} and {y}", "x": x, "y": y,
                     "reason": f"Explore association, clusters, and outliers between {x} and {y}. This is descriptive and does not establish causation.",
                     "score": min(91, score), "agg": "None"})

    if usable_numeric:
        y = max(usable_numeric, key=lambda c: (semantic_score(c, "outcome"), frame[c].notna().mean(), -usable_numeric.index(c)))
        recs.append({"kind": "Histogram", "title": f"Distribution of {y}", "x": None, "y": y,
                     "reason": f"A histogram shows the shape, spread, and possible skew of {y}; bin width can change the apparent pattern.",
                     "score": 78 + 4 * semantic_score(y, "outcome"), "agg": "None"})

    if len(usable_numeric) >= 3:
        recs.append({"kind": "Heatmap", "title": "Numeric correlation overview", "x": None, "y": None,
                     "reason": "Summarises pairwise linear associations among numeric measures. Correlation is not causation, and outliers can influence Pearson correlation.",
                     "score": 74, "agg": "None"})

    # De-duplicate candidates and cap the list for a focused UI.
    unique = {}
    for rec in recs:
        key = (rec["kind"], rec.get("x"), rec.get("y"))
        if key not in unique or rec["score"] > unique[key]["score"]:
            unique[key] = rec
    return sorted(unique.values(), key=lambda r: r["score"], reverse=True)[:6]


m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(data):,}")
m2.metric("Columns", f"{len(data.columns):,}")
m3.metric("Numeric fields", f"{len(numeric_cols):,}")
m4.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")

tab_recommendations, tab_build, tab_data, tab_quality = st.tabs(["✨ Chart recommendations", "🎨 Figure builder", "🧾 Dataset", "🩺 Data checks"])


with tab_recommendations:
    st.subheader("Recommendations for this dataset")
    st.caption("Suggestions are generated from transparent data-type and cardinality rules. Review each chart in context; recommendations are not a substitute for domain knowledge.")
    recommendations = _make_recommendations(data, numeric_cols, categorical_cols, date_cols)
    recommendation_numeric = [c for c in numeric_cols if not _is_identifier_column(data[c], c)]
    if not recommendations:
        st.warning("Not enough suitable variables were detected. Try a dataset with numeric measures and/or date or categorical fields.")
    else:
        st.write(f"Found **{len(recommendations)}** candidate charts from {len(data.columns)} columns.")
        for i, rec in enumerate(recommendations, start=1):
            with st.container(border=True):
                head1, head2 = st.columns([4, 1])
                with head1:
                    st.markdown(f"**{i}. {rec['title']}**")
                    st.caption(f"Chart: {rec['kind']} · Heuristic suitability: {rec['score']}/100")
                with head2:
                    st.metric("Rank", f"#{i}")
                st.write(rec["reason"])
                try:
                    fig_rec, ax_rec = plt.subplots(figsize=(6.5, 3.4))
                    if rec["kind"] == "Line":
                        work = data[[rec["x"], rec["y"]]].dropna().copy()
                        work[rec["x"]] = pd.to_datetime(work[rec["x"]], errors="coerce")
                        work = work.dropna().set_index(rec["x"])
                        if not work.empty:
                            freq = rec.get("time_freq", "D")
                            # Use mean as the default for typical values; additive measures may use sum.
                            if any(t in rec["y"].lower() for t in ("revenue", "sales", "profit", "cost", "amount", "total", "units", "quantity")):
                                series = work[rec["y"]].resample(freq).sum(min_count=1)
                                ylabel = f"Total {rec['y']} ({rec.get('time_label', 'time')})"
                            else:
                                series = work[rec["y"]].resample(freq).mean()
                                ylabel = f"Mean {rec['y']} ({rec.get('time_label', 'time')})"
                            series = series.dropna()
                            ax_rec.plot(series.index, series.values, color="#CC79A7", linewidth=1.8, marker="o" if len(series) <= 36 else None, markersize=3)
                            if len(series) > 0:
                                ax_rec.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
                                ax_rec.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_rec.xaxis.get_major_locator()))
                            fig_rec.autofmt_xdate(rotation=0, ha="center")
                            ax_rec.set(xlabel=f"{rec['x']} ({rec.get('time_label', 'time')} aggregation)", ylabel=ylabel)
                    elif rec["kind"] == "Bar":
                        work = data[[rec["x"], rec["y"]]].dropna().groupby(rec["x"])[rec["y"]].agg(rec.get("agg", "Mean").lower()).sort_values(ascending=False).head(15).sort_values()
                        ax_rec.barh(work.index.astype(str), work.values, color="#CC79A7")
                        ax_rec.set(xlabel=f"{rec.get('agg', 'Mean')} {rec['y']}", ylabel=rec["x"])
                    elif rec["kind"] == "Scatter":
                        work = data[[rec["x"], rec["y"]]].dropna()
                        if len(work) > 3000:
                            work = work.sample(3000, random_state=42)
                        ax_rec.scatter(work[rec["x"]], work[rec["y"]], s=15, alpha=.35, color="#0072B2", edgecolors="none")
                        ax_rec.set(xlabel=rec["x"], ylabel=rec["y"])
                    elif rec["kind"] == "Histogram":
                        ax_rec.hist(data[rec["y"]].dropna(), bins=30, color="#E69F00", edgecolor="#222222", linewidth=.4)
                        ax_rec.set(xlabel=rec["y"], ylabel="Frequency")
                    elif rec["kind"] == "Box plot":
                        work = data[[rec["x"], rec["y"]]].dropna()
                        groups = work[rec["x"]].value_counts().head(8).index
                        vals = [work.loc[work[rec["x"]] == g, rec["y"]].to_numpy() for g in groups]
                        ax_rec.boxplot(vals, tick_labels=[str(g) for g in groups], showfliers=False)
                        ax_rec.set(xlabel=rec["x"], ylabel=rec["y"])
                        ax_rec.tick_params(axis="x", labelrotation=25)
                    elif rec["kind"] == "Heatmap":
                        corr = data[recommendation_numeric].corr(numeric_only=True)
                        im = ax_rec.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                        ax_rec.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
                        ax_rec.set_yticks(range(len(corr.index)), labels=corr.index)
                        fig_rec.colorbar(im, ax=ax_rec, fraction=.046, pad=.04, label="Correlation")
                    ax_rec.set_title(rec["title"])
                    ax_rec.grid(True, alpha=.2)
                    fig_rec.tight_layout()
                    st.pyplot(fig_rec, use_container_width=True)
                    plt.close(fig_rec)
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    st.info(f"Preview unavailable for this recommendation: {exc}")
                st.caption("This score is a rule-based ranking, not a validated probability or expert rating.")

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(data.head(100), use_container_width=True, height=320)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download uploaded CSV", data=upload.getvalue(),
                           file_name=Path(upload.name).name, mime="text/csv")
    with c2:
        st.download_button("Download summary statistics",
                           data=data.describe(include="all").transpose().to_csv().encode("utf-8"),
                           file_name="pubfig_summary_statistics.csv", mime="text/csv")

with tab_quality:
    st.subheader("Data quality snapshot")
    quality = pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing": data.isna().sum(),
        "missing_%": (data.isna().mean() * 100).round(2),
        "unique_values": data.nunique(dropna=True),
    })
    st.dataframe(quality, use_container_width=True)
    duplicates = int(data.duplicated().sum())
    st.write(f"**Duplicate rows:** {duplicates:,}")
    if data.isna().sum().sum():
        st.warning("Missing values detected. Choose columns carefully; some chart types may reject missing data.")
    else:
        st.success("No missing cells detected.")

with tab_build:
    left, right = st.columns([0.8, 2.2], gap="large")
    with left:
        chart_type = st.selectbox("Chart type", [
            "Line", "Area", "Bar", "Scatter", "Histogram",
            "Box plot", "Violin plot", "Heatmap", "ECDF"
        ])
        title = st.text_input("Figure title", value="")
        custom_xlabel = st.text_input("X-axis label (optional)", value="")
        custom_ylabel = st.text_input("Y-axis label (optional)", value="")
        show_grid = st.checkbox("Show grid", value=True)
        annotate = st.checkbox("Annotate values where practical", value=False)
        st.caption("Preflight checks flag technical concerns; they do not certify IEEE/ACM compliance.")

    fig = None
    notes = []
    try:
        if chart_type == "Heatmap":
            if len(numeric_cols) < 2:
                st.error("Heatmap requires at least two numeric columns.")
                st.stop()
            heat_mode = st.radio("Heatmap data", ["Correlation matrix", "Standardized observations"], horizontal=True)
            chosen = st.multiselect("Numeric fields", numeric_cols, default=numeric_cols[:min(8, len(numeric_cols))])
            if len(chosen) < 2:
                st.warning("Choose at least two numeric fields.")
                st.stop()
            if heat_mode == "Correlation matrix":
                mat = data[chosen].corr(numeric_only=True)
                fig, ax = heatmap(mat, title=title or "Correlation matrix", xlabel="Variable", ylabel="Variable",
                                  annot=annot and len(chosen) <= 15, cmap="coolwarm")
                notes.append("Correlation describes association, not causation.")
            else:
                max_rows = st.slider("Rows to display", 10, min(300, len(data)), min(60, len(data)))
                raw = data[chosen].head(max_rows)
                std = raw.std(ddof=0).replace(0, 1)
                mat = ((raw - raw.mean()) / std).fillna(0).reset_index(drop=True)
                fig, ax = heatmap(mat, title=title or "Standardized observations", xlabel="Variable", ylabel="Observation")
                notes.append("Values are standardized per selected column.")
        elif chart_type in ("Line", "Area"):
            if not numeric_cols:
                st.error("This chart requires a numeric field.")
                st.stop()
            x_options = list(dict.fromkeys(date_cols + categorical_cols + numeric_cols))
            x = st.selectbox("X-axis (ordered/time variable recommended)", x_options)
            y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            aggregation = st.selectbox("Aggregation", ["Mean", "Sum", "Median", "Count", "Keep rows"], index=0)
            if x == y:
                st.warning("Choose different X and Y fields.")
                st.stop()
            if aggregation != "Keep rows":
                fn = {"Mean":"mean", "Sum":"sum", "Median":"median", "Count":"count"}[aggregation]
                work = data[[x, y]].dropna().groupby(x, as_index=False, dropna=False)[y].agg(fn)
                try:
                    work = work.sort_values(x)
                except (TypeError, ValueError):
                    pass
            else:
                work = data[[x, y]].dropna().copy()
                if len(work) > 1000:
                    notes.append("More than 1,000 raw rows are displayed; aggregation may improve readability.")
            if chart_type == "Line":
                fig, ax = line_plot(work, x, y, title=title or f"{aggregation} {y} by {x}",
                                    xlabel=custom_xlabel or x, ylabel=custom_ylabel or f"{aggregation} {y}")
            else:
                from matplotlib.ticker import FuncFormatter
                fig, ax = plt.subplots(figsize=(6.5, 3.6))
                ax.fill_between(np.arange(len(work)), work[y].to_numpy(), alpha=.25, color="#D55E00")
                ax.plot(np.arange(len(work)), work[y].to_numpy(), color="#0072B2", linewidth=1.8)
                ax.set(title=title or f"{y} by {x}", xlabel=custom_xlabel or x, ylabel=custom_ylabel or y)
                if len(work) <= 20:
                    ax.set_xticks(np.arange(len(work)), labels=work[x].astype(str), rotation=25, ha="right")
                else:
                    ticks = np.linspace(0, len(work)-1, min(8, len(work)), dtype=int)
                    ax.set_xticks(ticks, labels=work[x].iloc[ticks].astype(str), rotation=25, ha="right")
                if "inr" in y.lower():
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M" if abs(v)>=1e6 else f"₹{v/1e3:.0f}k" if abs(v)>=1e3 else f"₹{v:.0f}"))
        elif chart_type == "Bar":
            if not numeric_cols:
                st.error("Bar chart requires a numeric field.")
                st.stop()
            x_options = categorical_cols + date_cols
            if not x_options:
                x_options = list(data.columns)
            x = st.selectbox("Category / grouping field", x_options)
            y = st.selectbox("Numeric value", numeric_cols)
            aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Count"], index=0)
            fn = {"Sum":"sum", "Mean":"mean", "Median":"median", "Count":"count"}[aggregation]
            grouped = data.groupby(x, dropna=False)[y].agg(fn).reset_index(name="_pubfig_value")
            if len(grouped) > 30:
                st.warning(f"{len(grouped)} groups found. Showing top 30 by {aggregation.lower()}.")
                grouped = grouped.nlargest(30, "_pubfig_value")
            fig, ax = bar_plot(grouped, x, "_pubfig_value", title=title or f"{aggregation} {y} by {x}",
                               xlabel=custom_xlabel or x, ylabel=custom_ylabel or f"{aggregation} {y}")
            if "inr" in y.lower():
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M" if abs(v)>=1e6 else f"₹{v/1e3:.0f}k" if abs(v)>=1e3 else f"₹{v:.0f}"))
        elif chart_type == "Scatter":
            if len(numeric_cols) < 2:
                st.error("Scatter plot requires at least two numeric fields.")
                st.stop()
            x = st.selectbox("X-axis", numeric_cols)
            y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            alpha = st.slider("Point transparency", 0.1, 1.0, 0.35, 0.05)
            marker_size = st.slider("Marker size", 8, 80, 20)
            if x == y:
                st.warning("Choose different fields.")
                st.stop()
            fig, ax = scatter_plot(data.dropna(subset=[x, y]), x, y,
                                   title=title or f"{y} vs {x}",
                                   xlabel=custom_xlabel or x, ylabel=custom_ylabel or y,
                                   alpha=alpha, marker_size=marker_size)
            if "inr" in x.lower():
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M" if abs(v)>=1e6 else f"₹{v/1e3:.0f}k" if abs(v)>=1e3 else f"₹{v:.0f}"))
            if "inr" in y.lower():
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M" if abs(v)>=1e6 else f"₹{v/1e3:.0f}k" if abs(v)>=1e3 else f"₹{v:.0f}"))
            if len(data) > 2000:
                notes.append("Dense scatter data may overplot. Consider lowering marker size/transparency or sampling.")
        elif chart_type in ("Histogram", "Box plot", "Violin plot", "ECDF"):
            if not numeric_cols:
                st.error(f"{chart_type} requires a numeric field.")
                st.stop()
            y = st.selectbox("Numeric field", numeric_cols)
            group_options = ["(No grouping)"] + categorical_cols
            group = st.selectbox("Group by (optional)", group_options)
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            series = data[y].dropna()
            if chart_type == "Histogram":
                bins = st.slider("Number of bins", 5, 100, 30)
                ax.hist(series, bins=bins, color="#CC79A7", edgecolor="#222222", linewidth=.5)
                ax.set(xlabel=custom_xlabel or y, ylabel=custom_ylabel or "Frequency")
            elif chart_type == "ECDF":
                vals = np.sort(series.to_numpy())
                probs = np.arange(1, len(vals)+1)/len(vals) if len(vals) else []
                ax.step(vals, probs, where="post", color="#CC79A7")
                ax.set(xlabel=custom_xlabel or y, ylabel=custom_ylabel or "Cumulative proportion", ylim=(0,1))
            else:
                if group != "(No grouping)":
                    top_groups = data[group].value_counts().head(8).index
                    subset = data[data[group].isin(top_groups)]
                    vals = [subset.loc[subset[group] == g, y].dropna().to_numpy() for g in top_groups]
                    labels = [str(g) for g in top_groups]
                    if chart_type == "Box plot":
                        ax.boxplot(vals, tick_labels=labels, showfliers=False)
                    else:
                        ax.violinplot(vals, showmedians=True, showextrema=True)
                        ax.set_xticks(np.arange(1, len(labels)+1), labels=labels)
                    ax.tick_params(axis="x", labelrotation=25)
                else:
                    if chart_type == "Box plot":
                        ax.boxplot(series.to_numpy(), vert=True, showfliers=False)
                    else:
                        ax.violinplot(series.to_numpy(), showmedians=True, showextrema=True)
                ax.set_ylabel(custom_ylabel or y)
            ax.set_title(title or f"{chart_type}: {y}")
            ax.grid(show_grid, alpha=.22)
        if fig is not None:
            ax.grid(show_grid, alpha=.22)
            if chart_type in ("Histogram", "Box plot", "Violin plot", "ECDF", "Area"):
                fig.tight_layout()
            preview, export_tab, checks_tab = st.tabs(["Figure preview", "Export", "Preflight"])
            with preview:
                st.pyplot(fig, use_container_width=True)
                if notes:
                    for note in notes:
                        st.warning(note)
            with export_tab:
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = save_publication_figure(fig, "pubfig_figure", tmpdir, dpi=export_dpi)
                    download_cols = st.columns(3)
                    mime = {"png":"image/png", "pdf":"application/pdf", "svg":"image/svg+xml"}
                    for col, (fmt, path) in zip(download_cols, result["files"].items()):
                        with open(path, "rb") as handle:
                            col.download_button(f"Download {fmt.upper()}", data=handle.read(),
                                                 file_name=f"pubfig_figure.{fmt}", mime=mime[fmt],
                                                 use_container_width=True)
            with checks_tab:
                problems = validate_figure(fig, min_width_in=3.25, min_height_in=2.0, min_dpi=0)
                if problems:
                    for problem in problems:
                        st.warning(problem)
                else:
                    st.success("Basic figure-size checks passed. This is not official IEEE/ACM compliance certification.")
                st.write({"figure_size_inches": [round(x, 2) for x in fig.get_size_inches()],
                          "export_dpi": export_dpi, "output_formats": ["PNG", "PDF", "SVG"],
                          "rows_used": int(len(data))})
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        st.error(f"Could not create this figure: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)
