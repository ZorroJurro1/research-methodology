# PubFig Phase 1 validation report

Run environment: Python 3.13.5; matplotlib, numpy, pandas, pytest and Pillow were available in the execution environment.

## Executed checks

- Automated test suite: **8 passed**.
- Python syntax compilation: passed for package modules and the Streamlit app.
- Deterministic demo script: executed and generated line, bar, scatter and heatmap examples.
- Each demo exports PNG at requested 600 DPI plus PDF and SVG vector-format files.
- PNG metadata and image dimensions were checked by automated tests.
- Input validation tests cover missing columns, non-numeric/missing values, empty data, and invalid export formats.
- Heatmap annotation guard tested for matrices larger than 400 cells.

## Observed output

All four demo exports completed. No configured preflight warnings were reported after the validator was adjusted to avoid treating a heatmap colorbar as a missing axis label. The demo data are synthetic and are for software demonstration only.

## Not verified / outstanding

- A package installation attempt with `pip install -e . --no-deps` failed because the environment could not resolve package indexes to fetch build dependencies. Therefore a clean isolated install is **not yet verified**.
- Streamlit is not installed in this environment. The web app passed Python syntax compilation but was not launched interactively.
- The app has not been deployed to a public hosting service.
- No independent visual accessibility review, color-vision simulation, grayscale contrast test, user study, or venue-specific compliance audit has been performed.
- No performance benchmark has been run.

## Interpretation

This validates core prototype behavior and export creation in the current environment. It does not establish research novelty, superiority over Matplotlib, or formal IEEE/ACM compliance.

## UI usability fix (2026-10-19)
- Scatter plot API now supports `alpha` and `marker_size`; points remain unconnected.
- Streamlit UI offers aggregation for line and bar charts (`sum`, `mean`, `median`, `count`).
- Line chart UI warns when users choose raw rows with many observations.
- Heatmap defaults to a numeric correlation matrix rather than plotting thousands of raw rows; raw heatmaps are capped and standardized for display.
- Streamlit interface includes CSV preview and clearer chart-selection guidance.
- Verification after changes: `MPLBACKEND=Agg python -m pytest -q` => **10 passed**.
- Not yet verified: running the Streamlit interface end-to-end in a browser or deploying it publicly.
