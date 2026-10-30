# PubFig deployment checklist

This package is a local prototype; public deployment has not yet been performed.

## Local run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,demo]"
python -m pytest -v
python -m streamlit run webapp/app.py
```

Before public hosting, verify all chart types and exports in a clean environment and browser; add upload limits, pinned dependencies, security checks, license/attribution, privacy notice, and feedback route.
