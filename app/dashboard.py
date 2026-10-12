"""Entrypoint shim for dashboard when executed via app/dashboard.py."""

from pathlib import Path
import runpy

DASHBOARD_FILE = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"

if __name__ == "__main__" or "streamlit" in __name__:
    runpy.run_path(str(DASHBOARD_FILE), run_name="__main__")
