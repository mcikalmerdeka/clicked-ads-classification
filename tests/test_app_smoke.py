"""Smoke test that the Streamlit app boots without exceptions."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "main.py"


def test_app_boots_without_exception():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()

    assert not app.exception
