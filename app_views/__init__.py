"""Streamlit views for the Clicked Ads Analyzer."""

from .about import render_about_tab
from .batch import render_batch_tab
from .insights import render_insights_tab
from .predict import render_predict_tab

__all__ = [
    "render_about_tab",
    "render_batch_tab",
    "render_insights_tab",
    "render_predict_tab",
]
