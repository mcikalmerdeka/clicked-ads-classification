"""Clicked Ads Analyzer — Streamlit entry point."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from app_pipeline import ClickedAdsPipeline, initial_data_transform
from app_views import (
    render_about_tab,
    render_batch_tab,
    render_insights_tab,
    render_predict_tab,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "tuned_logistic_regression_model.joblib"

# The artifact was pickled with scikit-learn 1.4.0 while the app runs 1.8.0.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Clicked Ads Analyzer",
    page_icon=":material/ads_click:",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Load the model, reference data, and fitted pipeline once per session."""
    model = joblib.load(MODEL_PATH)
    reference = pd.read_csv(DATA_DIR / "df_model.csv")
    training = initial_data_transform(
        pd.read_csv(DATA_DIR / "clicked_ads_dataset.csv", index_col=0)
    )
    pipeline = ClickedAdsPipeline.from_project_data(
        DATA_DIR / "clicked_ads_dataset.csv",
        DATA_DIR / "df_model.csv",
    )
    return model, reference, training, pipeline


if not MODEL_PATH.exists():
    st.error(f"Model artifact not found at {MODEL_PATH}.")
    st.stop()

model, reference, training, pipeline = load_artifacts()

st.title("Clicked Ads Analyzer")
st.caption(
    "Predict who will click a digital ad, score batches, and see what drives "
    "the model."
)

with st.container(horizontal=True):
    st.badge(
        "Logistic regression",
        icon=":material/model_training:",
        color="blue",
    )
    st.badge(
        f"{len(model.feature_names_in_)} features",
        icon=":material/tune:",
        color="gray",
    )
    st.badge(
        f"{len(training):,} training rows",
        icon=":material/database:",
        color="gray",
    )

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["Predict", "Batch", "Insights", "About"]
)

with tab_predict:
    render_predict_tab(model, pipeline, training)

with tab_batch:
    render_batch_tab(model, pipeline, training)

with tab_insights:
    render_insights_tab(model, reference, training)

with tab_about:
    render_about_tab(reference, training)
