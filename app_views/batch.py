"""Batch scoring view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_analytics import (
    confidence_label,
    outcome_label,
    predict_frame,
)
from app_pipeline import NUMERIC_COLUMNS, initial_data_transform

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REQUIRED_COLUMNS = NUMERIC_COLUMNS + ["Gender", "Category"]
GENDERS = ["Perempuan", "Laki-Laki"]

DISPLAY_COLUMNS = [
    "Source row",
    "Gender",
    "Category",
    "Age",
    "Daily Time Spent on Site",
    "Daily Internet Usage",
    "Area Income",
    "Predicted outcome",
    "Click probability (%)",
    "Confidence",
]


def _drop_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove index columns produced by ``to_csv(index=True)`` exports."""
    return frame.loc[
        :, [column for column in frame.columns if not str(column).startswith("Unnamed:")]
    ]


def _score(model, pipeline, raw: pd.DataFrame):
    cleaned = pipeline.clean(raw)
    if cleaned.empty:
        return cleaned, None, None

    model_input = pipeline.scale(pipeline.encode(cleaned))
    predictions, probabilities = predict_frame(model, model_input)
    return cleaned, predictions, probabilities


def _render_results(raw: pd.DataFrame, cleaned: pd.DataFrame, predictions, probabilities) -> None:
    results = raw.loc[cleaned.index].copy()
    results.insert(0, "Source row", cleaned.index + 1)
    results["Predicted outcome"] = [outcome_label(value) for value in predictions]
    results["Click probability"] = probabilities
    results["Click probability (%)"] = probabilities * 100
    results["Confidence"] = [confidence_label(value) for value in probabilities]

    with st.container(horizontal=True):
        st.metric("Rows scored", f"{len(results):,}", border=True)
        st.metric("Predicted clicks", f"{int(predictions.sum()):,}", border=True)
        st.metric(
            "Average click probability",
            f"{probabilities.mean():.1%}",
            border=True,
        )

    dropped = len(raw) - len(cleaned)
    if dropped:
        st.caption(
            f"{dropped:,} row(s) were removed as area-income outliers; "
            f"{len(results):,} scored."
        )

    st.dataframe(
        results[DISPLAY_COLUMNS],
        hide_index=True,
        column_config={
            "Source row": st.column_config.NumberColumn("Row", format="%d"),
            "Area Income": st.column_config.NumberColumn("Area income", format="%d"),
            "Click probability (%)": st.column_config.ProgressColumn(
                "Click probability",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="clicked_ads_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    with st.expander("Input preview"):
        st.dataframe(raw.head(20), hide_index=True)


def render_batch_tab(model, pipeline, training: pd.DataFrame) -> None:
    st.markdown(
        "Upload a customer CSV and score every row in one pass. "
        "Results include click probability, a predicted outcome, and a download."
    )

    categories = sorted(training["Category"].dropna().unique().tolist())

    source = st.segmented_control(
        "Data source",
        ["Upload CSV", "Example data"],
        default="Upload CSV",
        key="batch_source",
    )

    raw = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Customer CSV",
            type="csv",
            help="Same columns as the training data. The target column is optional.",
        )
        if uploaded is not None:
            try:
                raw = _drop_unnamed(
                    pd.read_csv(uploaded).reset_index(drop=True)
                )
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")
                return
    elif source == "Example data":
        raw = _drop_unnamed(
            pd.read_csv(DATA_DIR / "batch_example.csv").reset_index(drop=True)
        )
        st.caption("Using the five-row example from data/batch_example.csv.")

    if raw is None:
        with st.expander("Expected columns"):
            st.markdown(
                "- Daily Time Spent on Site\n"
                "- Age\n"
                "- Area Income\n"
                "- Daily Internet Usage\n"
                "- Gender\n"
                "- Category\n\n"
                "Visit Time, City, and Province are accepted but not used by the "
                "current model."
            )
        return

    if raw.empty:
        st.error("The uploaded file has no rows.")
        return

    try:
        raw = initial_data_transform(raw)
    except Exception as exc:
        st.error(f"Could not parse the input: {exc}")
        return

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    normalized_gender = raw["Gender"].astype("string").str.title()
    unknown_genders = sorted(set(normalized_gender.dropna()) - set(GENDERS))
    unknown_categories = sorted(set(raw["Category"].dropna()) - set(categories))
    if unknown_genders:
        st.warning(
            "Unrecognized gender values: "
            + ", ".join(map(str, unknown_genders))
            + ". They are treated as unknown by the encoder."
        )
    if unknown_categories:
        st.warning(
            "Unrecognized ad categories: "
            + ", ".join(map(str, unknown_categories))
            + ". They fall back to the baseline category."
        )

    try:
        cleaned, predictions, probabilities = _score(model, pipeline, raw)
    except Exception as exc:
        st.error(f"Could not score the batch: {exc}")
        return

    if cleaned.empty:
        st.error("No rows remain after cleaning and outlier filtering.")
        return

    _render_results(raw, cleaned, predictions, probabilities)
