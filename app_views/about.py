"""About and model card view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_pipeline import (
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    get_feature_definitions,
)
from utils import check_data_information


def _feature_table() -> pd.DataFrame:
    used = set(NUMERIC_COLUMNS) | {"Gender", "Category"}
    rows = []

    for key, meta in get_feature_definitions().items():
        name = key.replace("_", " ")
        if name == TARGET_COLUMN:
            status = "Target"
        elif name in used:
            status = "Yes"
        else:
            status = "No"

        rows.append(
            {
                "Feature": name,
                "Description": meta["description"],
                "Type": f"{meta['data_type']} · {meta['specific_type']}",
                "Used by model": status,
            }
        )

    return pd.DataFrame(rows)


def render_about_tab(reference: pd.DataFrame, training: pd.DataFrame) -> None:
    st.markdown(
        "A digital marketing team wants to spend ad budget on the customers most "
        "likely to click. This app scores individual customers and batches, then "
        "shows which factors drove each decision."
    )

    st.markdown("**Reported impact from the original analysis**")
    with st.container(horizontal=True):
        st.metric("Click-through rate", "99.8%", "from 50%", border=True)
        st.metric("ROAS", "2.43 IDR", "from 1.25", border=True)
        st.metric("Profit growth", "+581.8%", border=True)
    st.caption(
        "These figures come from the original business analysis and are not "
        "recomputed by this app."
    )

    with st.container(border=True):
        st.markdown("**Model card**")
        st.markdown(
            f"- **Algorithm** · Logistic regression (binary classification)\n"
            f"- **Target** · {TARGET_COLUMN}\n"
            f"- **Training data** · {len(training):,} raw customers, "
            f"{len(reference):,} after outlier filtering\n"
            f"- **Features** · 4 continuous, 1 ordinal (Gender), "
            f"9 one-hot (Category)\n"
            f"- **Artifact** · `models/tuned_logistic_regression_model.joblib`\n"
            f"- **Evaluation** · held-out 30% split, random_state=42"
        )

    st.subheader("Data dictionary")
    st.caption(
        "Visit time, city, and province exist in the source data but are not "
        "used by the deployed model."
    )
    st.dataframe(
        _feature_table(),
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Used by model": st.column_config.TextColumn(width="small"),
        },
    )

    with st.expander("Training data preview"):
        st.dataframe(training.head(10), hide_index=True)
        st.markdown("**Data information**")
        st.dataframe(
            check_data_information(training, training.columns),
            hide_index=True,
        )

    st.subheader("Limitations")
    st.markdown(
        "- The model is trained on a 1,000-customer sample; it may not represent "
        "every market or campaign.\n"
        "- Predictions reflect correlation, not causation. They should support "
        "campaign decisions, not replace them.\n"
        "- Only six raw fields influence the score. Visit time and geography are "
        "collected in the source data but unused.\n"
        "- Probabilities are model estimates; recalibrate and retrain as new "
        "campaign data arrives.\n"
        "- The artifact was trained with scikit-learn 1.4.0 and is loaded under "
        "1.8.0, which raises a compatibility warning at startup."
    )

    st.subheader("Author")
    st.markdown(
        "Developed by **Muhammad Cikal Merdeka** · Data Analyst / Data Scientist\n\n"
        "[Email](mailto:mcikalmerdeka@gmail.com) · "
        "[GitHub](https://github.com/mcikalmerdeka) · "
        "[LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)"
    )

    st.caption(
        "Light and dark themes are available from the app menu "
        "(⋮ → Settings → Theme)."
    )
