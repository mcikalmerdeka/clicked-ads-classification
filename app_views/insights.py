"""Model and dataset insights view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_analytics import (
    category_counts,
    coefficient_table,
    model_metrics,
    numeric_correlation,
    target_balance,
)
from app_pipeline import NUMERIC_COLUMNS
from app_ui import (
    category_chart,
    coefficient_chart,
    confusion_chart,
    correlation_chart,
    distribution_chart,
    target_chart,
)


@st.cache_data(show_spinner=False)
def _cached_metrics(_model, reference: pd.DataFrame):
    return model_metrics(_model, reference)


def render_insights_tab(model, reference: pd.DataFrame, training: pd.DataFrame) -> None:
    st.markdown(
        "How the model performs, what drives its decisions, and what the "
        "training data looks like."
    )

    metrics = _cached_metrics(model, reference)

    with st.container(horizontal=True):
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}", border=True)
        st.metric("Precision", f"{metrics['precision']:.1%}", border=True)
        st.metric("Recall", f"{metrics['recall']:.1%}", border=True)
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", border=True)

    st.caption(
        f"Evaluated on the notebook's held-out 30% split "
        f"({metrics['n_test']:,} customers, random_state=42)."
    )

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Confusion matrix**")
            st.altair_chart(
                confusion_chart(metrics["confusion_matrix"]), width="stretch"
            )
            st.caption("Rows are actual outcomes; columns are predictions.")

    with right:
        with st.container(border=True):
            st.markdown("**What moves the prediction**")
            st.altair_chart(
                coefficient_chart(coefficient_table(model)), width="stretch"
            )
            st.caption(
                "Standardized logistic coefficients. Positive values increase "
                "click odds; bar length is effect size."
            )

    st.subheader("Dataset overview")
    st.caption(
        f"Raw training data: {len(training):,} customers. "
        f"Reference frame after outlier filtering: {len(reference):,}."
    )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Target balance**")
            st.altair_chart(
                target_chart(target_balance(reference)), width="stretch"
            )

    with right:
        with st.container(border=True):
            st.markdown("**Ad categories**")
            st.altair_chart(
                category_chart(category_counts(training)), width="stretch"
            )

    feature = st.segmented_control(
        "Feature distribution",
        NUMERIC_COLUMNS,
        default=NUMERIC_COLUMNS[0],
    )
    if feature:
        with st.container(border=True):
            st.markdown(f"**Distribution · {feature}**")
            st.altair_chart(
                distribution_chart(reference, feature), width="stretch"
            )

    with st.container(border=True):
        st.markdown("**Correlation with the target**")
        st.altair_chart(
            correlation_chart(numeric_correlation(reference)), width="stretch"
        )
        st.caption(
            "Spearman correlation across the continuous features and the target. "
            "Correlation is not causation."
        )
