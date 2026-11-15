"""Shared presentation helpers for the Clicked Ads app."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ACCENT = "#0D9488"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
INK = "#1C1917"


def fmt_percent(value: float) -> str:
    """Format a probability without rounding extreme values to 0% or 100%."""
    if value >= 0.9995:
        return ">99.9%"
    if value <= 0.0005:
        return "<0.1%"
    return f"{value:.1%}"


def probability_progress(click_probability: float) -> None:
    st.progress(
        min(max(float(click_probability), 0.0), 1.0),
        text=f"Click probability · {fmt_percent(click_probability)}",
    )


def contribution_chart(contributions: pd.DataFrame) -> alt.Chart:
    """Horizontal bars showing one prediction's top drivers."""
    order = contributions["display"].tolist()
    bars = (
        alt.Chart(contributions)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("contribution:Q", title="Effect on click log-odds"),
            y=alt.Y("display:N", title=None, sort=order),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Increases click odds", "Decreases click odds"],
                    range=[POSITIVE, NEGATIVE],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip("input:Q", title="Model input", format=".3f"),
                alt.Tooltip("contribution:Q", title="Contribution", format="+.3f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#A8A29E", strokeWidth=1)
        .encode(x="zero:Q")
    )
    return (bars + rule).properties(height=alt.Step(30))


def coefficient_chart(table: pd.DataFrame) -> alt.Chart:
    """Global standardized coefficients for the model."""
    order = table["display"].tolist()
    return (
        alt.Chart(table)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("coefficient:Q", title="Coefficient (log-odds per SD)"),
            y=alt.Y("display:N", title=None, sort=order),
            color=alt.condition(
                "datum.coefficient > 0", alt.value(POSITIVE), alt.value(NEGATIVE)
            ),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip("coefficient:Q", title="Coefficient", format="+.3f"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def confusion_chart(matrix: np.ndarray) -> alt.Chart:
    """2x2 confusion matrix with customer counts."""
    rows = ["Actual · Not clicked", "Actual · Clicked"]
    columns = ["Predicted · Not clicked", "Predicted · Clicked"]
    frame = (
        pd.DataFrame(matrix, index=rows, columns=columns)
        .rename_axis("actual")
        .reset_index()
        .melt(id_vars="actual", var_name="predicted", value_name="customers")
    )

    base = alt.Chart(frame).mark_rect(cornerRadius=4).encode(
        x=alt.X("predicted:N", title=None, sort=columns),
        y=alt.Y("actual:N", title=None, sort=rows),
        color=alt.Color(
            "customers:Q",
            scale=alt.Scale(range=["#F0FDFA", "#99F6E4", "#5EEAD4"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("customers:Q", title="Customers"),
        ],
    )
    text = base.mark_text(fontSize=20, fontWeight="bold", color="#134E4A").encode(
        text="customers:Q"
    )
    return (base + text).properties(height=220)


def correlation_chart(correlation: pd.DataFrame) -> alt.Chart:
    """Spearman correlation heatmap."""
    order = correlation.columns.tolist()
    frame = (
        correlation.rename_axis("row")
        .reset_index()
        .melt(id_vars="row", var_name="column", value_name="rho")
    )

    base = alt.Chart(frame).mark_rect().encode(
        x=alt.X("column:N", title=None, sort=order),
        y=alt.Y("row:N", title=None, sort=order),
        color=alt.Color(
            "rho:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
            title="Spearman ρ",
        ),
        tooltip=[
            alt.Tooltip("row:N", title="Feature"),
            alt.Tooltip("column:N", title="Against"),
            alt.Tooltip("rho:Q", title="ρ", format="+.2f"),
        ],
    )
    text = base.mark_text(fontSize=10).encode(
        text=alt.Text("rho:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.rho) > 0.55", alt.value("#FFFFFF"), alt.value(INK)
        ),
    )
    return (base + text).properties(height=alt.Step(42))


def distribution_chart(reference: pd.DataFrame, column: str) -> alt.Chart:
    """Histogram of one continuous feature."""
    return (
        alt.Chart(reference)
        .mark_bar(color=ACCENT, opacity=0.85, cornerRadius=2)
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=28), title=column),
            y=alt.Y("count():Q", title="Customers"),
            tooltip=[alt.Tooltip("count():Q", title="Customers")],
        )
        .properties(height=260)
    )


def target_chart(balance: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(balance)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("Outcome:N", title=None),
            y=alt.Y("Customers:Q", title="Customers"),
            tooltip=["Outcome:N", "Customers:Q"],
        )
        .properties(height=240)
    )


def category_chart(counts: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadius=3, color=ACCENT, opacity=0.85)
        .encode(
            x=alt.X("Customers:Q", title="Customers"),
            y=alt.Y("Category:N", title=None, sort="-x"),
            tooltip=["Category:N", "Customers:Q"],
        )
        .properties(height=300)
    )
