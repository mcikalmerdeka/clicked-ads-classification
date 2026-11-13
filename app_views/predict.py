"""Individual customer prediction view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_analytics import confidence_label, explain_row, outcome_label
from app_ui import contribution_chart, fmt_percent, probability_progress

GENDERS = ["Perempuan", "Laki-Laki"]

TIME_KEY = "predict_time"
AGE_KEY = "predict_age"
INCOME_KEY = "predict_income"
USAGE_KEY = "predict_usage"
GENDER_KEY = "predict_gender"
CATEGORY_KEY = "predict_category"
RESULT_KEY = "predict_result"

EXAMPLE_CLICKED = {
    "time": 33.0,
    "age": 55.0,
    "income": 97_975_500.0,
    "usage": 105.0,
    "gender": "Perempuan",
    "category": "Fashion",
}

EXAMPLE_NOT_CLICKED = {
    "time": 92.0,
    "age": 19.0,
    "income": 556_393_600.0,
    "usage": 267.0,
    "gender": "Laki-Laki",
    "category": "Otomotif",
}


def _load_example(example: dict) -> None:
    st.session_state[TIME_KEY] = example["time"]
    st.session_state[AGE_KEY] = example["age"]
    st.session_state[INCOME_KEY] = example["income"]
    st.session_state[USAGE_KEY] = example["usage"]
    st.session_state[GENDER_KEY] = example["gender"]
    st.session_state[CATEGORY_KEY] = example["category"]
    st.session_state.pop(RESULT_KEY, None)


def _bounds(data: pd.DataFrame, column: str) -> tuple[float, float, float]:
    series = data[column]
    return float(series.min()), float(series.max()), float(series.mean())


def _render_result(model, result: dict) -> None:
    st.markdown("### Prediction")

    with st.container(horizontal=True):
        st.metric(
            "Predicted outcome",
            outcome_label(result["prediction"]),
            border=True,
        )
        st.metric(
            "Click probability",
            fmt_percent(result["probability"]),
            border=True,
        )
        st.metric(
            "Confidence",
            confidence_label(result["probability"]),
            border=True,
        )

    probability_progress(result["probability"])

    inputs = result["inputs"]
    st.caption(
        f"Scored {inputs['Gender']} · {inputs['Category']} · age "
        f"{inputs['Age']:.0f} · area income {inputs['Area Income']:,.0f}"
    )

    with st.container(border=True):
        st.markdown("**Why this prediction**")
        explanations = explain_row(model, result["model_input"])
        st.altair_chart(contribution_chart(explanations), width="stretch")
        st.caption(
            "Each bar is the feature's coefficient multiplied by its standardized "
            "input. Positive values push toward a click; negative values push away."
        )

    with st.expander("Model input (14 features)"):
        st.dataframe(result["model_input"], hide_index=True)


def render_predict_tab(model, pipeline, training: pd.DataFrame) -> None:
    st.markdown(
        "Score one customer from their behavior and audience profile. "
        "Only the fields the model uses are shown."
    )

    categories = sorted(training["Category"].dropna().unique().tolist())

    with st.container(horizontal=True):
        st.button(
            "Load clicked example",
            icon=":material/check_circle:",
            type="tertiary",
            on_click=_load_example,
            args=(EXAMPLE_CLICKED,),
        )
        st.button(
            "Load not-clicked example",
            icon=":material/cancel:",
            type="tertiary",
            on_click=_load_example,
            args=(EXAMPLE_NOT_CLICKED,),
        )

    time_min, time_max, time_mean = _bounds(training, "Daily Time Spent on Site")
    usage_min, usage_max, usage_mean = _bounds(training, "Daily Internet Usage")
    age_min, age_max, age_mean = _bounds(training, "Age")
    income_min, income_max, income_mean = _bounds(training, "Area Income")

    with st.form("predict_form", border=True):
        st.markdown("**Customer profile**")
        st.caption(
            "Visit time, city, and province are not used by the current model, "
            "so they are not requested here."
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Behavior**")
            time_on_site = st.number_input(
                "Daily time spent on site (minutes)",
                min_value=time_min,
                max_value=time_max,
                value=float(round(time_mean)),
                step=1.0,
                format="%.0f",
                help="Average minutes per day spent on the site.",
                key=TIME_KEY,
            )
            internet_usage = st.number_input(
                "Daily internet usage (minutes)",
                min_value=usage_min,
                max_value=usage_max,
                value=float(round(usage_mean)),
                step=1.0,
                format="%.0f",
                help="Average minutes per day spent online.",
                key=USAGE_KEY,
            )

        with right:
            st.markdown("**Audience**")
            age = st.number_input(
                "Age",
                min_value=age_min,
                max_value=age_max,
                value=float(round(age_mean)),
                step=1.0,
                format="%.0f",
                help="Customer age in years.",
                key=AGE_KEY,
            )
            area_income = st.number_input(
                "Area income (IDR)",
                min_value=income_min,
                max_value=income_max,
                value=float(round(income_mean / 1_000_000) * 1_000_000),
                step=1_000_000.0,
                format="%.0f",
                help="Average income of the customer's area.",
                key=INCOME_KEY,
            )
            gender = st.selectbox(
                "Gender",
                GENDERS,
                help="Gender recorded in the dataset.",
                key=GENDER_KEY,
            )
            category = st.selectbox(
                "Ad category",
                categories,
                help="Category of the advertisement shown.",
                key=CATEGORY_KEY,
            )

        submitted = st.form_submit_button(
            "Predict click probability",
            type="primary",
            icon=":material/bolt:",
        )

    if submitted:
        raw = pd.DataFrame(
            [
                {
                    "Daily Time Spent on Site": float(time_on_site),
                    "Age": float(age),
                    "Area Income": float(area_income),
                    "Daily Internet Usage": float(internet_usage),
                    "Gender": gender,
                    "Category": category,
                }
            ]
        )
        try:
            model_input = pipeline.transform(raw)
            probability = float(model.predict_proba(model_input)[0, 1])
        except Exception as exc:  # surface the real reason instead of a blank page
            st.error(f"Could not score this customer: {exc}")
        else:
            st.session_state[RESULT_KEY] = {
                "prediction": int(probability >= 0.5),
                "probability": probability,
                "model_input": model_input,
                "inputs": raw.iloc[0].to_dict(),
            }

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.caption("Submit the form to score a customer.")
        return

    _render_result(model, result)
