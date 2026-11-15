"""Behaviour tests for the model and dataset analytics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from clicked_ads.analytics import (
    category_counts,
    coefficient_table,
    confidence_label,
    explain_row,
    feature_display_name,
    model_metrics,
    numeric_correlation,
    outcome_label,
    predict_frame,
    target_balance,
)
from clicked_ads.schema import NUMERIC_COLUMNS, TARGET_COLUMN


@pytest.fixture(scope="module")
def model_and_reference():
    rng = np.random.default_rng(42)
    n_rows = 200

    reference = pd.DataFrame(
        {
            "Daily Time Spent on Site": rng.normal(60, 15, n_rows),
            "Age": rng.normal(40, 10, n_rows),
            "Area Income": rng.normal(50_000_000, 15_000_000, n_rows),
            "Daily Internet Usage": rng.normal(180, 40, n_rows),
            "Gender": rng.integers(0, 2, n_rows),
            "Category_Fashion": rng.integers(0, 2, n_rows),
        }
    )
    reference[TARGET_COLUMN] = np.tile([0, 1], n_rows // 2)

    features = reference[list(reference.columns[:-1])]
    model = LogisticRegression(max_iter=1000).fit(features, reference[TARGET_COLUMN])

    return model, reference


def test_feature_display_name():
    assert feature_display_name("Category_Fashion") == "Category · Fashion"
    assert feature_display_name("Age") == "Age"


def test_outcome_label():
    assert outcome_label(0) == "Not clicked"
    assert outcome_label(1) == "Clicked"


def test_confidence_label():
    assert confidence_label(0.95) == "High"
    assert confidence_label(0.05) == "High"
    assert confidence_label(0.80) == "Moderate"
    assert confidence_label(0.60) == "Low"


def test_predict_frame(model_and_reference):
    model, reference = model_and_reference
    model_input = reference[list(model.feature_names_in_)]

    predictions, probabilities = predict_frame(model, model_input)

    assert len(predictions) == len(reference)
    assert set(np.unique(predictions)) <= {0, 1}
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_metrics(model_and_reference):
    model, reference = model_and_reference

    metrics = model_metrics(model, reference)

    assert metrics["n_test"] == 60
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert metrics["confusion_matrix"].sum() == metrics["n_test"]


def test_coefficient_table_is_sorted_by_absolute_effect(model_and_reference):
    model, _reference = model_and_reference

    table = coefficient_table(model)

    assert list(table.columns) == [
        "feature",
        "coefficient",
        "abs_coefficient",
        "display",
    ]
    assert len(table) == len(model.feature_names_in_)
    assert table["abs_coefficient"].is_monotonic_decreasing


def test_explain_row_decomposes_the_linear_prediction(model_and_reference):
    model, reference = model_and_reference
    model_input = reference[list(model.feature_names_in_)].head(1)

    explanations = explain_row(model, model_input, top_n=3)

    assert len(explanations) == 3
    assert set(explanations["direction"]) <= {
        "Increases click odds",
        "Decreases click odds",
    }
    np.testing.assert_allclose(
        explanations["contribution"],
        explanations["coefficient"] * explanations["input"],
    )


def test_target_balance(model_and_reference):
    _model, reference = model_and_reference

    balance = target_balance(reference)

    assert list(balance.columns) == ["Outcome", "Customers"]
    assert balance["Customers"].sum() == len(reference)
    assert set(balance["Outcome"]) == {"Not clicked", "Clicked"}


def test_category_counts():
    training = pd.DataFrame({"Category": ["Fashion", "Fashion", "Food"]})

    counts = category_counts(training)

    assert list(counts.columns) == ["Category", "Customers"]
    assert counts.set_index("Category").loc["Fashion", "Customers"] == 2


def test_numeric_correlation_has_expected_shape(model_and_reference):
    _model, reference = model_and_reference

    correlation = numeric_correlation(reference)

    expected_columns = NUMERIC_COLUMNS + [TARGET_COLUMN]
    assert list(correlation.columns) == expected_columns
    assert correlation.shape == (5, 5)
    np.testing.assert_allclose(np.diag(correlation), 1.0)
