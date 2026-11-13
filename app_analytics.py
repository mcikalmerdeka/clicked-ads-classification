"""Model and dataset analytics for the Clicked Ads app.

Pure functions with no Streamlit dependency so they stay testable and reusable
by both the app views and any future report.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app_pipeline import NUMERIC_COLUMNS, TARGET_COLUMN

OUTCOME_LABELS = {0: "Not clicked", 1: "Clicked"}


def feature_display_name(feature: str) -> str:
    """Turn a model column name into a label that reads well in charts."""
    if feature.startswith("Category_"):
        return f"Category · {feature.removeprefix('Category_')}"
    return feature


def outcome_label(prediction: int) -> str:
    return OUTCOME_LABELS[int(prediction)]


def predict_frame(model, model_input: pd.DataFrame):
    """Return class predictions and click probabilities for a model matrix."""
    probabilities = model.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


def confidence_label(click_probability: float) -> str:
    """How far the prediction sits from a coin flip."""
    confidence = max(click_probability, 1.0 - click_probability)
    if confidence >= 0.9:
        return "High"
    if confidence >= 0.7:
        return "Moderate"
    return "Low"


def model_metrics(model, reference: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the persisted model on the notebook's held-out 30% split.

    The split reproduces ``train_test_split(..., test_size=0.3,
    random_state=42)`` from the modelling notebook, so these metrics describe
    rows the model was not fitted on.
    """
    features = reference[list(model.feature_names_in_)]
    target = reference[TARGET_COLUMN]

    _X_train, X_test, _y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    # The reference frame stores unscaled values. Recreate the notebook's
    # scaler (fitted on the training split) before evaluating the model.
    scaler = StandardScaler().fit(_X_train[NUMERIC_COLUMNS])
    X_test = X_test.copy()
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    return {
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def coefficient_table(model) -> pd.DataFrame:
    """Global feature importance as standardized logistic coefficients."""
    table = pd.DataFrame(
        {
            "feature": list(model.feature_names_in_),
            "coefficient": model.coef_[0],
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    table["display"] = table["feature"].map(feature_display_name)
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def explain_row(model, model_input: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Per-prediction contribution: coefficient multiplied by model input.

    Because the model is linear, this decomposes the click log-odds exactly.
    Positive contributions push the prediction toward "Clicked".
    """
    row = model_input[list(model.feature_names_in_)].iloc[0]

    contributions = pd.DataFrame(
        {
            "feature": list(model.feature_names_in_),
            "input": row.to_numpy(dtype=float),
            "coefficient": model.coef_[0],
        }
    )
    contributions["contribution"] = (
        contributions["coefficient"] * contributions["input"]
    )
    contributions["display"] = contributions["feature"].map(feature_display_name)
    contributions["direction"] = np.where(
        contributions["contribution"] >= 0,
        "Increases click odds",
        "Decreases click odds",
    )

    order = contributions["contribution"].abs().sort_values(ascending=False).index
    return contributions.loc[order].head(top_n).reset_index(drop=True)


def target_balance(reference: pd.DataFrame) -> pd.DataFrame:
    """Customer counts per target class."""
    counts = reference[TARGET_COLUMN].map(OUTCOME_LABELS).value_counts()
    return counts.rename_axis("Outcome").reset_index(name="Customers")


def category_counts(training: pd.DataFrame) -> pd.DataFrame:
    """Ad category counts from the raw training data."""
    counts = training["Category"].value_counts()
    return counts.rename_axis("Category").reset_index(name="Customers")


def numeric_correlation(reference: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation across the continuous features and the target."""
    columns = NUMERIC_COLUMNS + [TARGET_COLUMN]
    return reference[columns].corr(method="spearman")
