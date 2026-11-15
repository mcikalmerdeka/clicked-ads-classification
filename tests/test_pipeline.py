"""Behaviour tests for the Clicked Ads preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from clicked_ads.config import REFERENCE_DATA_PATH, TRAINING_DATA_PATH
from clicked_ads.pipeline import ClickedAdsPipeline, initial_data_transform
from clicked_ads.schema import NUMERIC_COLUMNS, TARGET_COLUMN


@pytest.fixture(scope="module")
def pipeline() -> ClickedAdsPipeline:
    return ClickedAdsPipeline.from_project_data(
        TRAINING_DATA_PATH, REFERENCE_DATA_PATH
    )


def _raw_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Daily Time Spent on Site": 60.0,
                "Age": 35.0,
                "Area Income": 50_000_000.0,
                "Daily Internet Usage": 180.0,
                "Gender": "Perempuan",
                "Category": "Fashion",
            }
        ]
    )


def test_initial_data_transform_renames_and_moves_target_last():
    raw = pd.DataFrame(
        [
            {
                "Male": "Laki-Laki",
                "Timestamp": "2024-01-01",
                "city": "Jakarta",
                "province": "DKI Jakarta",
                "category": "Fashion",
                "Age": 30,
                TARGET_COLUMN: 1,
            }
        ]
    )

    transformed = initial_data_transform(raw)

    assert {"Gender", "Visit Time", "City", "Province", "Category"} <= set(
        transformed.columns
    )
    assert transformed.columns[-1] == TARGET_COLUMN
    assert pd.api.types.is_datetime64_any_dtype(transformed["Visit Time"])


def test_transform_returns_the_model_input_columns(pipeline: ClickedAdsPipeline):
    model_input = pipeline.transform(_raw_row())

    assert list(model_input.columns) == pipeline.feature_columns
    assert model_input.shape == (1, len(pipeline.feature_columns))
    assert not model_input.isna().any().any()
    assert model_input[NUMERIC_COLUMNS].to_numpy().dtype.kind == "f"


def test_transform_treats_unknown_category_as_baseline(
    pipeline: ClickedAdsPipeline,
):
    raw = _raw_row()
    raw.loc[0, "Category"] = "Unknown category"

    model_input = pipeline.transform(raw)

    category_columns = [
        column
        for column in pipeline.feature_columns
        if column.startswith("Category_")
    ]
    assert (model_input[category_columns] == 0).all().all()
