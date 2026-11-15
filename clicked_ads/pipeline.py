"""Project-specific glue for the Clicked Ads app.

``utils/`` is a vendored copy of the ``data-science-utils`` skill package and is
kept pristine so it can be replaced wholesale when the skill is upgraded.
Everything that is specific to this dataset and the persisted model lives here
instead.

The adapter fits its state at startup from data already in the repository:

* categorical encoders from ``data/clicked_ads_dataset.csv``
* the numeric scaler from ``data/df_model.csv``

so there is no extra artifact to keep in sync with the deployed model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from utils import (
    feature_encoding,
    feature_scaling,
    filter_outliers,
    handle_missing_values,
)

from .schema import (
    COLUMN_RENAMES,
    DATETIME_COLUMN,
    GENDER_CATEGORIES,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    TARGET_COLUMN,
)


def initial_data_transform(data: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns, move the target last, and parse timestamps.

    Accepts both the raw Kaggle-style header (``Male``, ``Timestamp``,
    ``city``, ...) and the canonical header used by the app.
    """
    data = data.rename(columns=COLUMN_RENAMES)

    if TARGET_COLUMN in data.columns:
        data = data[
            [col for col in data.columns if col != TARGET_COLUMN] + [TARGET_COLUMN]
        ]

    if DATETIME_COLUMN in data.columns:
        data[DATETIME_COLUMN] = pd.to_datetime(data[DATETIME_COLUMN])

    return data


@dataclass
class ClickedAdsPipeline:
    """Turns raw customer rows into the exact matrix the deployed model expects.

    The persisted model is a bare ``LogisticRegression`` with no preprocessing
    attached (``feature_names_in_`` is the 14 columns of ``df_model.csv``), so
    every step has to be reproduced here in the same order.
    """

    encoders: Dict[str, Any]
    scaler: Any
    feature_columns: List[str]

    @classmethod
    def from_project_data(
        cls,
        raw_path: Union[str, Path],
        reference_path: Union[str, Path],
    ) -> "ClickedAdsPipeline":
        """Fit encoders/scaler from the repository's training artifacts."""
        raw = initial_data_transform(pd.read_csv(raw_path, index_col=0))
        reference = pd.read_csv(reference_path)

        feature_columns = [
            col for col in reference.columns if col != TARGET_COLUMN
        ]

        # Fit only on the columns the model actually consumes. City/Province
        # were explored in the notebook but are not part of the final schema.
        fit_frame = raw.drop(
            columns=[TARGET_COLUMN, DATETIME_COLUMN, "City", "Province"],
            errors="ignore",
        )
        # The raw CSV has a few missing genders; the notebook imputes before
        # encoding, so mirror that here instead of feeding NaN to the encoder.
        fit_frame, _ = handle_missing_values(
            fit_frame, columns=["Gender"], strategy="mode"
        )
        _, encoders = feature_encoding(
            fit_frame,
            ordinal_columns=ORDINAL_COLUMNS,
            ordinal_categories={"Gender": GENDER_CATEGORIES},
            nominal_columns=NOMINAL_COLUMNS,
            drop_first=True,
            handle_unknown="ignore",
        )

        scaler_columns = [col for col in NUMERIC_COLUMNS if col in feature_columns]
        _, scaler = feature_scaling(
            reference[scaler_columns],
            columns=scaler_columns,
            method="standard",
        )

        return cls(
            encoders=encoders,
            scaler=scaler,
            feature_columns=feature_columns,
        )

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Step 1-3: normalise headers, impute, drop unused columns, filter outliers."""
        data = initial_data_transform(data)
        data = data.drop(
            columns=[DATETIME_COLUMN, TARGET_COLUMN], errors="ignore"
        )

        data, _ = handle_missing_values(
            data,
            columns=["Daily Time Spent on Site", "Daily Internet Usage"],
            strategy="mean",
        )
        data, _ = handle_missing_values(
            data, columns=["Area Income"], strategy="median"
        )
        data, _ = handle_missing_values(
            data, columns=["Gender"], strategy="mode"
        )

        return filter_outliers(
            data, columns=["Area Income"], method="iqr", verbose=False
        )

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Step 4: encode categoricals and align to the model's input columns."""
        data = data.copy()

        if "Gender" in data.columns:
            # The app hint uses "Laki-laki"; normalise before ordinal encoding.
            data["Gender"] = data["Gender"].str.title()

        encoded, _ = feature_encoding(
            data,
            ordinal_columns=ORDINAL_COLUMNS,
            ordinal_categories={"Gender": GENDER_CATEGORIES},
            nominal_columns=NOMINAL_COLUMNS,
            drop_first=True,
            handle_unknown="ignore",
            encoders=self.encoders,
        )

        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Step 5: standard-scale the four continuous features."""
        numeric_columns = [col for col in NUMERIC_COLUMNS if col in data.columns]
        scaled, _ = feature_scaling(
            data,
            columns=numeric_columns,
            method="standard",
            scaler=self.scaler,
        )
        return scaled

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the full clean -> encode -> scale pipeline in one call."""
        return self.scale(self.encode(self.clean(data)))
