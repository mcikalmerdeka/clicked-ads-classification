"""Dataset schema and feature metadata for the Clicked Ads app.

This module is the single source of truth for column names, model inputs, and
the data dictionary. It has no Streamlit or modelling dependencies, so both the
pipeline and the views can import it without creating a cycle.
"""

from __future__ import annotations

from typing import Dict

TARGET_COLUMN = "Clicked on Ad"
DATETIME_COLUMN = "Visit Time"

# The four continuous features the model expects, in the reference dataset order.
NUMERIC_COLUMNS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
]

# Gender is ordinal: Perempuan -> 0, Laki-Laki -> 1 (matches the trained model).
GENDER_CATEGORIES = ["Perempuan", "Laki-Laki"]
ORDINAL_COLUMNS = ["Gender"]
NOMINAL_COLUMNS = ["Category"]

# Columns a batch upload must contain before it can be scored.
REQUIRED_INPUT_COLUMNS = NUMERIC_COLUMNS + ["Gender", "Category"]

COLUMN_RENAMES = {
    "Male": "Gender",
    "Timestamp": "Visit Time",
    "city": "City",
    "province": "Province",
    "category": "Category",
}


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """Data dictionary rendered by the app's "Data Dictionary" expander."""
    return {
        "Daily_Time_Spent_on_Site": {
            "description": "Time spent on the site (Minutes)",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Age": {
            "description": "Age of the customer",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Area_Income": {
            "description": "Average income of geographical area of consumer",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Daily_Internet_Usage": {
            "description": "Time spent on the internet (Minutes)",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Gender": {
            "description": "Gender of the customer",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Visit_Time": {
            "description": "Time the customer visit on the website",
            "data_type": "Date",
            "specific_type": "Timestamp",
        },
        "City": {
            "description": "City of customer's residence",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Province": {
            "description": "Province of customer's residence",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Category": {
            "description": "Category of the advertisement",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Clicked_on_Ad": {
            "description": "Whether the customer clicked the ad or not (Target Variable)",
            "data_type": "Categorical",
            "specific_type": "Nominal (Binary)",
        },
    }
