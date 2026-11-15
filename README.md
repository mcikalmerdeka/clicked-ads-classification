# Predict Customer Clicked Ads Classification

![Project Header](assets/Project%20Header.jpg)

A machine learning solution to predict customer ad clicks and optimize digital marketing campaigns.

## Project Overview

End-to-end data science project that analyzes customer behavior data to predict ad click probability. Includes comprehensive EDA, preprocessing pipelines, model training, and an interactive Streamlit dashboard for real-time predictions.

## Key Results

- **Model Accuracy**: 97.3% (Tuned Logistic Regression)
- **CTR Improvement**: 50% → 99.8%
- **ROAS Improvement**: 1.25 → 2.43 IDR
- **Profit Increase**: 581.8% (Rp.1.5M → Rp.8.7M)

## Project Structure

```
├── clicked_ads/            # Application package
│   ├── config.py           # Filesystem paths
│   ├── schema.py           # Dataset schema, constants, and data dictionary
│   ├── pipeline.py         # Preprocessing and model input pipeline
│   ├── analytics.py        # Metrics, coefficients, prediction explanations
│   ├── ui.py               # Shared chart and formatting helpers
│   └── views/              # Tab views: predict, batch, insights, about
├── tests/                  # Pipeline, analytics, and app smoke tests
├── data/                   # Raw and processed datasets
├── models/                 # Trained model artifacts
├── utils/                  # Vendored data-science-utils package (do not edit)
├── tasks/                  # Project deliverables (reports, slides, images)
├── notebook.ipynb          # EDA, modeling, and evaluation notebook
├── main.py                 # Streamlit entry point
├── pyproject.toml          # Project dependencies and tool config
└── requirements.txt        # Pip-compatible dependencies
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning.git
cd Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning

# Install dependencies (using pip)
pip install -r requirements.txt

# Or using uv (faster alternative)
uv sync
```

### Run the App

```bash
streamlit run main.py
```

Access the app at `http://localhost:8501`

### Run Tests

```bash
uv run pytest
```

## Features

- **Individual scoring**: single-customer prediction with probability, confidence, and a per-prediction explanation of the top drivers
- **Batch scoring**: upload a CSV, review a results table, and download predictions
- **Model insights**: held-out metrics, confusion matrix, global coefficients, and dataset overview
- **Model card**: data dictionary, limitations, and provenance
- **Example data**: built-in test cases for quick validation

## Technical Stack

- Python 3.14+
- scikit-learn (modeling)
- pandas, numpy (data processing)
- Streamlit (web app)
- uv (dependency management)

## Business Problem

A digital marketing company needed to improve ad targeting precision to reduce wasted ad spend and increase ROI. The solution identifies high-probability clickers based on customer demographics, behavior patterns, and geographic data.

## Try the Live App

[Streamlit Cloud Deployment](https://clicked-ads-classification-6z2cxyrntysmfngkraa6rc.streamlit.app/)
