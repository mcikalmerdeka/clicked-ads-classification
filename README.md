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
├── analysis/               # Jupyter notebooks (EDA, modeling, evaluation)
├── data/                   # Raw and processed datasets
├── models/                 # Trained model artifacts
├── utils/                  # Reusable preprocessing and ML functions
├── main.py                 # Streamlit application
├── pyproject.toml          # Project dependencies (uv/pip)
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
uv add -r requirements.txt
```

### Run the App

```bash
streamlit run main.py
```

Access the app at `http://localhost:8501`

## Features

- **Individual Customer Analysis**: Single prediction with detailed probability breakdown
- **Batch Processing**: Upload CSV files for bulk predictions
- **Data Dictionary**: Comprehensive feature explanations
- **Example Data**: Built-in test cases for quick validation

## Technical Stack

- Python 3.12+
- scikit-learn (modeling)
- pandas, numpy (data processing)
- Streamlit (web app)
- uv (dependency management)

## Business Problem

A digital marketing company needed to improve ad targeting precision to reduce wasted ad spend and increase ROI. The solution identifies high-probability clickers based on customer demographics, behavior patterns, and geographic data.

## Try the Live App

[Streamlit Cloud Deployment](https://clicked-ads-classification-6z2cxyrntysmfngkraa6rc.streamlit.app/)
