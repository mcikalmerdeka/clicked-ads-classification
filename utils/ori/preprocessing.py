import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

# Import the preprocessed original data
url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/main/data/df_model.csv"
ori_df_preprocessed = pd.read_csv(url_ori_processed, index_col=0)
ori_df_preprocessed = ori_df_preprocessed.loc[:, ori_df_preprocessed.columns != "Clicked on Ad"]

# =====================================================================Functions for data pre-processing========================================================================

## Checking basic data information
def check_data_information(data, cols):
    list_item = []
    for col in cols:
        # Convert unique values to string representation
        unique_sample = ', '.join(map(str, data[col].unique()[:5]))
        
        list_item.append([
            col,                                           # The column name
            str(data[col].dtype),                          # The data type as string
            data[col].isna().sum(),                        # The count of null values
            round(100 * data[col].isna().sum() / len(data[col]), 2),  # The percentage of null values
            data.duplicated().sum(),                       # The count of duplicated rows
            data[col].nunique(),                           # The count of unique values
            unique_sample                                  # Sample of unique values as string
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Duplicated Values',
            'Unique Values',
            'Unique Sample'
        ]
    )
    return desc_df

## Initial data transformation
def initial_data_transform(data):

    # Rename column name for and maintain column name similarity
    data = data.rename(columns={'Male': 'Gender',
                                'Timestamp': 'Visit Time',
                                'city' : 'City',
                                'province' : 'Province',
                                'category' : 'Category'})

    # Re-arrange column (target 'Clicked on Ad' at the end --> personal preference)
    data = data[[col for col in data.columns if col != 'Clicked on Ad'] + ['Clicked on Ad']]

    # Change data type of Visit Time to datetime
    data['Visit Time'] = pd.to_datetime(data['Visit Time'])

    return data

## Handle missing values function
def handle_missing_values(data, columns, strategy='fill', imputation_method='median'):
    # Return the original data if the column is empty
    if columns is None:
        return data
    
    # Impute missing values based on the strategy
    if strategy == 'fill':
        if imputation_method == 'median':
            return data[columns].fillna(data[columns].median())
        elif imputation_method == 'mean':
            return data[columns].fillna(data[columns].mean())
        elif imputation_method == 'mode':
            return data[columns].fillna(data[columns].mode().iloc[0])
        elif imputation_method == 'ffill':
            return data[columns].fillna(method='ffill')
        elif imputation_method == 'bfill':
            return data[columns].fillna(method='bfill')
        else:
            return data[columns].fillna(data[columns].median())

    # Remove rows with missing values
    elif strategy == 'remove':
        return data.dropna(subset=columns)
    
## Drop columns function
def drop_columns(data, columns):
    return data.drop(columns=columns, errors='ignore')

## Handle outliers function
def filter_outliers(data, col_series, method='iqr', threshold=3):
    # Return the original data if the column series is empty
    if col_series is None:
        return data

    # Validate the method parameter
    if method.lower() not in ['iqr', 'zscore']:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    # Start with all rows marked as True (non-outliers)
    filtered_entries = np.array([True] * len(data))
    
    # Loop through each column
    for col in col_series:
        if method.lower() == 'iqr':
            # IQR method
            Q1 = data[col].quantile(0.25)  # First quartile (25th percentile)
            Q3 = data[col].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - (IQR * 1.5)  # Lower bound for outliers
            upper_bound = Q3 + (IQR * 1.5)  # Upper bound for outliers

            # Create a filter that identifies non-outliers for the current column
            filter_outlier = ((data[col] >= lower_bound) & (data[col] <= upper_bound))
            
        elif method.lower() == 'zscore':  # zscore method
            # Calculate Z-Scores and create filter
            z_scores = np.abs(stats.zscore(data[col]))

            # Create a filter that identifies non-outliers
            filter_outlier = (z_scores < threshold)
        
        # Update the filter to exclude rows that have outliers in the current column
        filtered_entries = filtered_entries & filter_outlier
    
    return data[filtered_entries]

## Feature encoding function (only columns that are used in the model)
def feature_encoding(data, original_data=ori_df_preprocessed):
        # A. Handle ordinal encoding for Gender
        data['Gender'] = data['Gender'].map({'Perempuan': 0, 'Laki-Laki': 1})
        
        # B. Handle one-hot encoding for Category using original data categories
        unique_category = original_data.filter(like='Category_').columns
        category_encoded = pd.DataFrame(0, index=data.index, columns=unique_category)
        if f"Category_{data['Category'].iloc[0]}" in unique_category:
            category_encoded[f"Category_{data['Category'].iloc[0]}"] = 1
        data = drop_columns(data, ['Category'])
        data = pd.concat([data, category_encoded], axis=1)

        # Ensure all expected columns are present before moving to scaling
        expected_columns = original_data.columns.tolist()
        for col in expected_columns:
            if col not in data.columns:
                data[col] = 0

        for col in data.columns:
            if col not in expected_columns:
                data.drop(columns=col, inplace=True)

        # Reorder and match columns to match training data
        data = data[expected_columns]

        return data, expected_columns

## Feature scaling function
def feature_scaling(data, original_data=ori_df_preprocessed):
        
        # Initialize scalers
        standard_scaler = StandardScaler()

        # Define feature groups for targeted scaling
        # Each feature group requires a specific scaling approach
        standard_scaling_features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']  # Apply standard scaling to these features

        scalers = {}  # Dictionary to store fitted scalers and feature info

        # TRAINING DATA SCALING
        # Step 1: Scale normally distributed features using standardization
        original_data[standard_scaling_features] = standard_scaler.fit_transform(original_data[standard_scaling_features])
        scalers['standardization'] = standard_scaler

        # INFERENCE DATA SCALING
        # Apply the same transformations used in training data
        # Use .transform() instead of .fit_transform() to maintain training distribution

        # Scale normal-distributed features
        data[standard_scaling_features] = scalers['standardization'].transform(data[standard_scaling_features])

        return data

