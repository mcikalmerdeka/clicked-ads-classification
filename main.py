import streamlit as st
import os
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all preprocessing functions
from preprocessing import (check_data_information,
                           initial_data_transform,
                           handle_missing_values,
                           drop_columns,
                           filter_outliers,
                           feature_encoding,
                           feature_scaling
)

from feature_definitions import get_feature_definitions

# Page Config
st.set_page_config(page_title="Ad Click Prediction Dashboard", layout="wide")
st.title("Digital Ad Performance Analyzer")

# Author Information
st.markdown("""
#### Author
Developed by : Muhammad Cikal Merdeka | Data Analyst/Data Scientist | Data Department

- [Email](mailto:mcikalmerdeka@gmail.com)
- [GitHub Profile](https://github.com/mcikalmerdeka)  
- [LinkedIn Profile](https://www.linkedin.com/in/mcikalmerdeka)
""")

# Add information about the app
with st.expander("**Read Instructions First: About This App**"):
    st.markdown("""
    ## Digital Ad Performance Analyzer

    ### 📌 Problem Statement
    A digital marketing company in Indonesia faces challenges in determining the effectiveness of their advertisements and identifying the right target audience. The current approach lacks precision in reaching potential customers who are likely to engage with the ads, resulting in inefficient ad spending and lower click-through rates. This leads to reduced return on investment (ROI) on their advertising campaigns and potential loss of market opportunities.
    
    ### 🎯 Goals & Objectives
    - **Primary Goal**: Enhance ad targeting by implementing machine learning to identify potential customers most likely to click on ads
    - **Secondary Goal**: Increase ROI by optimizing advertising spend for the most receptive audience segments
    
    The machine learning model aims to:
    - Accurately predict which users are most likely to click on advertisements
    - Identify patterns in customer behavior to optimize targeting strategies
    - Reduce advertising costs while improving engagement rates

    ### 📊 Key Business Metrics
    Two critical metrics that demonstrate our project's success:

    #### Primary Metric: Click-Through Rate (CTR)
    - Measures the percentage of users who click on the advertisement
    - Calculated as: (Number of Clicks / Total Ad Views) × 100
    - Improved from 50% to 99.8% after model implementation
    - Indicates effectiveness of targeting strategy

    #### Secondary Metric: Return on Ad Spend (ROAS)
    - Measures revenue generated per rupiah spent on advertising
    - Improved from 1.25 to 2.43 IDR
    - Demonstrates financial efficiency of ad campaigns
    - Profit increased from Rp.1,500,000 to Rp.8,727,810 (581.8% increase)

    ### 🔍 **How to Use the App**
                
    #### Prediction Options:
    - Input new customer data for ad targeting analysis
    - Receive instant prediction of click probability

    #### Data Input Methods:
    - A. Individual Customer Analysis
        - Enter details for a single customer through the form
    - B. Batch Data Processing
        - Upload multiple customer records
        - Ensure dataset includes: daily site time, age, area income, internet usage, etc.
    - Note: Sample data is available for testing the model

    #### Data Processing Pipeline:
    Behind the scenes, the application processes customer data through these steps:
    1. Data Type Conversion
        - Standardize input formats
    2. Missing Value Treatment
        - Handle incomplete data points
    3. Outlier Management
        - Address extreme values
    4. Feature Engineering
        - Create derived insights from raw data
    5. Feature Encoding
        - Convert categorical data to numerical format
    6. Feature Selection
        - Focus on most predictive variables
    7. Data Scaling
        - Normalize data for consistent analysis

    ### 🤖 Model Performance
    - Implements a tuned Logistic Regression model
    - Accuracy: 97.3%
    - Significant improvements in both CTR and ROAS
    - Trained on comprehensive customer behavior dataset

    ### ⚠️ <span style="color:red;"> Important Notes </span>
    - Predictions are probability-based and should guide, not replace, marketing strategy
    - **Combine model insights with marketing expertise for best results**
    - Regular model updates recommended to maintain performance
    - Results may vary based on market conditions and campaign specifics
    """, unsafe_allow_html=True)

# Load pre-trained model
@st.cache_resource
def load_model():
    # Get the current script's directory
    current_dir = os.getcwd()
    
    # Try multiple possible locations
    possible_paths = [
        # Try in a 'models' directory at the current level
        os.path.join(current_dir, 'models', 'tuned_logistic_regression_model.joblib'),
        # Try in a 'models' directory one level up
        os.path.join(current_dir, os.pardir, 'models', 'tuned_logistic_regression_model.joblib'),
        # Try directly in the current directory
        os.path.join(current_dir, 'tuned_logistic_regression_model.joblib')
    ]
    
    # Try each path
    for model_path in possible_paths:
        try:
            model = joblib.load(model_path)
            return model, os.path.dirname(os.path.dirname(model_path)), model_path
        except FileNotFoundError:
            continue
    
    # If no path worked, show error with all attempted paths
    st.error(f"Model file not found. Attempted the following paths:\n" + 
             "\n".join(possible_paths))
    return None, current_dir, None

# Load the model
model, model_dir, model_path = load_model() # To use the model, call model.predict(data)

# Load original CSV data form author github
url_ori = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/main/data/Clicked%20Ads%20Dataset.csv"
ori_df = pd.read_csv(url_ori, index_col=0)

# Initial transform for original dataframe
ori_df = initial_data_transform(ori_df)

# Display original data
st.subheader("Original Data Preview")
st.write(ori_df.head())

# Display data information
with st.expander("📊 Data Information"):
    st.markdown("### Data Information")
    st.write(check_data_information(ori_df, ori_df.columns))

# Add Data Dictionary section
with st.expander("📚 Data Dictionary"):
    st.markdown("### Feature Information")
    
    # Create DataFrame from feature definitions
    definitions = get_feature_definitions()
    feature_df = pd.DataFrame.from_dict(definitions, orient='index')
    
    # Reorder columns and reset index to show feature names as a column
    feature_df = feature_df.reset_index().rename(columns={'index': 'Feature Name'})
    feature_df = feature_df[['Feature Name', 'description', 'data_type', 'specific_type']]
    
    # Rename columns for display
    feature_df.columns = ['Feature Name', 'Description', 'Data Type', 'Specific Type']
    
    # Display as a styled table
    st.dataframe(
        feature_df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'lightgrey'
        })
    )
    
    st.markdown("""
    **Note:**
    - Categorical (Nominal): Categories without any natural order
    - Categorical (Ordinal): Categories with a natural order
    - Numerical (Discrete): Whole numbers
    - Numerical (Continuous): Any numerical value
    """)

## Specifying some variable values for the model and code flow
target_col = "Clicked on Ad"
gather_data = False

# Import the preprocessed original data (this will be used to match the columns used in the model)
url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/main/data/df_model.csv"
ori_df_preprocessed = pd.read_csv(url_ori_processed)
ori_df_preprocessed = ori_df_preprocessed.loc[:, ori_df_preprocessed.columns != target_col]

# Input type selection
input_type = st.empty()
input_type = st.radio('Select Input Type', ['Individual Customer', 'Batch Data'])
if input_type.lower() == 'individual customer':
    st.write('Please provide the details of the customer in the form below')

    # Input individual customer data
    st.subheader("Enter Customer Data")
    with st.form("customer_prediction_form"):
        # Create a dictionary to store input values
        prediction_input = {}

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        # Split columns into two groups for layout
        all_columns = [col for col in ori_df.columns if col != target_col]
        mid_point = len(all_columns) // 2

        with col1:
            for column in all_columns[:mid_point]:
                if pd.api.types.is_datetime64_any_dtype(ori_df[column]):
                    prediction_input[column] = st.date_input(f"Enter {column}")

                elif pd.api.types.is_numeric_dtype(ori_df[column]):
                    col_min = ori_df[column].min()
                    col_max = ori_df[column].max()
                    col_mean = ori_df[column].mean()

                    prediction_input[column] = st.number_input(
                        f"Enter {column}",
                        min_value=float(col_min) if not pd.isna(col_min) else 0.0,
                        max_value=float(col_max) if not pd.isna(col_max) else None,
                        value=float(col_mean) if not pd.isna(col_mean) else 0.0,
                        step=0.1
                    )
                    
                elif pd.api.types.is_categorical_dtype(ori_df[column]) or ori_df[column].dtype == 'object':
                    unique_values = ori_df[column].unique()
                    prediction_input[column] = st.selectbox(
                        f'Select {column}',
                        options=list(unique_values)
                    )
                
                else:
                    prediction_input[column] = st.text_input(f'Enter {column}')

        with col2:
            for column in all_columns[mid_point:]:
                if pd.api.types.is_datetime64_any_dtype(ori_df[column]):
                    prediction_input[column] = st.date_input(f"Enter {column}")
                
                elif pd.api.types.is_numeric_dtype(ori_df[column]):
                    col_min = ori_df[column].min()
                    col_max = ori_df[column].max()
                    col_mean = ori_df[column].mean()

                    prediction_input[column] = st.number_input(
                        f"Enter {column}",
                        min_value=float(col_min) if not pd.isna(col_min) else 0.0,
                        max_value=float(col_max) if not pd.isna(col_max) else None,
                        value=float(col_mean) if not pd.isna(col_mean) else 0.0,
                        step=0.1
                    )
                    
                elif pd.api.types.is_categorical_dtype(ori_df[column]) or ori_df[column].dtype == 'object':
                    unique_values = ori_df[column].unique()
                    prediction_input[column] = st.selectbox(
                        f'Select {column}',
                        options=list(unique_values)
                    )
                
                else:
                    prediction_input[column] = st.text_input(f'Enter {column}')

        # Add hint for user testing
        with st.expander("📌 Hint for Testing Model Prediction"):
            st.write("You can use the example data below as a reference for input values:")

            # Example data of a customer who is predicted as Clicked on Ad
            example_data_1 = {
                "Daily Time Spent on Site": 33,
                "Age": 55,
                "Area Income": 97975500,
                "Daily Internet Usage": 105,
                "Gender": "Perempuan",
                "Visit Date": "2023-08-10",
                "City": "Jakarta Timur",
                "Province": "Daerah Khusus Ibukota Jakarta",
                "Category": "Fashion",
            }
            st.table(pd.DataFrame([example_data_1]))
            st.write("Which will result in a prediction of <span style='color:green;'>**Clicked on Ad**</span>", unsafe_allow_html=True)
            
            # Example data of a customer who is predicted as Not Clicked on Ad
            example_data_2 = {
                "Daily Time Spent on Site": 92,
                "Age": 19,
                "Area Income": 556393600,
                "Daily Internet Usage": 267,
                "Gender": "Laki-laki",
                "Visit Date": "2023-08-10",
                "City": "Pekanbaru",
                "Province": "Riau",
                "Category": "Otomotif"
            }
            st.table(pd.DataFrame([example_data_2]))
            st.write("Which will result in a prediction of <span style='color:red;'>**Not Clicked on Ad**</span>", unsafe_allow_html=True)

            st.write("Note: You can see the behaviour of the model and how it prefer certain values to be predicted as Clicked on Ad or Not Clicked on Ad")


        # Submit button
        submit_prediction_button = st.form_submit_button("Predict Click Probability")
        gather_data = True

elif input_type.lower() == 'batch data':
    st.write('Please upload the dataset of the customers \n\n Ensure your dataset matches the required structure for Clicked Ads Prediction (check the example data preview, exclude the target column)')

    # File upload
    uploaded_data = st.file_uploader("Choose a CSV file (**Please make sure you convert it to csv first**)", type="csv")

    if uploaded_data is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_data)
            batch_input_df = initial_data_transform(batch_input_df)
            st.success("File uploaded successfully")
            gather_data = True

        except Exception as e:
            st.error(f"Error uploading the file: {str(e)}")

    # Add hint for user testing
    with st.expander("📌 Hint for Testing Model Prediction"):
        st.write("You can use the example data by clicking this button below as a reference for input values:")

        # First button with a unique key
        if st.button("Use Example Data", key="example_data_button"):
            # Load example CSV data from author's GitHub
            url_example_batch_df = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/main/data/batch_example.csv"
            batch_input_df = pd.read_csv(url_example_batch_df)
            batch_input_df = initial_data_transform(batch_input_df)
            gather_data = True

# Prediction Section

## Prediction for individual customer
if gather_data and input_type.lower() == 'individual customer':
    if submit_prediction_button:
        # Convert input data into dataframe
        input_df = pd.DataFrame([prediction_input])

        # Show input data
        st.subheader("New Customer Input Data Preview")
        st.write(input_df)

        # Preprocessing steps

        ## 1. Drop some uncessary columns
        try:
            input_df = drop_columns(input_df, columns=['Visit Time'])
        except Exception as e:  
            st.error(f"Error in dropping columns: {str(e)}")

        ## 2. Handle Missing Values
        try:
            input_df[['Daily Time Spent on Site', 'Daily Internet Usage']] = handle_missing_values(input_df, columns=['Daily Time Spent on Site', 'Daily Internet Usage'], strategy='fill', imputation_method='mean')
            input_df['Area Income'] = handle_missing_values(input_df, columns=['Area Income'], strategy='fill', imputation_method='median')
            input_df['Gender'] = handle_missing_values(input_df, columns=['Gender'], strategy='fill', imputation_method='mode')
        except Exception as e:
            st.error(f"Error in handling missing values: {str(e)}")

        ## 3. Handle Outliers
        try:
            input_df = filter_outliers(input_df, col_series=['Area Income'], method='iqr')
        except Exception as e:
            st.error(f"Error in handling outliers: {str(e)}")

        # Check data after drop unecessart columns, handling missing values and outliers
        st.subheader("After Drop Columns, Handling Missing Values, and Outliers")
        st.write(input_df)

        ## 4. Feature encoding
        try:
            input_df, expected_columns = feature_encoding(input_df, original_data=ori_df_preprocessed)
            st.session_state.expected_columns = expected_columns
        except Exception as e:
            st.error(f"Error in feature encoding: {str(e)}")
            st.write("Debug information:")
            st.write("Current columns:", input_df.columns.to_list())
            st.write("Expected columns:", expected_columns)

        # Check data after encoding
        st.subheader("After Feature Encoding")
        st.write(input_df)

        ## 5. Feature Scaling
        try:
            input_df = feature_scaling(data=input_df, original_data=ori_df_preprocessed)
        except Exception as e:
            st.error(f"Error in feature scaling: {str(e)}")

        # Check data after scaling
        st.subheader("After Feature Scaling")
        st.write(input_df)

        # Prediction Section
        st.subheader("Prediction Section")

        # Create a copy for preprocessing result
        model_df = input_df.copy()

        # Display the prediction result
        try:
            prediction = model.predict(model_df)
            prediction_proba = model.predict_proba(model_df)

            # Display prediction probability
            st.write(f"Probabilities for both classes: {prediction_proba[0]}")

            # Display prediction result with explanation
            # 0: Not Clicked, 1: Clicked
            if prediction[0] == 0:
                st.error("The customer is predicted as **Not Clicked**.\n\n**Not Clicked** means the customer is likely not to click on the ad.")
            else:
                st.success("The customer is predicted as **Clicked**.\n\n**Clicked** means the customer is likely to click on the ad.")
        
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

## Prediction for batch data
elif gather_data and input_type.lower() == 'batch data':
        # Show batch input data
        st.subheader("Batch Data Preview")
        st.write(batch_input_df.head())

        # Preprocessing steps

        ## 1. Drop some uncessary columns
        try:
            batch_input_df = drop_columns(batch_input_df, columns=['Visit Time'])
        except Exception as e:  
            st.error(f"Error in dropping columns: {str(e)}")

        ## 2. Handle Missing Values
        try:
            batch_input_df[['Daily Time Spent on Site', 'Daily Internet Usage']] = handle_missing_values(batch_input_df, columns=['Daily Time Spent on Site', 'Daily Internet Usage'], strategy='fill', imputation_method='mean')
            batch_input_df['Area Income'] = handle_missing_values(batch_input_df, columns=['Area Income'], strategy='fill', imputation_method='median')
            batch_input_df['Gender'] = handle_missing_values(batch_input_df, columns=['Gender'], strategy='fill', imputation_method='mode')
        except Exception as e:
            st.error(f"Error in handling missing values: {str(e)}")

        ## 3. Handle Outliers
        try:
            batch_input_df = filter_outliers(batch_input_df, col_series=['Area Income'], method='iqr')
        except Exception as e:
            st.error(f"Error in handling outliers: {str(e)}")

        # Check data after drop unecessart columns, handling missing values and outliers
        st.subheader("After Drop Columns, Handling Missing Values, and Outliers")
        st.write(batch_input_df)

        ## 4. Feature encoding
        try:
            batch_input_df, expected_columns = feature_encoding(batch_input_df, original_data=ori_df_preprocessed)
            st.session_state.expected_columns = expected_columns
        except Exception as e:
            st.error(f"Error in feature encoding: {str(e)}")
            st.write("Debug information:")
            st.write("Current columns:", batch_input_df.columns.to_list())
            st.write("Expected columns:", expected_columns)

        # Check data after encoding
        st.subheader("After Feature Encoding")
        st.write(batch_input_df)

        ## 5. Feature Scaling
        try:
            batch_input_df = feature_scaling(data=batch_input_df, original_data=ori_df_preprocessed)
        except Exception as e:
            st.error(f"Error in feature scaling: {str(e)}")

        # Check data after scaling
        st.subheader("After Feature Scaling")
        st.write(batch_input_df)     

        # Prediction Section
        st.subheader("Prediction Section")

        # Create a copy for preprocessing result
        model_df = batch_input_df.copy()

        # Display prediction result with explanation
        # 0: Not Clicked, 1: Clicked
        try:
            for row in range(len(model_df)):
                prediction = model.predict(model_df.iloc[row, :].values.reshape(1, -1))
                prediction_proba = model.predict_proba(model_df.iloc[row, :].values.reshape(1, -1))

                # Display prediction probability
                st.write(f"Probabilities for both classes for Customer {row + 1}: {prediction_proba[0]}")

                # Display prediction result with explanation
                if prediction[0] == 0:
                    st.error(f"Customer {row + 1} is predicted as **Not Clicked**.\n\n**Not Clicked** means the customer is likely not to click on the ad.")
                else:
                    st.success(f"Customer {row + 1} is predicted as **Clicked**.\n\n**Clicked** means the customer is likely to click on the ad.")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}") 