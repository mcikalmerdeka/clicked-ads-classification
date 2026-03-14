def get_feature_definitions():
    return {
        "Daily_Time_Spent_on_Site": {
            "description": "Time spent on the site (Minutes)",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Age": {
            "description": "Age of the customer",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Area_Income": {
            "description": "Average income of geographical area of consumer",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Daily_Internet_Usage": {
            "description": "Time spent on the internet (Minutes)",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Gender": {
            "description": "Gender of the customer",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Visit_Time": {
            "description": "Time the customer visit on the website",
            "data_type": "Date",
            "specific_type": "Timestamp"
        },
        "City": {
            "description": "City of customer's residence",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Province": {
            "description": "Province of customer's residence",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Category": {
            "description": "Category of the advertisement",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Clicked_on_Ad": {
            "description": "Whether the customer clicked the ad or not (Target Variable)",
            "data_type": "Categorical",
            "specific_type": "Nominal (Binary)"
        }
    }