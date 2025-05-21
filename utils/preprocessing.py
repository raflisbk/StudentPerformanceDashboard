import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess data for analysis and modeling
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Convert boolean string values to actual booleans
    for col in data.columns:
        if data[col].dtype == 'object':
            if set(data[col].dropna().unique()).issubset({'True', 'False'}):
                data[col] = data[col].map({'True': True, 'False': False})
    
    # Convert boolean columns to int for numerical analysis
    for col in data.select_dtypes(include=['bool']).columns:
        data[col] = data[col].astype(int)
    
    # Handle missing values if any
    if data.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        for col in data.select_dtypes(include=['float', 'int']).columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
        
        # Fill categorical columns with mode
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0])
    
    return data

def prepare_features_for_prediction(user_input):
    """
    Prepares user input features for model prediction
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Calculate additional features
    if 'Curricular_units_1st_sem_enrolled' in input_df.columns and 'Curricular_units_1st_sem_approved' in input_df.columns:
        input_df['Passing_ratio_1st_sem'] = input_df['Curricular_units_1st_sem_approved'] / input_df['Curricular_units_1st_sem_enrolled']
    
    if 'Previous_qualification_grade' in input_df.columns and 'Admission_grade' in input_df.columns:
        input_df['Grade_difference'] = input_df['Admission_grade'] - input_df['Previous_qualification_grade']
    
    # Expected features that might be needed by the model
    expected_features = [
        'Age_at_enrollment', 'Gender', 'Marital_status',
        'Previous_qualification_grade', 'Admission_grade',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved',
        'Passing_ratio_1st_sem', 'Scholarship_holder', 'Debtor',
        'Tuition_fees_up_to_date', 'International',
        'Educational_special_needs', 'Application_order', 'Application_mode',
        'Curricular_units_1st_sem_grade', 'Mothers_qualification',
        'Fathers_qualification', 'Grade_difference'
    ]
    
    # Add missing features with default values
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    return input_df