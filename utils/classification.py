import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
from sklearn.preprocessing import StandardScaler

def load_classification_model():
    """
    Load the classification model for risk prediction
    """
    try:
        # Try to load the SVM model
        model_path = 'models/svm_risk_category_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        
        # If SVM model not found, try RandomForest as fallback
        model_path = 'models/rf_risk_category_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        
        # Return None if no model is found
        return None
    
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

def load_scaler():
    """
    Load the scaler for preprocessing features
    """
    try:
        scaler_path = 'models/risk_category_scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            return scaler
        
        # Return a new scaler if not found
        return StandardScaler()
    
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return StandardScaler()

def predict_risk_level(input_data):
    """
    Predict risk level from input data
    """
    model = load_classification_model()
    
    if model is None:
        return {"error": "Classification model not found. Please train a model first."}
    
    try:
        # Prepare features
        features_expected = [
            'Age_at_enrollment', 'Gender', 'Marital_status',
            'Previous_qualification_grade', 'Admission_grade',
            'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved',
            'Passing_ratio_1st_sem', 'Scholarship_holder', 'Debtor',
            'Tuition_fees_up_to_date', 'International'
        ]
        
        # Calculate derived features
        if 'Curricular_units_1st_sem_enrolled' in input_data and 'Curricular_units_1st_sem_approved' in input_data:
            if input_data['Curricular_units_1st_sem_enrolled'] > 0:
                input_data['Passing_ratio_1st_sem'] = input_data['Curricular_units_1st_sem_approved'] / input_data['Curricular_units_1st_sem_enrolled']
            else:
                input_data['Passing_ratio_1st_sem'] = 0
        
        # Prepare input for prediction
        X = pd.DataFrame([input_data])
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                if col == 'Gender':
                    X[col] = X[col].map({'Male': 1, 'Female': 0})
                elif col in ['Scholarship_holder', 'Tuition_fees_up_to_date', 'International', 'Debtor']:
                    X[col] = X[col].map({'Yes': 1, 'No': 0})
        
        # Get model features
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        else:
            # Use all available features if model doesn't specify
            model_features = X.columns
        
        # Add missing features
        for feature in model_features:
            if feature not in X.columns:
                X[feature] = 0
        
        # Ensure correct order
        X = X[model_features]
        
        # Scale features
        scaler = load_scaler()
        X_scaled = scaler.fit_transform(X)
        
        # Rule-based risk assessment as fallback
        passing_ratio = input_data.get('Passing_ratio_1st_sem', 0)
        admission_grade = input_data.get('Admission_grade', 0)
        scholarship = input_data.get('Scholarship_holder', 'No')
        tuition_uptodate = input_data.get('Tuition_fees_up_to_date', 'No')
        
        # Calculate risk score (0-100)
        risk_score = 0
        
        # Passing ratio impact (0-40 points)
        if passing_ratio < 0.5:
            risk_score += 40  # High risk
        elif passing_ratio < 0.7:
            risk_score += 20  # Medium risk
        elif passing_ratio < 0.85:
            risk_score += 10  # Low risk
        
        # Admission grade impact (0-20 points)
        if admission_grade < 120:
            risk_score += 20
        elif admission_grade < 140:
            risk_score += 10
        
        # Scholarship impact (0-15 points)
        if scholarship == 'No':
            risk_score += 15
        
        # Tuition payment impact (0-15 points)
        if tuition_uptodate == 'No':
            risk_score += 15
        
        # Determine rule-based risk level
        rule_based_risk = "Medium"  # Default
        if risk_score >= 50:
            rule_based_risk = "High"
        elif risk_score <= 25:
            rule_based_risk = "Low"
        
        # Try model prediction
        try:
            model_prediction = model.predict(X_scaled)[0]
            
            # If model predicts Medium, use rule-based approach for more variation
            if model_prediction == "Medium":
                prediction = rule_based_risk
            else:
                prediction = model_prediction
                
            # If model provides probability estimates
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = float(np.max(probabilities)) * 100
            else:
                # Use rule-based confidence
                confidence = 100 - (abs(risk_score - 50) * 1.5)
                confidence = max(60, min(95, confidence))
        except:
            # Fallback to rule-based prediction
            prediction = rule_based_risk
            confidence = 100 - (abs(risk_score - 50) * 1.5)
            confidence = max(60, min(95, confidence))
        
        # Identify key risk factors
        key_factors = identify_key_risk_factors(input_data, prediction)
        
        # Return prediction result
        return {
            "risk_level": prediction,
            "confidence": confidence,
            "key_factors": key_factors
        }
        
    except Exception as e:
        return {"error": f"Error making prediction: {e}"}

def identify_key_risk_factors(input_data, prediction):
    """
    Identify key factors contributing to the risk level
    """
    risk_factors = []
    
    # High risk factors
    high_risk_indicators = {
        'low_passing_ratio': "Low passing ratio (below 0.7)",
        'low_admission_grade': "Low admission grade (below 130)",
        'not_scholarship': "Not a scholarship holder",
        'tuition_not_uptodate': "Tuition fees not up to date",
        'low_units_approved': "Few approved units (below 4)"
    }
    
    # Lower risk indicators
    low_risk_indicators = {
        'high_passing_ratio': "High passing ratio (above 0.8)",
        'high_admission_grade': "High admission grade (above 150)",
        'scholarship': "Scholarship holder",
        'tuition_uptodate': "Tuition fees up to date",
        'high_units_approved': "Many approved units (above 5)"
    }
    
    # Check for risk factors based on prediction
    if prediction == 'High':
        # Check for high risk indicators
        if 'Passing_ratio_1st_sem' in input_data and input_data['Passing_ratio_1st_sem'] < 0.7:
            risk_factors.append(high_risk_indicators['low_passing_ratio'])
        
        if 'Admission_grade' in input_data and input_data['Admission_grade'] < 130:
            risk_factors.append(high_risk_indicators['low_admission_grade'])
        
        if 'Scholarship_holder' in input_data and input_data['Scholarship_holder'] == 'No':
            risk_factors.append(high_risk_indicators['not_scholarship'])
        
        if 'Tuition_fees_up_to_date' in input_data and input_data['Tuition_fees_up_to_date'] == 'No':
            risk_factors.append(high_risk_indicators['tuition_not_uptodate'])
        
        if 'Curricular_units_1st_sem_approved' in input_data and input_data['Curricular_units_1st_sem_approved'] < 4:
            risk_factors.append(high_risk_indicators['low_units_approved'])
    
    elif prediction == 'Low':
        # Check for low risk indicators
        if 'Passing_ratio_1st_sem' in input_data and input_data['Passing_ratio_1st_sem'] > 0.8:
            risk_factors.append(low_risk_indicators['high_passing_ratio'])
        
        if 'Admission_grade' in input_data and input_data['Admission_grade'] > 150:
            risk_factors.append(low_risk_indicators['high_admission_grade'])
        
        if 'Scholarship_holder' in input_data and input_data['Scholarship_holder'] == 'Yes':
            risk_factors.append(low_risk_indicators['scholarship'])
        
        if 'Tuition_fees_up_to_date' in input_data and input_data['Tuition_fees_up_to_date'] == 'Yes':
            risk_factors.append(low_risk_indicators['tuition_uptodate'])
        
        if 'Curricular_units_1st_sem_approved' in input_data and input_data['Curricular_units_1st_sem_approved'] > 5:
            risk_factors.append(low_risk_indicators['high_units_approved'])
    
    # For Medium risk, identify mixed factors
    else:
        # Check for any high risk factors
        if 'Passing_ratio_1st_sem' in input_data and input_data['Passing_ratio_1st_sem'] < 0.7:
            risk_factors.append(high_risk_indicators['low_passing_ratio'])
        elif 'Passing_ratio_1st_sem' in input_data and input_data['Passing_ratio_1st_sem'] > 0.8:
            risk_factors.append(low_risk_indicators['high_passing_ratio'])
        
        # Add at least one factor
        if not risk_factors and 'Admission_grade' in input_data:
            if input_data['Admission_grade'] < 130:
                risk_factors.append(high_risk_indicators['low_admission_grade'])
            elif input_data['Admission_grade'] > 150:
                risk_factors.append(low_risk_indicators['high_admission_grade'])
            else:
                risk_factors.append("Average admission grade (between 130-150)")
        
        # If still no factors, add a default factor
        if not risk_factors:
            risk_factors.append("Mixed performance indicators")
    
    return risk_factors

def get_recommendations(prediction):
    """
    Get recommendations based on predicted risk level
    """
    recommendations = {
        'High': [
            "Schedule regular meetings with an academic advisor",
            "Seek tutoring for challenging courses",
            "Join study groups or peer support networks",
            "Explore financial aid or scholarship options",
            "Consider reducing course load if necessary"
        ],
        'Medium': [
            "Monitor academic progress closely",
            "Attend office hours for courses where improvement is needed",
            "Develop better time management and study skills",
            "Balance academic commitments with other activities",
            "Seek help proactively when challenges arise"
        ],
        'Low': [
            "Continue with current academic strategies",
            "Consider additional academic challenges like research projects",
            "Mentor other students who may benefit from your experience",
            "Plan ahead for advanced coursework and career opportunities",
            "Maintain good communication with instructors and advisors"
        ]
    }
    
    # Return recommendations based on prediction
    if prediction in recommendations:
        return recommendations[prediction]
    
    # Default recommendations if prediction not found
    return recommendations['Medium']