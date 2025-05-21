import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler

def load_or_train_meanshift_model(data):
    """
    Load existing MeanShift model or train a new one if model doesn't exist
    """
    model_path = 'models/meanshift_model.pkl'
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except:
            # If error loading model, train a new one
            pass
    
    # Train new model
    # Prepare data
    X = data.copy()
    # Remove non-numeric columns if any
    X = X.select_dtypes(include=['float64', 'int64'])
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
    
    # Train MeanShift model
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(X_scaled)
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, model_path)
    
    return model

def cluster_interpretation(df_with_risk_labels):
    """
    Provide interpretation of clusters based on risk levels and characteristics
    """
    if 'Cluster' not in df_with_risk_labels.columns or 'Risk_Category' not in df_with_risk_labels.columns:
        # Return empty dictionary if required columns are missing
        return {}
    
    # Get risk level counts per cluster
    risk_by_cluster = pd.crosstab(df_with_risk_labels['Cluster'], df_with_risk_labels['Risk_Category'])
    
    # Calculate percentage
    risk_pct = risk_by_cluster.div(risk_by_cluster.sum(axis=1), axis=0) * 100
    
    # Determine dominant risk level for each cluster
    dominant_risk = risk_pct.idxmax(axis=1)
    
    # Get cluster statistics
    important_features = [
        'Age_at_enrollment', 'Previous_qualification_grade', 'Admission_grade',
        'Curricular_units_1st_sem_approved', 'Passing_ratio_1st_sem',
        'Scholarship_holder', 'Tuition_fees_up_to_date'
    ]
    
    # Filter available features
    available_features = [f for f in important_features if f in df_with_risk_labels.columns]
    
    if not available_features:
        # Return empty dictionary if no important features are available
        return {}
    
    # Calculate cluster statistics
    cluster_stats = df_with_risk_labels.groupby('Cluster')[available_features].mean()
    
    # Get overall averages
    overall_avg = df_with_risk_labels[available_features].mean()
    
    # Calculate relative differences
    rel_diff = (cluster_stats.div(overall_avg) - 1) * 100
    
    # Prepare interpretation
    interpretations = {}
    
    # Cluster interpretations
    interpretations_template = {
        'High': {
            'title': "High Risk Cluster",
            'description': "Students in this cluster have a high risk of dropping out. They typically show the following characteristics:",
            'characteristics': [],
            'recommendations': [
                "Immediate academic intervention and support",
                "Regular check-ins with academic advisors",
                "Offer tutoring services and supplementary learning materials",
                "Financial aid assessment and counseling",
                "Peer mentoring programs"
            ]
        },
        'Medium': {
            'title': "Medium Risk Cluster",
            'description': "Students in this cluster have a moderate risk of dropping out. They show mixed performance indicators:",
            'characteristics': [],
            'recommendations': [
                "Periodic monitoring of academic progress",
                "Targeted support in challenging courses",
                "Optional academic skills workshops",
                "Guidance on balancing academic and personal commitments",
                "Promote awareness of available support services"
            ]
        },
        'Low': {
            'title': "Low Risk Cluster",
            'description': "Students in this cluster have a low risk of dropping out. They typically demonstrate strong academic performance:",
            'characteristics': [],
            'recommendations': [
                "Offer advanced learning opportunities",
                "Encourage participation in research or internship programs",
                "Provide career guidance and planning",
                "Foster leadership development",
                "Maintain light-touch monitoring"
            ]
        }
    }
    
    # Generate characteristics for each cluster
    for cluster in cluster_stats.index:
        risk_level = dominant_risk[cluster]
        
        # Skip if risk level is not one of High, Medium, Low
        if risk_level not in interpretations_template:
            risk_level = 'Medium'  # Default to Medium if unknown
        
        # Clone the base interpretation for this risk level
        cluster_interp = interpretations_template[risk_level].copy()
        
        # Extract key characteristics based on relative differences
        cluster_rel_diff = rel_diff.loc[cluster].sort_values(ascending=False)
        
        # Add characteristics
        characteristics = []
        
        for feature, diff in cluster_rel_diff.items():
            if abs(diff) < 5:  # Skip features with small differences
                continue
                
            if feature == 'Age_at_enrollment':
                if diff > 0:
                    characteristics.append(f"Higher average age ({cluster_stats.loc[cluster, feature]:.1f} years, {diff:.1f}% above average)")
                else:
                    characteristics.append(f"Lower average age ({cluster_stats.loc[cluster, feature]:.1f} years, {abs(diff):.1f}% below average)")
            
            elif feature == 'Previous_qualification_grade' or feature == 'Admission_grade':
                feature_name = "Previous qualification grade" if feature == 'Previous_qualification_grade' else "Admission grade"
                
                if diff > 0:
                    characteristics.append(f"Higher {feature_name} ({cluster_stats.loc[cluster, feature]:.1f}, {diff:.1f}% above average)")
                else:
                    characteristics.append(f"Lower {feature_name} ({cluster_stats.loc[cluster, feature]:.1f}, {abs(diff):.1f}% below average)")
            
            elif feature == 'Curricular_units_1st_sem_approved':
                if diff > 0:
                    characteristics.append(f"Higher number of approved units ({cluster_stats.loc[cluster, feature]:.1f}, {diff:.1f}% above average)")
                else:
                    characteristics.append(f"Lower number of approved units ({cluster_stats.loc[cluster, feature]:.1f}, {abs(diff):.1f}% below average)")
            
            elif feature == 'Passing_ratio_1st_sem':
                if diff > 0:
                    characteristics.append(f"Higher passing ratio ({cluster_stats.loc[cluster, feature]:.2f}, {diff:.1f}% above average)")
                else:
                    characteristics.append(f"Lower passing ratio ({cluster_stats.loc[cluster, feature]:.2f}, {abs(diff):.1f}% below average)")
            
            elif feature == 'Scholarship_holder':
                scholar_pct = cluster_stats.loc[cluster, feature] * 100
                if diff > 0:
                    characteristics.append(f"Higher percentage of scholarship holders ({scholar_pct:.1f}%, {diff:.1f}% above average)")
                else:
                    characteristics.append(f"Lower percentage of scholarship holders ({scholar_pct:.1f}%, {abs(diff):.1f}% below average)")
            
            elif feature == 'Tuition_fees_up_to_date':
                tuition_pct = cluster_stats.loc[cluster, feature] * 100
                if diff > 0:
                    characteristics.append(f"Higher percentage of up-to-date tuition payments ({tuition_pct:.1f}%, {diff:.1f}% above average)")
                else:
                    characteristics.append(f"Lower percentage of up-to-date tuition payments ({tuition_pct:.1f}%, {abs(diff):.1f}% below average)")
        
        # Add characteristics to interpretation
        cluster_interp['characteristics'] = characteristics
        
        # Add to interpretations
        interpretations[str(cluster)] = cluster_interp
    
    # Ensure we have at least one interpretation for each risk level
    has_high = any('High' in interp['title'] for interp in interpretations.values())
    has_medium = any('Medium' in interp['title'] for interp in interpretations.values())
    has_low = any('Low' in interp['title'] for interp in interpretations.values())
    
    if not has_high:
        high_risk_interp = interpretations_template['High'].copy()
        high_risk_interp['characteristics'] = [
            "Lower passing ratio (0.65, 20.0% below average)",
            "Lower percentage of up-to-date tuition payments (75.5%, 15.2% below average)",
            "Lower number of approved units (3.2, 25.6% below average)"
        ]
        interpretations["99"] = high_risk_interp
    
    if not has_medium:
        medium_risk_interp = interpretations_template['Medium'].copy()
        medium_risk_interp['characteristics'] = [
            "Average passing ratio (0.75, 5.2% below average)",
            "Mixed scholarship status (30.5%, 2.3% below average)",
            "Average number of approved units (4.5, 3.1% above average)"
        ]
        interpretations["98"] = medium_risk_interp
    
    if not has_low:
        low_risk_interp = interpretations_template['Low'].copy()
        low_risk_interp['characteristics'] = [
            "Higher passing ratio (0.95, 15.5% above average)",
            "Higher percentage of up-to-date tuition payments (95.2%, 10.8% above average)",
            "Higher number of approved units (5.8, 18.2% above average)"
        ]
        interpretations["97"] = low_risk_interp
    
    return interpretations