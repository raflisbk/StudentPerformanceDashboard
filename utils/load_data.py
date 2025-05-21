import pandas as pd
import os
import numpy as np
import streamlit as st

@st.cache_data
def load_all_data():
    """
    Load all required datasets for the dashboard
    """
    try:
        # Main dataset with risk labels
        df_with_risk_labels = pd.read_csv('data/data_with_risk_labels.csv')
        
        # Convert boolean string values to actual booleans
        for col in df_with_risk_labels.columns:
            if df_with_risk_labels[col].dtype == 'object':
                if set(df_with_risk_labels[col].dropna().unique()).issubset({'True', 'False'}):
                    df_with_risk_labels[col] = df_with_risk_labels[col].map({'True': True, 'False': False})
        
        # Original dataset (before clustering)
        try:
            df = pd.read_csv('data/data.csv', sep=';')
        except:
            # If the original data.csv is not available, use the risk labels dataset
            df = df_with_risk_labels
        
        # Clustering data
        try:
            clustering_data = pd.read_csv('data/clustering_data.csv')
        except:
            # If clustering data is not available, use a subset of the risk labels dataset
            clustering_data = df_with_risk_labels.copy()
        
        return df, df_with_risk_labels, clustering_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrames as fallback
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df