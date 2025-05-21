import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

# Import utility modules
from utils.load_data import load_all_data
from utils.preprocessing import preprocess_data
from pages_content import home, data_exploration, clustering, classification, about

# Set page config
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
def load_css():
    with open("assets/css/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add global colors
COLORS = {
    "background": "#13111C",
    "surface": "#1E1B2E",
    "surface_light": "#29263E",
    "primary": "#7B61FF",
    "primary_light": "#9F89FF",
    "secondary": "#00E7A2", 
    "accent": "#FF5E8F",
    "text": "#FFFFFF",
    "text_secondary": "#B4B4B4",
    "success": "#00E7A2",
    "warning": "#FFD166",
    "error": "#FF5E8F",
    "border": "#3B3651",
    "charts": {
        "high_risk": "#FF5E8F",
        "medium_risk": "#FFD166",
        "low_risk": "#00E7A2"
    }
}

# Define text for navbar
NAV_ITEMS = [
    {"icon": "🏠", "label": "Home"},
    {"icon": "📊", "label": "Data Exploration"},
    {"icon": "🔍", "label": "Clustering Analysis"},
    {"icon": "⚠️", "label": "Risk Prediction"},
    {"icon": "ℹ️", "label": "About"}
]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = NAV_ITEMS[0]["label"]
if 'show_stats' not in st.session_state:
    st.session_state.show_stats = False

def toggle_stats_visibility():
    """Function to toggle the visibility of statistics panel"""
    st.session_state.show_stats = not st.session_state.show_stats

def main():
    # Apply global styles
    st.markdown(f"""
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
        transition: all 0.2s ease;
    }}
    
    /* App background */
    .main {{
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
    }}
    
    .stApp {{
        background-color: {COLORS["background"]};
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS["text"]};
        font-weight: 600;
    }}
    
    p, div {{
        color: {COLORS["text_secondary"]};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {COLORS["surface"]};
        border-right: 1px solid {COLORS["border"]};
        padding: 2rem 1rem;
        position: relative; /* Untuk positioning footer */
    }}
    
    /* Containers */
    [data-testid="stContainer"] {{
        background-color: {COLORS["surface"]};
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid {COLORS["border"]};
    }}
    
    /* Metrics */
    [data-testid="stMetric"] {{
        background-color: {COLORS["surface_light"]};
        border-radius: 8px;
        padding: 1rem;
        transition: transform 0.2s;
    }}
    
    [data-testid="stMetric"]:hover {{
        transform: translateY(-5px);
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_secondary"]};
    }}
    
    [data-testid="stMetricValue"] {{
        color: {COLORS["text"]};
        font-weight: 600;
    }}
    
    /* Buttons */
    button[kind="primary"], div[data-testid="stButton"] > button:first-child {{
        background-color: {COLORS["primary"]};
        color: {COLORS["text"]};
        border: none;
        border-radius: 8px;
        transition: all 0.2s;
    }}
    
    button[kind="primary"]:hover, div[data-testid="stButton"] > button:first-child:hover {{
        background-color: {COLORS["primary_light"]};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(123, 97, 255, 0.3);
    }}
    
    /* Form elements */
    input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {{
        background-color: {COLORS["surface_light"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
    }}
    
    /* Sliders */
    [data-testid="stSlider"] div [data-baseweb="slider"] {{
        background-color: {COLORS["surface_light"]};
    }}
    
    [data-testid="stSlider"] div [data-baseweb="slider"] div {{
        background-color: {COLORS["primary"]} !important;
    }}
    
    /* Remove fullscreen button */
    button[title="View fullscreen"] {{
        display: none;
    }}
    
    /* Hide hamburger menu and footer */
    #MainMenu, footer {{
        display: none !important;
    }}

    /* Styling untuk Dataset Statistics header */
    .stats-header {{
        color: {COLORS['text']};
        font-weight: 500;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        background-color: {COLORS['surface_light']};
        margin: 1.5rem 0 0.5rem 0;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        transition: all 0.3s ease;
    }}
    
    .stats-header:hover {{
        background-color: {COLORS['surface_light']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}

    /* Styling untuk tombol toggle stats */
    .toggle-stats-btn {{
        background-color: {COLORS['surface_light']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        width: 100%;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    
    .toggle-stats-btn:hover {{
        background-color: {COLORS['primary']};
        color: {COLORS['text']};
    }}

    /* Styling untuk footer */
    .dashboard-footer {{
        color: #666;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid {COLORS['border']};
        width: 100%;
    }}

    /* Styling untuk tombol navigasi */
    div.element-container div.stButton > button {{
        width: 100% !important;
        text-align: left !important;
        padding: 0.7rem 1rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 400 !important;
        display: flex !important;
        align-items: center !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Load custom CSS
    load_css()
    
    # Load all required data
    with st.spinner("Loading data..."):
        df, df_with_risk_labels, clustering_data = load_all_data()
    
    # Sidebar
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/3413/3413535.png" width="80" style="margin-bottom: 1rem;">
            <h1 style="font-size: 1.5rem; margin: 0; color: white;">Student Performance<br>Dashboard</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        st.markdown("<div style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
        
        # Render navigation buttons
        for i, item in enumerate(NAV_ITEMS):
            is_active = st.session_state.page == item["label"]
            
            # Styling untuk tombol
            button_style = f"""
            <style>
            div.element-container:nth-child({i+6}) button {{
                background-color: {COLORS["primary"] if is_active else COLORS["surface_light"]};
                color: {COLORS["text"]};
                border: none;
                text-align: left;
                padding: 0.7rem 1rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
                font-weight: {600 if is_active else 400};
                width: 100%;
                transition: all 0.2s;
                display: flex;
                align-items: center;
            }}
            
            div.element-container:nth-child({i+6}) button:hover {{
                background-color: {COLORS["primary"] if is_active else "#3B3651"};
                transform: {f"translateY(-2px)" if not is_active else "none"};
            }}
            </style>
            """
            st.markdown(button_style, unsafe_allow_html=True)
            
            if st.button(f"{item['icon']} {item['label']}"):
                st.session_state.page = item["label"]
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset Statistics section
        with st.expander("📊 Dataset Statistics", expanded=st.session_state.show_stats):
            stats_content = f"""
            <div style="
                padding: 0.5rem;
                font-size: 0.9rem;
                color: {COLORS['text_secondary']};
            ">
                <div style="margin-bottom: 0.5rem;"><b>Total Students:</b> {df.shape[0]}</div>
            """
            
            if 'Status_Dropout' in df.columns:
                if df['Status_Dropout'].dtype == 'object':
                    dropout_rate = df['Status_Dropout'].map({'True': 1, 'False': 0}).mean() * 100
                else:
                    dropout_rate = df['Status_Dropout'].mean() * 100
                stats_content += f'<div style="margin-bottom: 0.5rem;"><b>Dropout Rate:</b> {dropout_rate:.1f}%</div>'
            
            if 'Passing_ratio_1st_sem' in df.columns:
                try:
                    avg_passing_ratio = df['Passing_ratio_1st_sem'].astype(float).mean() * 100
                    stats_content += f'<div style="margin-bottom: 0.5rem;"><b>Avg. Passing Ratio:</b> {avg_passing_ratio:.1f}%</div>'
                except:
                    stats_content += '<div style="margin-bottom: 0.5rem;"><b>Avg. Passing Ratio:</b> N/A</div>'
            
            if 'Scholarship_holder' in df.columns:
                if df['Scholarship_holder'].dtype == 'object':
                    scholarship_rate = df['Scholarship_holder'].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).mean() * 100
                else:
                    scholarship_rate = df['Scholarship_holder'].mean() * 100
                stats_content += f'<div style="margin-bottom: 0.5rem;"><b>Scholarship Rate:</b> {scholarship_rate:.1f}%</div>'
            
            stats_content += '<div style="margin: 0.7rem 0 0.5rem 0;"><b>Risk Categories:</b></div>'
            
            if 'Risk_Category' in df_with_risk_labels.columns:
                risk_counts = df_with_risk_labels['Risk_Category'].value_counts()
                for category, count in risk_counts.items():
                    percentage = (count / len(df_with_risk_labels)) * 100
                    color = COLORS["charts"]["high_risk"] if category == "High" else (COLORS["charts"]["medium_risk"] if category == "Medium" else COLORS["charts"]["low_risk"])
                    stats_content += f'<div style="margin-bottom: 0.3rem; margin-left: 1rem; display: flex; align-items: center;"><span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: {color}; margin-right: 0.5rem;"></span> {category}: {count} ({percentage:.1f}%)</div>'
            
            stats_content += "</div>"
            st.markdown(stats_content, unsafe_allow_html=True)
        
        
        st.markdown("<div style='flex-grow: 1; min-height: 2rem;'></div>", unsafe_allow_html=True)
        
       
        st.markdown(f"""
        <div class="dashboard-footer">
            © 2025 Student Performance Analysis<br>
            Mohamad Rafli Agung Subekti
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    selected = st.session_state.page
    
    if selected == "Home":
        home.show(df, COLORS)
    elif selected == "Data Exploration":
        data_exploration.show(df, df_with_risk_labels, COLORS)
    elif selected == "Clustering Analysis":
        clustering.show(df, clustering_data, df_with_risk_labels, COLORS)
    elif selected == "Risk Prediction":
        classification.show(df_with_risk_labels, COLORS)
    elif selected == "About":
        about.show(COLORS)

if __name__ == "__main__":
    main()