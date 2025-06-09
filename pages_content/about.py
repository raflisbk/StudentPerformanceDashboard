import streamlit as st

def show(COLORS):
    # Page title
    st.title("About This Project")
    st.markdown("##### Understanding the student performance analysis project")
    
    # Project overview
    st.markdown(f"""
    <div class="form-section">
        <h3>Project Overview</h3>
        <p>
            This project aims to analyze student performance data from a higher education institution to identify factors 
            that contribute to student success or dropout. By using machine learning techniques, we can identify patterns 
            and predict students who might be at risk of dropping out, allowing for early intervention.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source
    st.markdown(f"""
    <div class="form-section">
        <h3>Data Source</h3>
        <p>
            The dataset used in this project comes from a higher education institution and contains information about students 
            enrolled in different undergraduate degrees.
        </p>
        <div class="hint">
            Dataset source: Student Performance Data Dicoding. 
            <a href="https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv" style="color: {COLORS["primary"]};">Dataset</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Methodology
    st.markdown(f"""
    <div class="form-section">
        <h3>Methodology</h3>
        <ol>
            <li><strong>Data Preprocessing</strong>: Cleaning, normalization, and feature engineering to prepare the data.</li>
            <li><strong>Exploratory Data Analysis</strong>: Visualizing and understanding relationships between variables.</li>
            <li><strong>Clustering Analysis</strong>: Using MeanShift algorithm to identify natural groupings of students.</li>
            <li><strong>Classification Model</strong>: Building a predictive model to identify students at risk of dropping out.</li>
            <li><strong>Risk Assessment</strong>: Assigning risk levels (Low, Medium, High) based on cluster characteristics and model predictions.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information
    st.markdown(f"""
    <div class="form-section">
        <h3>Contact Information</h3>
        <p><strong>Name:</strong> Mohamad Rafli Agung Subekti</p>
        <p><strong>Email:</strong> <a href="mailto:raflisbk@gmail.com" style="color: {COLORS["primary"]};">raflisbk@gmail.com</a></p>
        <p><strong>ID Dicoding:</strong> raflisbk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical implementation
    st.markdown(f"""
    <div class="form-section">
        <h3>Technical Implementation</h3>
        <p>
            This dashboard is built using Streamlit, a Python library for creating web applications for data science and machine learning.
            The data visualizations are created using Plotly, and the machine learning models are implemented using scikit-learn.
        </p>
        <div class="hint">
            <strong>Tech Stack:</strong> Python, Streamlit, Pandas, NumPy, Plotly, Scikit-learn
        </div>
    </div>
    """, unsafe_allow_html=True)
