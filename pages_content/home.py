import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show(df, COLORS):
    # Page title
    st.title("Welcome to Student Performance Dashboard")
    st.markdown("##### A modern analytics tool for monitoring student performance and predicting dropout risk")
    
    # Description
    with st.container():
        st.markdown("""
        This dashboard provides comprehensive analysis of student performance data from a higher education institution. 
        It includes exploratory data analysis, clustering analysis to identify patterns among students, 
        and a predictive model to identify students at risk of dropping out.
        """)
    
    # Key Features
    st.subheader("Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    # Data Exploration Card
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Data Exploration</h3>
            <p>Visualize and understand student demographics, academic performance, and socio-economic factors.</p>
            <div class="bg-icon">📊</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Button for navigation
        if st.button("Explore Data", key="explore_data_btn", use_container_width=True):
            st.session_state.page = "Data Exploration"
            st.rerun()
    
    # Clustering Card
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Clustering Analysis</h3>
            <p>Discover natural groupings of students based on their characteristics and identify risk patterns.</p>
            <div class="bg-icon">🔍</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Button for navigation
        if st.button("View Clusters", key="cluster_btn", use_container_width=True):
            st.session_state.page = "Clustering Analysis"
            st.rerun()
    
    # Risk Prediction Card
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>Risk Prediction</h3>
            <p>Predict students' risk level for dropping out based on their profile and academic performance.</p>
            <div class="bg-icon">⚠️</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Button for navigation
        if st.button("Predict Risk", key="risk_pred_btn", use_container_width=True):
            st.session_state.page = "Risk Prediction"
            st.rerun()
    
    # Key metrics dashboard
    st.subheader("Key Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        total_students = len(df)
        st.metric(label="Total Students", value=f"{total_students}")
    
    with metrics_col2:
        # Perbaikan untuk Dropout Rate
        if 'Status_Dropout' in df.columns:
            if df['Status_Dropout'].dtype == 'object':
                # Jika kolom Status_Dropout bertipe objek (string), konversi 'True'/'False' ke bool
                dropout_rate = df['Status_Dropout'].map({'True': True, 'False': False}).mean() * 100
            elif df['Status_Dropout'].dtype == 'bool':
                # Jika sudah boolean, langsung gunakan mean
                dropout_rate = df['Status_Dropout'].mean() * 100
            else:
                # Assume it's numeric (0/1)
                dropout_rate = df['Status_Dropout'].mean() * 100
            
            # Format dengan satu tempat desimal
            st.metric(label="Dropout Rate", value=f"{dropout_rate:.1f}%")
        elif 'Status' in df.columns:
            # Jika menggunakan kolom 'Status', hitung dropout rate berdasarkan nilai 'Dropout'
            dropout_count = (df['Status'] == 'Dropout').sum()
            dropout_rate = (dropout_count / len(df)) * 100
            st.metric(label="Dropout Rate", value=f"{dropout_rate:.1f}%")
        else:
            st.metric(label="Dropout Rate", value="N/A")
    
    with metrics_col3:
        # Perbaikan untuk Average Passing Ratio
        if 'Passing_ratio_1st_sem' in df.columns:
            # Konversi ke float untuk memastikan kalkulasi benar
            try:
                avg_passing_ratio = df['Passing_ratio_1st_sem'].astype(float).mean() * 100
                st.metric(label="Avg. Passing Ratio", value=f"{avg_passing_ratio:.1f}%")
            except:
                # Fallback jika ada masalah konversi
                st.metric(label="Avg. Passing Ratio", value="Error")
        elif 'Curricular_units_1st_sem_approved' in df.columns and 'Curricular_units_1st_sem_enrolled' in df.columns:
            # Hitung passing ratio dari unit yang disetujui dan didaftarkan
            # Hindari pembagian dengan nol
            df_valid = df[df['Curricular_units_1st_sem_enrolled'] > 0].copy()
            if len(df_valid) > 0:
                df_valid['calc_ratio'] = df_valid['Curricular_units_1st_sem_approved'] / df_valid['Curricular_units_1st_sem_enrolled']
                avg_passing_ratio = df_valid['calc_ratio'].mean() * 100
                st.metric(label="Avg. Passing Ratio", value=f"{avg_passing_ratio:.1f}%")
            else:
                st.metric(label="Avg. Passing Ratio", value="N/A")
        else:
            st.metric(label="Avg. Passing Ratio", value="N/A")
    
    with metrics_col4:
        # Perbaikan untuk Scholarship Rate
        if 'Scholarship_holder' in df.columns:
            # Coba beberapa format yang mungkin untuk kolom Scholarship_holder
            if df['Scholarship_holder'].dtype == 'object':
                # Jika string 'Yes'/'No' atau 'True'/'False'
                if set(df['Scholarship_holder'].unique()).issubset({'Yes', 'No'}):
                    scholarship_rate = df['Scholarship_holder'].map({'Yes': True, 'No': False}).mean() * 100
                else:
                    scholarship_rate = df['Scholarship_holder'].map({'True': True, 'False': False}).mean() * 100
            elif df['Scholarship_holder'].dtype == 'bool':
                # Jika boolean
                scholarship_rate = df['Scholarship_holder'].mean() * 100
            else:
                # Jika numeric (0/1)
                scholarship_rate = df['Scholarship_holder'].mean() * 100
            
            st.metric(label="Scholarship Rate", value=f"{scholarship_rate:.1f}%")
        else:
            st.metric(label="Scholarship Rate", value="N/A")
    
    # Risk distribution chart
    if 'Risk_Category' in df.columns:
        st.subheader("Risk Level Distribution")
        
        # Count risk categories
        risk_counts = df['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Category', 'Count']
        
        # Order categories correctly
        if set(risk_counts['Risk_Category']) == {'High', 'Medium', 'Low'}:
            risk_counts['Order'] = risk_counts['Risk_Category'].map({'High': 0, 'Medium': 1, 'Low': 2})
            risk_counts = risk_counts.sort_values('Order').drop('Order', axis=1)
        
        # Create custom colors mapping
        color_map = {
            'High': COLORS["charts"]["high_risk"],
            'Medium': COLORS["charts"]["medium_risk"],
            'Low': COLORS["charts"]["low_risk"]
        }
        
        # Create bar chart
        fig = px.bar(
            risk_counts, 
            x='Risk_Category', 
            y='Count',
            color='Risk_Category',
            color_discrete_map=color_map,
            text='Count'
        )
        
        # Customize layout
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS["text"]),
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            xaxis=dict(
                title=None,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title=None,
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False
            )
        )
        
        # Add hover effect
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
            marker_line_width=0,
            texttemplate='%{y}',
            textposition='inside'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})