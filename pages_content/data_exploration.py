import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


def show(df, df_with_risk_labels, COLORS):
    # Page title
    st.title("Data Exploration")
    st.markdown("##### Understanding student characteristics and performance factors")
    
    # Create tabs for different categories of visualizations
    tab1, tab2, tab3 = st.tabs([
        "Student Demographics", 
        "Academic Performance", 
        "Socio-Economic Factors"
    ])
    
    # Common plot styling
    plot_style = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS["text"]),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        )
    )
    
    # Demographic visualizations - AT LEAST 5 VISUALIZATIONS
    with tab1:
        st.subheader("Student Demographics")
        
        # 1. Age distribution
        st.markdown("<h4>Age Distribution</h4>", unsafe_allow_html=True)
        
        if 'Age_at_enrollment' in df.columns:
            fig = px.histogram(
                df, 
                x='Age_at_enrollment',
                nbins=30,
                color_discrete_sequence=[COLORS["primary"]],
                opacity=0.8
            )
            
            # Add mean and median lines
            mean_age = df['Age_at_enrollment'].mean()
            median_age = df['Age_at_enrollment'].median()
            
            fig.add_vline(
                x=mean_age, 
                line_width=2, 
                line_dash="dash", 
                line_color=COLORS["secondary"],
                annotation_text=f"Mean: {mean_age:.1f}", 
                annotation_position="top right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            fig.add_vline(
                x=median_age, 
                line_width=2, 
                line_dash="dot", 
                line_color=COLORS["accent"],
                annotation_text=f"Median: {median_age:.1f}", 
                annotation_position="bottom right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 2. Gender distribution
        st.markdown("<h4>Gender Distribution</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        if 'Gender' in df.columns:
            # Process gender data
            if df['Gender'].dtype == 'bool' or (df['Gender'].dtype == 'int64' and set(df['Gender'].unique()).issubset({0, 1})):
                gender_mapping = {1: 'Male', 0: 'Female', True: 'Male', False: 'Female'}
                gender_data = df['Gender'].map(gender_mapping).value_counts().reset_index()
            else:
                gender_data = df['Gender'].value_counts().reset_index()
            
            gender_data.columns = ['Gender', 'Count']
            gender_data['Percentage'] = gender_data['Count'] / gender_data['Count'].sum() * 100
            
            # Create pie chart
            with col1:
                fig_pie = px.pie(
                    gender_data, 
                    values='Count', 
                    names='Gender', 
                    color='Gender',
                    color_discrete_map={'Male': COLORS["primary"], 'Female': COLORS["accent"]},
                    hole=0.5
                )
                
                fig_pie.update_traces(
                    textinfo='percent+label',
                    marker=dict(line=dict(color=COLORS["background"], width=2))
                )
                
                # Apply common styling
                fig_pie.update_layout(**plot_style)
                
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
            
            # Create bar chart
            with col2:
                fig_bar = px.bar(
                    gender_data,
                    x='Gender',
                    y='Count',
                    color='Gender',
                    text='Count',
                    color_discrete_map={'Male': COLORS["primary"], 'Female': COLORS["accent"]}
                )
                
                fig_bar.update_traces(
                    texttemplate='%{text}',
                    textposition='outside',
                    marker_line_width=0
                )
                
                # Apply common styling
                fig_bar.update_layout(**plot_style)
                
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        # 3. Marital status
        st.markdown("<h4>Marital Status</h4>", unsafe_allow_html=True)
        
        if 'Marital_status' in df.columns:
            # Process marital status data
            if df['Marital_status'].dtype == 'int64':
                marital_mapping = {
                    1: 'Single', 
                    2: 'Married', 
                    3: 'Widower', 
                    4: 'Divorced', 
                    5: 'Facto union', 
                    6: 'Legally separated'
                }
                marital_data = df['Marital_status'].map(marital_mapping).value_counts().reset_index()
            else:
                marital_data = df['Marital_status'].value_counts().reset_index()
            
            marital_data.columns = ['Status', 'Count']
            marital_data['Percentage'] = marital_data['Count'] / marital_data['Count'].sum() * 100
            
            # Create bar chart
            fig = px.bar(
                marital_data,
                x='Status',
                y='Count',
                color='Status',
                text='Count',
                color_discrete_sequence=px.colors.sequential.Magma_r
            )
            
            fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                marker_line_width=0
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 4. Age distribution by gender
        st.markdown("<h4>Age Distribution by Gender</h4>", unsafe_allow_html=True)
        
        if 'Age_at_enrollment' in df.columns and 'Gender' in df.columns:
            # Process gender data if needed
            gender_col = 'Gender'
            if df[gender_col].dtype == 'bool' or (df[gender_col].dtype == 'int64' and set(df[gender_col].unique()).issubset({0, 1})):
                gender_mapping = {1: 'Male', 0: 'Female', True: 'Male', False: 'Female'}
                df_temp = df.copy()
                df_temp['Gender'] = df_temp[gender_col].map(gender_mapping)
                gender_col = 'Gender'
            else:
                df_temp = df
            
            # Create box plot
            fig = px.box(
                df_temp, 
                x=gender_col, 
                y='Age_at_enrollment',
                color=gender_col,
                color_discrete_map={'Male': COLORS["primary"], 'Female': COLORS["accent"]},
                points="all",
                notched=True
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 5. Age groups
        st.markdown("<h4>Age Groups</h4>", unsafe_allow_html=True)
        
        if 'Age_at_enrollment' in df.columns:
            # Create age groups
            age_bins = [15, 20, 25, 30, 35, 40, 100]
            age_labels = ['16-20', '21-25', '26-30', '31-35', '36-40', '40+']
            
            df_temp = df.copy()
            df_temp['Age_Group'] = pd.cut(df_temp['Age_at_enrollment'], bins=age_bins, labels=age_labels)
            
            age_counts = df_temp['Age_Group'].value_counts().sort_index().reset_index()
            age_counts.columns = ['Age_Group', 'Count']
            
            # Create horizontal bar chart
            fig = px.bar(
                age_counts,
                y='Age_Group',
                x='Count',
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Viridis,
                text='Count'
            )
            
            fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                marker_line_width=0
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 6. Age distribution by marital status
        st.markdown("<h4>Age Distribution by Marital Status</h4>", unsafe_allow_html=True)
        
        if 'Age_at_enrollment' in df.columns and 'Marital_status' in df.columns:
            # Process marital status data if needed
            if df['Marital_status'].dtype == 'int64':
                marital_mapping = {
                    1: 'Single', 
                    2: 'Married', 
                    3: 'Widower', 
                    4: 'Divorced', 
                    5: 'Facto union', 
                    6: 'Legally separated'
                }
                df_temp = df.copy()
                df_temp['Marital_Status'] = df_temp['Marital_status'].map(marital_mapping)
            else:
                df_temp = df.copy()
                df_temp['Marital_Status'] = df_temp['Marital_status']
            
            # Create violin plot
            fig = px.violin(
                df_temp, 
                x='Marital_Status', 
                y='Age_at_enrollment',
                color='Marital_Status',
                box=True,
                points="all",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Performance metrics visualizations - AT LEAST 5 VISUALIZATIONS
    with tab2:
        st.subheader("Academic Performance Metrics")
        
        # 1. Admission grade distribution
        st.markdown("<h4>Admission Grade Distribution</h4>", unsafe_allow_html=True)
        
        if 'Admission_grade' in df.columns:
            fig = px.histogram(
                df, 
                x='Admission_grade',
                nbins=30,
                color_discrete_sequence=[COLORS["accent"]],
                opacity=0.8
            )
            
            # Add mean and median lines
            mean_grade = df['Admission_grade'].mean()
            median_grade = df['Admission_grade'].median()
            
            fig.add_vline(
                x=mean_grade, 
                line_width=2, 
                line_dash="dash", 
                line_color=COLORS["primary"],
                annotation_text=f"Mean: {mean_grade:.1f}", 
                annotation_position="top right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            fig.add_vline(
                x=median_grade, 
                line_width=2, 
                line_dash="dot", 
                line_color=COLORS["secondary"],
                annotation_text=f"Median: {median_grade:.1f}", 
                annotation_position="bottom right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 2. Passing ratio distribution
        st.markdown("<h4>Passing Ratio Distribution</h4>", unsafe_allow_html=True)
        
        if 'Passing_ratio_1st_sem' in df.columns:
            fig = px.histogram(
                df, 
                x='Passing_ratio_1st_sem',
                nbins=20,
                color_discrete_sequence=[COLORS["secondary"]],
                opacity=0.8
            )
            
            # Add mean and median lines
            mean_ratio = df['Passing_ratio_1st_sem'].mean()
            median_ratio = df['Passing_ratio_1st_sem'].median()
            
            fig.add_vline(
                x=mean_ratio, 
                line_width=2, 
                line_dash="dash", 
                line_color=COLORS["primary"],
                annotation_text=f"Mean: {mean_ratio:.2f}", 
                annotation_position="top right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            fig.add_vline(
                x=median_ratio, 
                line_width=2, 
                line_dash="dot", 
                line_color=COLORS["accent"],
                annotation_text=f"Median: {median_ratio:.2f}", 
                annotation_position="bottom right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 3. Units enrolled vs approved
        st.markdown("<h4>Units Enrolled vs Approved (1st Semester)</h4>", unsafe_allow_html=True)
        
        if 'Curricular_units_1st_sem_enrolled' in df.columns and 'Curricular_units_1st_sem_approved' in df.columns:
            # Count combinations
            unit_counts = df.groupby(['Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved']).size().reset_index(name='count')
            
            fig = px.scatter(
                unit_counts,
                x='Curricular_units_1st_sem_enrolled',
                y='Curricular_units_1st_sem_approved',
                size='count',
                color='count',
                color_continuous_scale='Viridis',
                labels={
                    'Curricular_units_1st_sem_enrolled': 'Units Enrolled',
                    'Curricular_units_1st_sem_approved': 'Units Approved',
                    'count': 'Students'
                }
            )
            
            # Add perfect passing line
            max_units = max(df['Curricular_units_1st_sem_enrolled'].max(), df['Curricular_units_1st_sem_approved'].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_units],
                    y=[0, max_units],
                    mode='lines',
                    name='Perfect Passing',
                    line=dict(color=COLORS["accent"], width=2, dash='dash')
                )
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 4. Previous qualification vs admission grade
        st.markdown("<h4>Previous Qualification vs Admission Grade</h4>", unsafe_allow_html=True)
        
        if 'Previous_qualification_grade' in df.columns and 'Admission_grade' in df.columns:
            fig = px.scatter(
                df,
                x='Previous_qualification_grade',
                y='Admission_grade',
                color='Passing_ratio_1st_sem' if 'Passing_ratio_1st_sem' in df.columns else None,
                color_continuous_scale='Viridis',
                opacity=0.7,
                labels={
                    'Previous_qualification_grade': 'Previous Qualification Grade',
                    'Admission_grade': 'Admission Grade',
                    'Passing_ratio_1st_sem': 'Passing Ratio'
                }
            )
            
            # Add reference line
            max_grade = max(df['Previous_qualification_grade'].max(), df['Admission_grade'].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_grade],
                    y=[0, max_grade],
                    mode='lines',
                    name='Equal Grades',
                    line=dict(color=COLORS["accent"], width=2, dash='dash')
                )
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 5. Grade improvement distribution
        st.markdown("<h4>Grade Improvement (Admission - Previous)</h4>", unsafe_allow_html=True)
        
        if 'Previous_qualification_grade' in df.columns and 'Admission_grade' in df.columns:
            df_temp = df.copy()
            df_temp['Grade_Improvement'] = df_temp['Admission_grade'] - df_temp['Previous_qualification_grade']
            
            fig = px.histogram(
                df_temp,
                x='Grade_Improvement',
                nbins=30,
                color_discrete_sequence=[COLORS["primary"]],
                labels={'Grade_Improvement': 'Grade Improvement'}
            )
            
            # Add mean and median lines
            mean_improvement = df_temp['Grade_Improvement'].mean()
            median_improvement = df_temp['Grade_Improvement'].median()
            
            fig.add_vline(
                x=mean_improvement, 
                line_width=2, 
                line_dash="dash", 
                line_color=COLORS["secondary"],
                annotation_text=f"Mean: {mean_improvement:.1f}", 
                annotation_position="top right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            fig.add_vline(
                x=median_improvement, 
                line_width=2, 
                line_dash="dot", 
                line_color=COLORS["accent"],
                annotation_text=f"Median: {median_improvement:.1f}", 
                annotation_position="bottom right",
                annotation_font=dict(color=COLORS["text"])
            )
            
            # Add zero reference line
            fig.add_vline(
                x=0, 
                line_width=2, 
                line_dash="solid", 
                line_color='rgba(255,255,255,0.3)'
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 6. Passing ratio by gender
        st.markdown("<h4>Passing Ratio by Gender</h4>", unsafe_allow_html=True)
        
        if 'Passing_ratio_1st_sem' in df.columns and 'Gender' in df.columns:
            # Process gender data if needed
            gender_col = 'Gender'
            if df[gender_col].dtype == 'bool' or (df[gender_col].dtype == 'int64' and set(df[gender_col].unique()).issubset({0, 1})):
                gender_mapping = {1: 'Male', 0: 'Female', True: 'Male', False: 'Female'}
                df_temp = df.copy()
                df_temp['Gender'] = df_temp[gender_col].map(gender_mapping)
                gender_col = 'Gender'
            else:
                df_temp = df
            
            # Create box plot
            fig = px.box(
                df_temp, 
                x=gender_col, 
                y='Passing_ratio_1st_sem',
                color=gender_col,
                color_discrete_map={'Male': COLORS["primary"], 'Female': COLORS["accent"]},
                notched=True,
                points="all"
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 7. Grade heatmap by units
        st.markdown("<h4>Admission Grade vs Units Approved</h4>", unsafe_allow_html=True)
        
        if 'Admission_grade' in df.columns and 'Curricular_units_1st_sem_approved' in df.columns:
            # Create grade bins
            grade_bins = [0, 100, 120, 140, 160, 180, 200]
            grade_labels = ['0-100', '101-120', '121-140', '141-160', '161-180', '181-200']
            
            df_temp = df.copy()
            df_temp['Grade_Range'] = pd.cut(df_temp['Admission_grade'], bins=grade_bins, labels=grade_labels)
            
            # Create heatmap data
            heatmap_data = pd.crosstab(
                df_temp['Grade_Range'], 
                df_temp['Curricular_units_1st_sem_approved']
            )
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale='Viridis',
                labels=dict(x="Units Approved", y="Admission Grade Range", color="Count"),
                text_auto=True
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Socioeconomic factors visualizations - AT LEAST 5 VISUALIZATIONS
    with tab3:
        st.subheader("Socio-Economic Factors")
        
        # 1-2. Scholarship and Tuition status
        col1, col2 = st.columns(2)
        
        # 1. Scholarship status
        with col1:
            st.markdown("<h4>Scholarship Status</h4>", unsafe_allow_html=True)
            
            if 'Scholarship_holder' in df.columns:
                # Process scholarship data
                if df['Scholarship_holder'].dtype == 'bool' or (df['Scholarship_holder'].dtype == 'int64' and set(df['Scholarship_holder'].unique()).issubset({0, 1})):
                    scholarship_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                    scholarship_data = df['Scholarship_holder'].map(scholarship_mapping).value_counts().reset_index()
                else:
                    scholarship_data = df['Scholarship_holder'].value_counts().reset_index()
                
                scholarship_data.columns = ['Status', 'Count']
                scholarship_data['Percentage'] = scholarship_data['Count'] / scholarship_data['Count'].sum() * 100
                
                # Create pie chart
                fig = px.pie(
                    scholarship_data,
                    values='Count',
                    names='Status',
                    color='Status',
                    color_discrete_map={'Yes': COLORS["charts"]["low_risk"], 'No': COLORS["charts"]["high_risk"]},
                    hole=0.5
                )
                
                fig.update_traces(
                    textinfo='percent+label',
                    marker=dict(line=dict(color=COLORS["background"], width=2))
                )
                
                # Apply common styling
                fig.update_layout(**plot_style)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 2. Tuition payment status
        with col2:
            st.markdown("<h4>Tuition Payment Status</h4>", unsafe_allow_html=True)
            
            if 'Tuition_fees_up_to_date' in df.columns:
                # Process tuition data
                if df['Tuition_fees_up_to_date'].dtype == 'bool' or (df['Tuition_fees_up_to_date'].dtype == 'int64' and set(df['Tuition_fees_up_to_date'].unique()).issubset({0, 1})):
                    tuition_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                    tuition_data = df['Tuition_fees_up_to_date'].map(tuition_mapping).value_counts().reset_index()
                else:
                    tuition_data = df['Tuition_fees_up_to_date'].value_counts().reset_index()
                
                tuition_data.columns = ['Status', 'Count']
                tuition_data['Percentage'] = tuition_data['Count'] / tuition_data['Count'].sum() * 100
                
                # Create pie chart
                fig = px.pie(
                    tuition_data,
                    values='Count',
                    names='Status',
                    color='Status',
                    color_discrete_map={'Yes': COLORS["charts"]["low_risk"], 'No': COLORS["charts"]["high_risk"]},
                    hole=0.5
                )
                
                fig.update_traces(
                    textinfo='percent+label',
                    marker=dict(line=dict(color=COLORS["background"], width=2))
                )
                
                # Apply common styling
                fig.update_layout(**plot_style)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 3. International student status
        st.markdown("<h4>International Student Status</h4>", unsafe_allow_html=True)
        
        if 'International' in df.columns:
            # Process international data
            if df['International'].dtype == 'bool' or (df['International'].dtype == 'int64' and set(df['International'].unique()).issubset({0, 1})):
                international_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                international_data = df['International'].map(international_mapping).value_counts().reset_index()
            else:
                international_data = df['International'].value_counts().reset_index()
            
            international_data.columns = ['Status', 'Count']
            international_data['Percentage'] = international_data['Count'] / international_data['Count'].sum() * 100
            
            # Create pie chart
            fig = px.pie(
                international_data,
                values='Count',
                names='Status',
                color='Status',
                color_discrete_map={'Yes': COLORS["primary"], 'No': COLORS["primary_light"]},
                hole=0.5
            )
            
            fig.update_traces(
                textinfo='percent+label',
                marker=dict(line=dict(color=COLORS["background"], width=2))
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 4. Scholarship status vs Passing ratio
        st.markdown("<h4>Scholarship vs Performance</h4>", unsafe_allow_html=True)
        
        if 'Scholarship_holder' in df.columns and 'Passing_ratio_1st_sem' in df.columns:
            # Process scholarship data
            if df['Scholarship_holder'].dtype == 'bool' or (df['Scholarship_holder'].dtype == 'int64' and set(df['Scholarship_holder'].unique()).issubset({0, 1})):
                scholarship_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp = df.copy()
                df_temp['Scholarship'] = df_temp['Scholarship_holder'].map(scholarship_mapping)
            else:
                df_temp = df.copy()
                df_temp['Scholarship'] = df_temp['Scholarship_holder']
            
            # Create violin plot
            fig = px.violin(
                df_temp,
                x='Scholarship',
                y='Passing_ratio_1st_sem',
                color='Scholarship',
                color_discrete_map={'Yes': COLORS["charts"]["low_risk"], 'No': COLORS["charts"]["high_risk"]},
                box=True,
                points="all"
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 5. Tuition payment status vs Admission grade
        st.markdown("<h4>Tuition Payment vs Admission Grade</h4>", unsafe_allow_html=True)
        
        if 'Tuition_fees_up_to_date' in df.columns and 'Admission_grade' in df.columns:
            # Process tuition data
            if df['Tuition_fees_up_to_date'].dtype == 'bool' or (df['Tuition_fees_up_to_date'].dtype == 'int64' and set(df['Tuition_fees_up_to_date'].unique()).issubset({0, 1})):
                tuition_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp = df.copy()
                df_temp['Tuition_Up_To_Date'] = df_temp['Tuition_fees_up_to_date'].map(tuition_mapping)
            else:
                df_temp = df.copy()
                df_temp['Tuition_Up_To_Date'] = df_temp['Tuition_fees_up_to_date']
            
            # Create box plot
            fig = px.box(
                df_temp,
                x='Tuition_Up_To_Date',
                y='Admission_grade',
                color='Tuition_Up_To_Date',
                color_discrete_map={'Yes': COLORS["charts"]["low_risk"], 'No': COLORS["charts"]["high_risk"]},
                notched=True,
                points="all"
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 6. Combined socioeconomic factors
        st.markdown("<h4>Combined Socioeconomic Factors</h4>", unsafe_allow_html=True)
        
        if all(col in df.columns for col in ['Scholarship_holder', 'Tuition_fees_up_to_date', 'International']):
            # Process data
            df_temp = df.copy()
            
            # Process scholarship data
            if df['Scholarship_holder'].dtype == 'bool' or (df['Scholarship_holder'].dtype == 'int64' and set(df['Scholarship_holder'].unique()).issubset({0, 1})):
                scholarship_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Scholarship'] = df_temp['Scholarship_holder'].map(scholarship_mapping)
            else:
                df_temp['Scholarship'] = df_temp['Scholarship_holder']
            
            # Process tuition data
            if df['Tuition_fees_up_to_date'].dtype == 'bool' or (df['Tuition_fees_up_to_date'].dtype == 'int64' and set(df['Tuition_fees_up_to_date'].unique()).issubset({0, 1})):
                tuition_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Tuition'] = df_temp['Tuition_fees_up_to_date'].map(tuition_mapping)
            else:
                df_temp['Tuition'] = df_temp['Tuition_fees_up_to_date']
            
            # Process international data
            if df['International'].dtype == 'bool' or (df['International'].dtype == 'int64' and set(df['International'].unique()).issubset({0, 1})):
                international_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['International_Student'] = df_temp['International'].map(international_mapping)
            else:
                df_temp['International_Student'] = df_temp['International']
            
            # Count combinations
            factor_counts = df_temp.groupby(['Scholarship', 'Tuition', 'International_Student']).size().reset_index(name='Count')
            
            # Sort by count
            factor_counts = factor_counts.sort_values('Count', ascending=False)
            
            # Create grouped bar chart
            fig = px.bar(
                factor_counts,
                x='Scholarship',
                y='Count',
                color='Tuition',
                barmode='group',
                facet_col='International_Student',
                text='Count',
                color_discrete_map={'Yes': COLORS["charts"]["low_risk"], 'No': COLORS["charts"]["high_risk"]}
            )
            
            fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                marker_line_width=0
            )
            
            # Apply common styling
            fig.update_layout(
                **plot_style,
                title='Combined Socioeconomic Factors'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 7. Socioeconomic impact on passing ratio
        st.markdown("<h4>Socioeconomic Impact on Academic Performance</h4>", unsafe_allow_html=True)
        
        if all(col in df.columns for col in ['Scholarship_holder', 'Tuition_fees_up_to_date', 'Passing_ratio_1st_sem']):
            # Process data
            df_temp = df.copy()
            
            # Process scholarship data
            if df['Scholarship_holder'].dtype == 'bool' or (df['Scholarship_holder'].dtype == 'int64' and set(df['Scholarship_holder'].unique()).issubset({0, 1})):
                scholarship_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Scholarship'] = df_temp['Scholarship_holder'].map(scholarship_mapping)
            else:
                df_temp['Scholarship'] = df_temp['Scholarship_holder']
            
            # Process tuition data
            if df['Tuition_fees_up_to_date'].dtype == 'bool' or (df['Tuition_fees_up_to_date'].dtype == 'int64' and set(df['Tuition_fees_up_to_date'].unique()).issubset({0, 1})):
                tuition_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Tuition'] = df_temp['Tuition_fees_up_to_date'].map(tuition_mapping)
            else:
                df_temp['Tuition'] = df_temp['Tuition_fees_up_to_date']
            
            # Create combined column
            df_temp['Scholarship'] = df_temp['Scholarship'].astype(str)
            df_temp['Tuition'] = df_temp['Tuition'].astype(str)
            df_temp['Socioeconomic_Status'] = df_temp['Scholarship'] + ' / ' + df_temp['Tuition']
            
            # Average passing ratio by combination
            passing_avg = df_temp.groupby('Socioeconomic_Status')['Passing_ratio_1st_sem'].agg(['mean', 'count']).reset_index()
            passing_avg.columns = ['Status', 'Avg_Passing_Ratio', 'Count']
            
            # Sort by average passing ratio
            passing_avg = passing_avg.sort_values('Avg_Passing_Ratio', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                passing_avg,
                x='Status',
                y='Avg_Passing_Ratio',
                color='Avg_Passing_Ratio',
                color_continuous_scale='Viridis',
                text='Avg_Passing_Ratio',
                hover_data=['Count']
            )
            
            fig.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside',
                marker_line_width=0
            )
            
            # Apply common styling
            fig.update_layout(**plot_style)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 8. Relationship between international status and scholarship
        st.markdown("<h4>International Status vs Scholarship</h4>", unsafe_allow_html=True)
        
        if 'International' in df.columns and 'Scholarship_holder' in df.columns:
            # Process data
            df_temp = df.copy()
            
            # Process scholarship data
            if df['Scholarship_holder'].dtype == 'bool' or (df['Scholarship_holder'].dtype == 'int64' and set(df['Scholarship_holder'].unique()).issubset({0, 1})):
                scholarship_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Scholarship'] = df_temp['Scholarship_holder'].map(scholarship_mapping)
            else:
                df_temp['Scholarship'] = df_temp['Scholarship_holder']
            
            # Process international data
            if df['International'].dtype == 'bool' or (df['International'].dtype == 'int64' and set(df['International'].unique()).issubset({0, 1})):
                international_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['International_Student'] = df_temp['International'].map(international_mapping)
            else:
                df_temp['International_Student'] = df_temp['International']
            
            # Create cross-tabulation
            cross_tab = pd.crosstab(
                df_temp['International_Student'],
                df_temp['Scholarship'],
                normalize='index'
            ) * 100
            
            # Create grouped bar chart
            fig = go.Figure()
            
            for col in cross_tab.columns:
                fig.add_trace(go.Bar(
                    x=cross_tab.index,
                    y=cross_tab[col],
                    name=col,
                    text=[f"{val:.1f}%" for val in cross_tab[col]],
                    textposition='auto',
                    marker_color=COLORS["charts"]["low_risk"] if col == 'Yes' else COLORS["charts"]["high_risk"]
                ))
            
            # Apply styling
            fig.update_layout(
                title="International Students vs Scholarship Ratio",
                xaxis_title="International Student",
                yaxis_title="Percentage",
                barmode='group',
                **plot_style
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 9. Risk level by socioeconomic factors
        st.markdown("<h4>Risk Level by Socioeconomic Factors</h4>", unsafe_allow_html=True)
        
        if all(col in df_with_risk_labels.columns for col in ['Risk_Category', 'Scholarship_holder', 'Tuition_fees_up_to_date']):
            # Process data
            df_temp = df_with_risk_labels.copy()
            
            # Process scholarship data
            if df_temp['Scholarship_holder'].dtype == 'bool' or (df_temp['Scholarship_holder'].dtype == 'int64' and set(df_temp['Scholarship_holder'].unique()).issubset({0, 1})):
                scholarship_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Scholarship'] = df_temp['Scholarship_holder'].map(scholarship_mapping)
            else:
                df_temp['Scholarship'] = df_temp['Scholarship_holder']
            
            # Process tuition data
            if df_temp['Tuition_fees_up_to_date'].dtype == 'bool' or (df_temp['Tuition_fees_up_to_date'].dtype == 'int64' and set(df_temp['Tuition_fees_up_to_date'].unique()).issubset({0, 1})):
                tuition_mapping = {1: 'Yes', 0: 'No', True: 'Yes', False: 'No'}
                df_temp['Tuition'] = df_temp['Tuition_fees_up_to_date'].map(tuition_mapping)
            else:
                df_temp['Tuition'] = df_temp['Tuition_fees_up_to_date']
            
            # Create combined column
            df_temp['Scholarship'] = df_temp['Scholarship'].astype(str)
            df_temp['Tuition'] = df_temp['Tuition'].astype(str)
            df_temp['Socioeconomic_Status'] = df_temp['Scholarship'] + ' / ' + df_temp['Tuition']
            
            # Cross-tabulation
            risk_by_socio = pd.crosstab(
                df_temp['Socioeconomic_Status'],
                df_temp['Risk_Category'],
                normalize='index'
            ) * 100
            
            # Check if risk categories exist
            if all(cat in risk_by_socio.columns for cat in ['High', 'Medium', 'Low']):
                # Reorder columns
                risk_by_socio = risk_by_socio[['High', 'Medium', 'Low']]
            
            # Convert to long format
            risk_long = risk_by_socio.reset_index().melt(
                id_vars=['Socioeconomic_Status'],
                var_name='Risk_Category',
                value_name='Percentage'
            )
            
            # Create stacked bar chart
            fig = px.bar(
                risk_long,
                x='Socioeconomic_Status',
                y='Percentage',
                color='Risk_Category',
                barmode='stack',
                text='Percentage',
                color_discrete_map={
                    'High': COLORS["charts"]["high_risk"],
                    'Medium': COLORS["charts"]["medium_risk"],
                    'Low': COLORS["charts"]["low_risk"]
                }
            )
            
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='inside',
                marker_line_width=0
            )
            
            # Apply styling
            fig.update_layout(
                title="Risk Level Distribution by Socioeconomic Status",
                xaxis_title="Socioeconomic Status (Scholarship / Tuition)",
                yaxis_title="Percentage",
                **plot_style
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Data table with filters
    with st.expander("View Data Table", expanded=False):
        # Select columns to display
        all_columns = df.columns.tolist()
        
        # Default columns to show
        default_columns = [
            'Age_at_enrollment', 'Gender', 'Marital_status',
            'Previous_qualification_grade', 'Admission_grade',
            'Scholarship_holder', 'Tuition_fees_up_to_date',
            'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved'
        ]
        
        # Only include columns that actually exist in the dataframe
        default_columns = [col for col in default_columns if col in all_columns]
        
        # Let user select columns
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=default_columns
        )
        
        # Filter data based on selected columns
        if selected_columns:
            filtered_df = df[selected_columns]
            
            # Apply custom styling to dataframe
            st.markdown(f"""
            <style>
            .dataframe-container {{
                background-color: {COLORS["surface_light"]};
                border-radius: 8px;
                padding: 1rem;
                overflow: auto;
                max-height: 500px;
            }}
            
            .dataframe {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            .dataframe th {{
                background-color: {COLORS["surface"]};
                color: {COLORS["text"]};
                text-align: left;
                padding: 0.75rem;
                border-bottom: 1px solid {COLORS["border"]};
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            
            .dataframe td {{
                padding: 0.5rem 0.75rem;
                border-bottom: 1px solid {COLORS["border"]};
                color: {COLORS["text_secondary"]};
            }}
            
            .dataframe tr:hover td {{
                background-color: {COLORS["surface"]};
            }}
            </style>
            
            <div class="dataframe-container">
            """, unsafe_allow_html=True)
            
            st.dataframe(filtered_df, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="student_data.csv",
                mime="text/csv",
                key="download-csv",
                help="Download the filtered data as a CSV file"
            )
        else:
            st.info("Please select at least one column to display.")