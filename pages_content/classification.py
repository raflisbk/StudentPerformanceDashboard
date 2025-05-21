import streamlit as st
import pandas as pd
import plotly.express as px
from utils.classification import predict_risk_level, get_recommendations
from utils.preprocessing import prepare_features_for_prediction

def show(df_with_risk_labels, COLORS):
    # Page title
    st.title("Dropout Risk Prediction")
    st.markdown("##### Predict the risk level for a student based on academic and demographic information")
    
    # Create two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # User input section
        st.subheader("Enter Student Information")
        
        # Create a form container with custom styling
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        
        # Academic information
        st.markdown(f"<h3>Academic Information</h3>", unsafe_allow_html=True)
        
        prev_qual_grade = st.slider(
            "Previous Qualification Grade (0-200)",
            min_value=0.0,
            max_value=200.0,
            value=140.0,
            step=1.0
        )
        
        admission_grade = st.slider(
            "Admission Grade (0-200)",
            min_value=0.0,
            max_value=200.0,
            value=130.0,
            step=1.0
        )
        
        units_enrolled = st.slider(
            "Curricular Units Enrolled (1st semester)",
            min_value=1,
            max_value=10,
            value=6,
            step=1
        )
        
        units_approved = st.slider(
            "Curricular Units Approved (1st semester)",
            min_value=0,
            max_value=units_enrolled,
            value=4,
            step=1
        )
        
        # Calculate passing ratio automatically
        passing_ratio = units_approved / units_enrolled if units_enrolled > 0 else 0
        
        # Display passing ratio with custom styling
        st.markdown(f"""
        <div style="
            background-color: {COLORS["surface_light"]};
            padding: 0.7rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0 1.5rem 0;
        ">
            <span style="color: {COLORS["text_secondary"]};">Passing Ratio:</span> 
            <span style="color: {COLORS["text"]}; font-weight: 600;">{passing_ratio:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demographic information
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown(f"<h3>Demographic Information</h3>", unsafe_allow_html=True)
        
        age = st.slider(
            "Age",
            min_value=16,
            max_value=60,
            value=22,
            step=1
        )
        
        gender = st.radio(
            "Gender",
            options=["Male", "Female"]
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            options=["Single", "Married", "Divorced", "Widower", "Facto union", "Legally separated"]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Socio-economic information
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown(f"<h3>Socio-Economic Information</h3>", unsafe_allow_html=True)
        
        # Using columns for radio buttons to save space
        soc1_col1, soc1_col2 = st.columns(2)
        with soc1_col1:
            scholarship = st.radio(
                "Scholarship Holder",
                options=["Yes", "No"]
            )
        
        with soc1_col2:
            debtor = st.radio(
                "Debtor",
                options=["Yes", "No"]
            )
        
        soc2_col1, soc2_col2 = st.columns(2)
        with soc2_col1:
            tuition_uptodate = st.radio(
                "Tuition Fees Up to Date",
                options=["Yes", "No"]
            )
        
        with soc2_col2:
            international = st.radio(
                "International Student",
                options=["Yes", "No"]
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        predict_button = st.button("Predict Risk Level", type="primary", use_container_width=True)
    
    with col2:
        # Prediction result section
        st.subheader("Prediction Result")
        
        if predict_button:
            # Prepare input data
            input_data = {
                'Age_at_enrollment': age,
                'Gender': gender,
                'Marital_status': marital_status,
                'Previous_qualification_grade': prev_qual_grade,
                'Admission_grade': admission_grade,
                'Curricular_units_1st_sem_enrolled': units_enrolled,
                'Curricular_units_1st_sem_approved': units_approved,
                'Passing_ratio_1st_sem': passing_ratio,
                'Scholarship_holder': scholarship,
                'Debtor': debtor,
                'Tuition_fees_up_to_date': tuition_uptodate,
                'International': international
            }
            
            # Map categorical values
            marital_status_mapping = {
                'Single': 1, 
                'Married': 2, 
                'Widower': 3, 
                'Divorced': 4, 
                'Facto union': 5, 
                'Legally separated': 6
            }
            
            input_data['Marital_status'] = marital_status_mapping.get(marital_status, 1)
            
            # Prepare features for prediction
            prepared_input = prepare_features_for_prediction(input_data)
            
            # Make prediction
            result = predict_risk_level(input_data)
            
            # Display prediction
            if 'error' in result:
                st.error(result['error'])
            else:
                # Display risk level with appropriate color
                risk_level = result['risk_level']
                risk_colors = {
                    'High': COLORS["charts"]["high_risk"],
                    'Medium': COLORS["charts"]["medium_risk"],
                    'Low': COLORS["charts"]["low_risk"]
                }
                
                color = risk_colors.get(risk_level)
                
                st.markdown(f"""
                <div class="risk-card {risk_level.lower()}">
                    <h2 style="color: {color};">Risk Level: {risk_level}</h2>
                    <p>Prediction Confidence: {result['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display key risk factors
                st.markdown("""
                <div class="form-section">
                    <h3>Key Risk Factors</h3>
                """, unsafe_allow_html=True)
                
                if result['key_factors']:
                    for factor in result['key_factors']:
                        st.markdown(f"- {factor}")
                else:
                    st.info("No specific risk factors identified.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("""
                <div class="form-section">
                    <h3>Recommendations</h3>
                """, unsafe_allow_html=True)
                
                recommendations = get_recommendations(risk_level)
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Display placeholder with styled message
            st.markdown(f"""
            <div style="
                background-color: {COLORS["surface_light"]};
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin-top: 2rem;
                border: 1px dashed {COLORS["border"]};
            ">
                <img src="https://cdn-icons-png.flaticon.com/512/1584/1584892.png" width="80" style="margin-bottom: 1rem; opacity: 0.5;">
                <p style="color: {COLORS["text_secondary"]}; margin-bottom: 0;">Enter student information and click "Predict Risk Level" to get a prediction.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample visualization
            st.markdown("<div class='form-section'>", unsafe_allow_html=True)
            st.markdown("<h3>Sample Risk Distribution</h3>", unsafe_allow_html=True)
            
            # Create sample data
            sample_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Proportion': [0.45, 0.35, 0.20]
            })
            
            # Create color map
            color_map = {
                'High': COLORS["charts"]["high_risk"],
                'Medium': COLORS["charts"]["medium_risk"],
                'Low': COLORS["charts"]["low_risk"]
            }
            
            # Create bar chart
            fig = px.bar(
                sample_data,
                x='Risk Level',
                y='Proportion',
                color='Risk Level',
                color_discrete_map=color_map,
                text_auto='.0%'
            )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS["text"]),
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
                ),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            # Remove lines around bars
            fig.update_traces(marker_line_width=0)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)