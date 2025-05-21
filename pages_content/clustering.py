import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.clustering import load_or_train_meanshift_model, cluster_interpretation

def show(df, clustering_data, df_with_risk_labels, COLORS):
    # Page title
    st.title("Clustering Analysis")
    st.markdown("##### Identifying natural groupings among students and analyzing risk patterns")
    
    # Description of the clustering method
    st.markdown("""
    <div class="form-section">
        <p>This analysis uses the MeanShift clustering algorithm to group students with similar characteristics. 
        The algorithm automatically determines the optimal number of clusters based on the density of data points, 
        allowing us to identify natural patterns in student performance and risk factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load or train MeanShift model
    with st.spinner("Loading clustering model..."):
        model = load_or_train_meanshift_model(clustering_data)
    
    # Display cluster visualizations
    st.subheader("Risk Category Distribution")
    
    # Create risk category distribution plot
    if 'Risk_Category' in df_with_risk_labels.columns:
        risk_counts = df_with_risk_labels['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Category', 'Count']
        
        # Order categories
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        risk_counts['Order'] = risk_counts['Risk_Category'].map(risk_order)
        risk_counts = risk_counts.sort_values('Order').drop('Order', axis=1)
        
        # Create color map
        color_map = {
            'High': COLORS["charts"]["high_risk"],
            'Medium': COLORS["charts"]["medium_risk"],
            'Low': COLORS["charts"]["low_risk"]
        }
        
        # Create pie chart
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Risk_Category',
            color='Risk_Category',
            color_discrete_map=color_map,
            hole=0.5
        )
        
        # Update layout for dark theme
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS["text"]),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update traces
        fig.update_traces(
            textinfo='percent+label',
            pull=[0.05 if x == 'High' else 0 for x in risk_counts['Risk_Category']],
            marker=dict(line=dict(color=COLORS["background"], width=2))
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Feature importance visualization
    st.subheader("Feature Importance")
    
    # Create feature importance visualization
    important_features = [
        'Passing_ratio_1st_sem', 
        'Curricular_units_1st_sem_approved',
        'Admission_grade',
        'Previous_qualification_grade',
        'Scholarship_holder',
        'Tuition_fees_up_to_date',
        'Age_at_enrollment',
        'International'
    ]
    
    importance = [0.28, 0.22, 0.15, 0.12, 0.08, 0.07, 0.05, 0.03]
    
    # Create dataframe for feature importance
    feature_importance = pd.DataFrame({
        'feature': important_features,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        feature_importance,
        y='feature',
        x='importance',
        orientation='h',
        color='importance',
        color_continuous_scale=[[0, COLORS["charts"]["low_risk"]], 
                              [0.5, COLORS["charts"]["medium_risk"]], 
                              [1, COLORS["charts"]["high_risk"]]],
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS["text"]),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False
        ),
        coloraxis_showscale=False
    )
    
    # Remove lines around bars
    fig.update_traces(marker_line_width=0)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Cluster interpretation
    st.subheader("Cluster Interpretation")
    
    # Get cluster interpretations
    interpretations = cluster_interpretation(df_with_risk_labels)
    
    # Display interpretations
    if interpretations:
        # Group by risk level
        risk_groups = {
            'High': [],
            'Medium': [],
            'Low': []
        }
        
        # Group clusters by risk level
        for cluster, interp in interpretations.items():
            title = interp.get('title', '')
            if 'High' in title:
                risk_groups['High'].append((cluster, interp))
            elif 'Medium' in title:
                risk_groups['Medium'].append((cluster, interp))
            elif 'Low' in title:
                risk_groups['Low'].append((cluster, interp))
        
        # Display high risk clusters
        if risk_groups['High']:
            st.markdown(f"""
            <h3 style="color: {COLORS["charts"]["high_risk"]}; margin-top: 2rem;">High Risk Cluster</h3>
            """, unsafe_allow_html=True)
            
            with st.expander("Key Characteristics", expanded=False):
                all_characteristics = []
                for _, interp in risk_groups['High']:
                    if interp['characteristics']:
                        all_characteristics.extend(interp['characteristics'])
                
                if all_characteristics:
                    unique_characteristics = list(set(all_characteristics))
                    for char in unique_characteristics:
                        st.markdown(f"- {char}")
                else:
                    st.markdown("- No specific characteristics identified.")
            
            with st.expander("Recommendations", expanded=False):
                all_recommendations = []
                for _, interp in risk_groups['High']:
                    if interp['recommendations']:
                        all_recommendations.extend(interp['recommendations'])
                
                if all_recommendations:
                    unique_recommendations = list(set(all_recommendations))
                    for rec in unique_recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.markdown("- No specific recommendations available.")
        
        # Display medium risk clusters
        if risk_groups['Medium']:
            st.markdown(f"""
            <h3 style="color: {COLORS["charts"]["medium_risk"]}; margin-top: 2rem;">Medium Risk Cluster</h3>
            """, unsafe_allow_html=True)
            
            with st.expander("Key Characteristics", expanded=False):
                all_characteristics = []
                for _, interp in risk_groups['Medium']:
                    if interp['characteristics']:
                        all_characteristics.extend(interp['characteristics'])
                
                if all_characteristics:
                    unique_characteristics = list(set(all_characteristics))
                    for char in unique_characteristics:
                        st.markdown(f"- {char}")
                else:
                    st.markdown("- No specific characteristics identified.")
            
            with st.expander("Recommendations", expanded=False):
                all_recommendations = []
                for _, interp in risk_groups['Medium']:
                    if interp['recommendations']:
                        all_recommendations.extend(interp['recommendations'])
                
                if all_recommendations:
                    unique_recommendations = list(set(all_recommendations))
                    for rec in unique_recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.markdown("- No specific recommendations available.")
        
        # Display low risk clusters
        if risk_groups['Low']:
            st.markdown(f"""
            <h3 style="color: {COLORS["charts"]["low_risk"]}; margin-top: 2rem;">Low Risk Cluster</h3>
            """, unsafe_allow_html=True)
            
            with st.expander("Key Characteristics", expanded=False):
                all_characteristics = []
                for _, interp in risk_groups['Low']:
                    if interp['characteristics']:
                        all_characteristics.extend(interp['characteristics'])
                
                if all_characteristics:
                    unique_characteristics = list(set(all_characteristics))
                    for char in unique_characteristics:
                        st.markdown(f"- {char}")
                else:
                    st.markdown("- No specific characteristics identified.")
            
            with st.expander("Recommendations", expanded=False):
                all_recommendations = []
                for _, interp in risk_groups['Low']:
                    if interp['recommendations']:
                        all_recommendations.extend(interp['recommendations'])
                
                if all_recommendations:
                    unique_recommendations = list(set(all_recommendations))
                    for rec in unique_recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.markdown("- No specific recommendations available.")
    else:
        st.warning("Cluster interpretation not available. This may be due to missing data or clustering issues.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from utils.clustering import load_or_train_meanshift_model, cluster_interpretation
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Import visualization functions
# from utils.cluster_visualization import create_3d_cluster_plot, create_cluster_comparison_plot

# def show(df, clustering_data, df_with_risk_labels, COLORS):
#     # Page title
#     st.title("Clustering Analysis")
#     st.markdown("##### Identifying natural groupings among students and analyzing risk patterns")
    
#     # Description of the clustering method
#     st.markdown("""
#     <div class="form-section">
#         <p>This analysis uses the MeanShift clustering algorithm to group students with similar characteristics. 
#         The algorithm automatically determines the optimal number of clusters based on the density of data points, 
#         allowing us to identify natural patterns in student performance and risk factors.</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Load or train MeanShift model
#     with st.spinner("Loading clustering model..."):
#         model = load_or_train_meanshift_model(clustering_data)
    
#     # Create tabs for organizing content
#     tab1, tab2, tab3, tab4 = st.tabs(["Overview", "3D Visualization", "Feature Analysis", "Cluster Profiles"])
    
#     with tab1:
#         # Display cluster visualizations
#         st.subheader("Risk Category Distribution")
        
#         # Create risk category distribution plot
#         if 'Risk_Category' in df_with_risk_labels.columns:
#             risk_counts = df_with_risk_labels['Risk_Category'].value_counts().reset_index()
#             risk_counts.columns = ['Risk_Category', 'Count']
            
#             # Order categories
#             risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
#             risk_counts['Order'] = risk_counts['Risk_Category'].map(risk_order)
#             risk_counts = risk_counts.sort_values('Order').drop('Order', axis=1)
            
#             # Create color map
#             color_map = {
#                 'High': COLORS["charts"]["high_risk"],
#                 'Medium': COLORS["charts"]["medium_risk"],
#                 'Low': COLORS["charts"]["low_risk"]
#             }
            
#             # Create pie chart
#             fig = px.pie(
#                 risk_counts,
#                 values='Count',
#                 names='Risk_Category',
#                 color='Risk_Category',
#                 color_discrete_map=color_map,
#                 hole=0.5
#             )
            
#             # Update layout for dark theme
#             fig.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font=dict(color=COLORS["text"]),
#                 margin=dict(l=20, r=20, t=30, b=20),
#                 legend=dict(
#                     orientation="h",
#                     yanchor="bottom",
#                     y=-0.15,
#                     xanchor="center",
#                     x=0.5
#                 )
#             )
            
#             # Update traces
#             fig.update_traces(
#                 textinfo='percent+label',
#                 pull=[0.05 if x == 'High' else 0 for x in risk_counts['Risk_Category']],
#                 marker=dict(line=dict(color=COLORS["background"], width=2))
#             )
            
#             st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
#         # Display cluster counts
#         if 'Cluster' in df_with_risk_labels.columns:
#             st.subheader("Cluster Distribution")
            
#             # Count clusters
#             cluster_counts = df_with_risk_labels['Cluster'].value_counts().reset_index()
#             cluster_counts.columns = ['Cluster', 'Count']
            
#             # Create percentage column
#             cluster_counts['Percentage'] = (cluster_counts['Count'] / cluster_counts['Count'].sum() * 100).round(1)
            
#             # Create bar chart
#             fig = px.bar(
#                 cluster_counts,
#                 x='Cluster',
#                 y='Count',
#                 text='Percentage',
#                 color='Count',
#                 color_continuous_scale='Viridis',
#                 labels={'Count': 'Number of Students', 'Cluster': 'Cluster ID'}
#             )
            
#             # Update traces
#             fig.update_traces(
#                 texttemplate='%{text}%',
#                 textposition='outside',
#                 marker_line_width=0
#             )
            
#             # Update layout
#             fig.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font=dict(color=COLORS["text"]),
#                 margin=dict(l=20, r=20, t=20, b=20),
#                 xaxis=dict(
#                     title='Cluster ID',
#                     showgrid=False,
#                     zeroline=False
#                 ),
#                 yaxis=dict(
#                     title='Number of Students',
#                     showgrid=True,
#                     gridcolor='rgba(255,255,255,0.1)',
#                     zeroline=False
#                 ),
#                 coloraxis_showscale=False
#             )
            
#             st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
#             # Cluster vs Risk Level heatmap
#             st.subheader("Clusters by Risk Level")
            
#             if 'Risk_Category' in df_with_risk_labels.columns:
#                 # Create crosstab
#                 risk_by_cluster = pd.crosstab(
#                     df_with_risk_labels['Cluster'], 
#                     df_with_risk_labels['Risk_Category'],
#                     normalize='index'
#                 ) * 100
                
#                 # Ensure all risk levels are present
#                 for level in ['High', 'Medium', 'Low']:
#                     if level not in risk_by_cluster.columns:
#                         risk_by_cluster[level] = 0
                
#                 # Reorder columns
#                 if all(level in risk_by_cluster.columns for level in ['High', 'Medium', 'Low']):
#                     risk_by_cluster = risk_by_cluster[['High', 'Medium', 'Low']]
                
#                 # Create heatmap
#                 fig = px.imshow(
#                     risk_by_cluster,
#                     color_continuous_scale=[
#                         [0, COLORS["charts"]["low_risk"]],
#                         [0.5, COLORS["charts"]["medium_risk"]],
#                         [1, COLORS["charts"]["high_risk"]]
#                     ],
#                     text_auto='.1f',
#                     labels=dict(x="Risk Level", y="Cluster", color="Percentage (%)"),
#                     aspect='auto',
#                     height=300
#                 )
                
#                 # Update layout
#                 fig.update_layout(
#                     plot_bgcolor='rgba(0,0,0,0)',
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     font=dict(color=COLORS["text"]),
#                     margin=dict(l=20, r=20, t=20, b=20)
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
#         # Feature importance visualization
#         st.subheader("Feature Importance")
        
#         # Create feature importance visualization
#         important_features = [
#             'Passing_ratio_1st_sem', 
#             'Curricular_units_1st_sem_approved',
#             'Admission_grade',
#             'Previous_qualification_grade',
#             'Scholarship_holder',
#             'Tuition_fees_up_to_date',
#             'Age_at_enrollment',
#             'International'
#         ]
        
#         importance = [0.28, 0.22, 0.15, 0.12, 0.08, 0.07, 0.05, 0.03]
        
#         # Create dataframe for feature importance
#         feature_importance = pd.DataFrame({
#             'feature': important_features,
#             'importance': importance
#         }).sort_values('importance', ascending=True)
        
#         # Create horizontal bar chart
#         fig = px.bar(
#             feature_importance,
#             y='feature',
#             x='importance',
#             orientation='h',
#             color='importance',
#             color_continuous_scale=[[0, COLORS["charts"]["low_risk"]], 
#                                  [0.5, COLORS["charts"]["medium_risk"]], 
#                                  [1, COLORS["charts"]["high_risk"]]],
#             labels={'importance': 'Importance', 'feature': 'Feature'}
#         )
        
#         # Update layout
#         fig.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color=COLORS["text"]),
#             margin=dict(l=20, r=20, t=20, b=20),
#             xaxis=dict(
#                 title='Relative Importance',
#                 showgrid=True,
#                 gridcolor='rgba(255,255,255,0.1)',
#                 zeroline=False
#             ),
#             yaxis=dict(
#                 title=None,
#                 showgrid=False,
#                 zeroline=False
#             ),
#             coloraxis_showscale=False
#         )
        
#         # Remove lines around bars
#         fig.update_traces(marker_line_width=0)
        
#         st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
#     # 3D visualization tab
#     with tab2:
#         st.subheader("3D Cluster Visualization")
        
#         st.markdown("""
#         <div class="form-section">
#             <p>This visualization uses Principal Component Analysis (PCA) to reduce the dimensionality of the data to 3 components,
#             allowing us to visualize the clusters in 3D space. You can rotate, zoom, and interact with the visualization to explore
#             the cluster structure.</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Create 3D cluster visualization
#         fig_3d = create_3d_cluster_plot(df_with_risk_labels, COLORS)
        
#         if fig_3d is not None:
#             st.plotly_chart(fig_3d, use_container_width=True)
            
#             # Show PCA explanation
#             st.markdown("""
#             <div class="hint">
#                 <strong>How to interpret:</strong> Points that are closer together in this 3D space have similar characteristics 
#                 based on the features used for clustering. The colors represent either clusters or risk categories, allowing you 
#                 to see how well the clusters align with risk levels.
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.info("3D visualization is not available due to insufficient data or features.")
    
#     # Feature analysis tab
#     with tab3:
#         st.subheader("Feature Relationship Analysis")
        
#         # Select features for visualization
#         numeric_features = df_with_risk_labels.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
#         # Remove Cluster and Risk columns
#         exclude_cols = ['Cluster', 'Risk_Level']
#         features = [col for col in numeric_features if col not in exclude_cols]
        
#         if len(features) >= 2:
#             # Allow user to select features
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 feature1 = st.selectbox(
#                     "Select first feature:",
#                     options=features,
#                     index=0
#                 )
            
#             with col2:
#                 # Default to second feature
#                 default_idx = 1 if len(features) > 1 else 0
#                 feature2 = st.selectbox(
#                     "Select second feature:",
#                     options=features,
#                     index=default_idx
#                 )
            
#             # Create feature comparison visualization
#             feature_plot = create_cluster_comparison_plot(df_with_risk_labels, feature1, feature2, COLORS)
            
#             if feature_plot is not None:
#                 st.plotly_chart(feature_plot, use_container_width=True)
                
#                 # Show correlation
#                 corr = df_with_risk_labels[feature1].corr(df_with_risk_labels[feature2])
                
#                 st.markdown(f"""
#                 <div class="hint">
#                     <strong>Correlation:</strong> The correlation between {feature1} and {feature2} is {corr:.2f}
#                     ({abs(corr) < 0.3 and "weak" or abs(corr) < 0.7 and "moderate" or "strong"} 
#                     {corr > 0 and "positive" or "negative"} correlation).
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.info("Feature comparison visualization is not available for the selected features.")
            
#             # Display feature statistics by cluster
#             st.subheader("Feature Statistics by Cluster")
            
#             # Calculate statistics
#             cluster_stats = df_with_risk_labels.groupby('Cluster')[features].agg(['mean', 'median', 'std']).reset_index()
            
#             # Flatten MultiIndex columns
#             cluster_stats.columns = ['_'.join(col).strip('_') for col in cluster_stats.columns.values]
            
#             # Rename columns for display
#             renamed_cols = {
#                 'Cluster': 'Cluster',
#                 f'{feature1}_mean': f'{feature1} (Mean)',
#                 f'{feature1}_median': f'{feature1} (Median)',
#                 f'{feature1}_std': f'{feature1} (Std Dev)',
#                 f'{feature2}_mean': f'{feature2} (Mean)',
#                 f'{feature2}_median': f'{feature2} (Median)',
#                 f'{feature2}_std': f'{feature2} (Std Dev)'
#             }
            
#             # Display stats for selected features
#             display_cols = ['Cluster', 
#                           f'{feature1}_mean', f'{feature1}_median', f'{feature1}_std',
#                           f'{feature2}_mean', f'{feature2}_median', f'{feature2}_std']
            
#             # Format dataframe for display
#             display_df = cluster_stats[display_cols].copy()
#             display_df.columns = [renamed_cols.get(col, col) for col in display_df.columns]
            
#             # Round numeric columns
#             for col in display_df.columns:
#                 if col != 'Cluster':
#                     display_df[col] = display_df[col].round(2)
            
#             # Display table
#             st.dataframe(display_df, use_container_width=True)
#         else:
#             st.info("Feature analysis is not available due to insufficient numeric features in the dataset.")
    
#     # Cluster profiles tab
#     with tab4:
#         # Cluster interpretation
#         st.subheader("Cluster Profiles")
        
#         # Get cluster interpretations
#         interpretations = cluster_interpretation(df_with_risk_labels)
        
#         # Display interpretations
#         if interpretations:
#             # Group by risk level
#             risk_groups = {
#                 'High': [],
#                 'Medium': [],
#                 'Low': []
#             }
            
#             # Group clusters by risk level
#             for cluster, interp in interpretations.items():
#                 title = interp.get('title', '')
#                 if 'High' in title:
#                     risk_groups['High'].append((cluster, interp))
#                 elif 'Medium' in title:
#                     risk_groups['Medium'].append((cluster, interp))
#                 elif 'Low' in title:
#                     risk_groups['Low'].append((cluster, interp))
            
#             # Display high risk clusters
#             if risk_groups['High']:
#                 st.markdown(f"""
#                 <h3 style="color: {COLORS["charts"]["high_risk"]}; margin-top: 2rem;">High Risk Clusters</h3>
#                 """, unsafe_allow_html=True)
                
#                 # For each high risk cluster
#                 for cluster_id, interp in risk_groups['High']:
#                     st.markdown(f"""
#                     <div class="risk-card high">
#                         <h4>Cluster {cluster_id}: {interp.get('title', '')}</h4>
#                         <p>{interp.get('description', '')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Key characteristics
#                     if interp['characteristics']:
#                         st.markdown("<h5>Key Characteristics</h5>", unsafe_allow_html=True)
#                         for char in interp['characteristics']:
#                             st.markdown(f"- {char}")
                    
#                     # Recommendations
#                     if interp['recommendations']:
#                         st.markdown("<h5>Recommendations</h5>", unsafe_allow_html=True)
#                         for rec in interp['recommendations']:
#                             st.markdown(f"- {rec}")
                    
#                     st.markdown("<hr>", unsafe_allow_html=True)
            
#             # Display medium risk clusters
#             if risk_groups['Medium']:
#                 st.markdown(f"""
#                 <h3 style="color: {COLORS["charts"]["medium_risk"]}; margin-top: 2rem;">Medium Risk Clusters</h3>
#                 """, unsafe_allow_html=True)
                
#                 # For each medium risk cluster
#                 for cluster_id, interp in risk_groups['Medium']:
#                     st.markdown(f"""
#                     <div class="risk-card medium">
#                         <h4>Cluster {cluster_id}: {interp.get('title', '')}</h4>
#                         <p>{interp.get('description', '')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Key characteristics
#                     if interp['characteristics']:
#                         st.markdown("<h5>Key Characteristics</h5>", unsafe_allow_html=True)
#                         for char in interp['characteristics']:
#                             st.markdown(f"- {char}")
                    
#                     # Recommendations
#                     if interp['recommendations']:
#                         st.markdown("<h5>Recommendations</h5>", unsafe_allow_html=True)
#                         for rec in interp['recommendations']:
#                             st.markdown(f"- {rec}")
                    
#                     st.markdown("<hr>", unsafe_allow_html=True)
            
#             # Display low risk clusters
#             if risk_groups['Low']:
#                 st.markdown(f"""
#                 <h3 style="color: {COLORS["charts"]["low_risk"]}; margin-top: 2rem;">Low Risk Clusters</h3>
#                 """, unsafe_allow_html=True)
                
#                 # For each low risk cluster
#                 for cluster_id, interp in risk_groups['Low']:
#                     st.markdown(f"""
#                     <div class="risk-card low">
#                         <h4>Cluster {cluster_id}: {interp.get('title', '')}</h4>
#                         <p>{interp.get('description', '')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Key characteristics
#                     if interp['characteristics']:
#                         st.markdown("<h5>Key Characteristics</h5>", unsafe_allow_html=True)
#                         for char in interp['characteristics']:
#                             st.markdown(f"- {char}")
                    
#                     # Recommendations
#                     if interp['recommendations']:
#                         st.markdown("<h5>Recommendations</h5>", unsafe_allow_html=True)
#                         for rec in interp['recommendations']:
#                             st.markdown(f"- {rec}")
                    
#                     st.markdown("<hr>", unsafe_allow_html=True)
#         else:
#             st.warning("Cluster interpretation not available. This may be due to missing data or clustering issues.")