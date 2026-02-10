import streamlit as st
import pandas as pd
import os
from src.preprocess import load_data, validate_data, preprocess_data
from src.model import train_model, predict_anomalies, train_kmeans, predict_kmeans, calculate_risk_level
from src.utils import plot_anomalies, plot_clusters, plot_anomaly_scores, plot_risk_distribution

# Set page config
st.set_page_config(page_title="AI Attendance Anomaly Detector", layout="wide", page_icon="🕵️‍♂️")

def run_pipeline(df):
    """
    Runs the full anomaly detection and risk analysis pipeline.
    """
    # 1. Preprocessing
    X_scaled, df_processed = preprocess_data(df)
    
    # 2. Isolation Forest (Anomaly Detection)
    model_if = train_model(X_scaled, contamination=0.25)
    results_if = predict_anomalies(model_if, X_scaled)
    df['Anomaly_Label'] = results_if['labels']
    df['Anomaly_Score'] = results_if['scores']
    
    # 3. K-Means Clustering
    k = 3
    model_km = train_kmeans(X_scaled, k=k)
    results_km = predict_kmeans(model_km, X_scaled)
    df['Cluster'] = results_km['clusters']
    df['Centroid_Distance'] = results_km['distances']
    
    # 4. Risk Level Calculation
    risk_levels = calculate_risk_level(df['Anomaly_Label'], df['Centroid_Distance'])
    df['Risk_Level'] = risk_levels
    
    return df, df_processed

def main():
    st.title("🕵️‍♂️ AI Attendance Anomaly Detector")
    st.markdown("""
    Upload your attendance data to automatically detect irregular patterns, cluster behavior, and assess risk levels.
    """)

    # Sidebar
    st.sidebar.header("📂 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    # Load Data logic
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("Awaiting upload... using demo data? 👇")
        if st.sidebar.button("Load Demo Data"):
            try:
                df = load_data("data/attendance.csv")
                st.sidebar.success("Demo data loaded!")
            except FileNotFoundError:
                st.sidebar.error("Demo data not found.")

    if df is not None:
        # Run Analysis Automatically
        with st.spinner("Running Analysis Pipeline..."):
            try:
                # Validation
                validation = validate_data(df)
                if not validation['numeric_check']:
                    st.warning("⚠️ Some columns are non-numeric. They will be ignored during analysis.")
                
                # Execute Pipeline
                analyzed_df, df_processed = run_pipeline(df)
                
                # --- DASHBOARD METRICS ---
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                total_students = len(analyzed_df)
                high_risk = len(analyzed_df[analyzed_df['Risk_Level'] == 'High'])
                anomalies = len(analyzed_df[analyzed_df['Anomaly_Label'] == 'Anomaly'])
                avg_score = analyzed_df['Anomaly_Score'].mean()
                
                col1.metric("Total Students", total_students)
                col2.metric("🚨 High Risk", high_risk, delta_color="inverse")
                col3.metric("⚠️ Anomalies Detected", anomalies)
                col4.metric("Avg Anomaly Score", f"{avg_score:.2f}")
                
                # --- VISUALIZATIONS ---
                st.subheader("📊 Analysis Visualizations")
                tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Anomaly Scores", "Clusters & Patterns"])
                
                with tab1:
                    st.pyplot(plot_risk_distribution(analyzed_df))
                    
                with tab2:
                    st.pyplot(plot_anomaly_scores(analyzed_df))
                    
                with tab3:
                    col_x, col_y = st.columns(2)
                    feature_cols = df_processed.columns.tolist()
                    if len(feature_cols) >= 2:
                        x_axis = col_x.selectbox("X-axis", feature_cols, index=0)
                        y_axis = col_y.selectbox("Y-axis", feature_cols, index=1)
                        
                        st.write("#### Cluster View")
                        st.pyplot(plot_clusters(analyzed_df, x_axis, y_axis))
                        
                        st.write("#### Anomaly View")
                        st.pyplot(plot_anomalies(analyzed_df, x_axis, y_axis))
                    else:
                        st.warning("Not enough numeric features for scatter plots.")

                # --- DETAILED DATA ---
                st.markdown("---")
                st.subheader("📋 Detailed Results")
                
                # Filter options
                filter_risk = st.multiselect("Filter by Risk Level", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
                filtered_df = analyzed_df[analyzed_df['Risk_Level'].isin(filter_risk)]
                
                st.dataframe(filtered_df.style.map(lambda x: 'color: red' if x == 'High' else ('color: orange' if x == 'Medium' else 'color: green'), subset=['Risk_Level']))
                
                # --- DOWNLOAD ---
                st.markdown("---")
                st.subheader("💾 Export Report")
                
                # Prepare results.csv format
                output_columns = ['student_id', 'anomaly_score', 'cluster_label', 'distance_from_centroid', 'risk_level']
                final_export = analyzed_df.copy()
                
                # Rename for export standard
                final_export = final_export.rename(columns={
                    'Anomaly_Score': 'anomaly_score',
                    'Cluster': 'cluster_label',
                    'Centroid_Distance': 'distance_from_centroid',
                    'Risk_Level': 'risk_level'
                })
                
                # Ensure student_id exists (if it was in original df)
                if 'student_id' not in final_export.columns and 'student_id' in df.columns:
                     final_export['student_id'] = df['student_id']
                
                # Select only relevant columns if they exist
                export_cols = [c for c in output_columns if c in final_export.columns]
                final_csv = final_export[export_cols].to_csv(index=False).encode('utf-8')
                
                # Save locally logic from Phase 7 (optional but good for persistence)
                if not os.path.exists('output'):
                    os.makedirs('output')
                final_export[export_cols].to_csv('output/results.csv', index=False)

                st.download_button(
                    label="Download results.csv",
                    data=final_csv,
                    file_name='results.csv',
                    mime='text/csv',
                    type='primary'
                )
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                raise e

if __name__ == "__main__":
    main()
