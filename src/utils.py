import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_clusters(df, x_col, y_col):
    """
    Scatter plot of students colored by Cluster.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue='Cluster', 
        palette='viridis', 
        style='Risk_Level',
        s=100,
        ax=ax
    )
    plt.title(f"Cluster Visualization: {x_col} vs {y_col}")
    return fig

def plot_anomaly_scores(df):
    """
    Histogram and KDE of Anomaly Scores.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Anomaly_Score'], kde=True, color='purple', bins=30, ax=ax)
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score (Lower score = More Anomalous)")
    return fig

def plot_risk_distribution(df):
    """
    Bar chart of Risk Level counts.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Order: Low, Medium, High
    risk_order = ['Low', 'Medium', 'High']
    # Check present keys to avoid error if some category is missing
    present_risks = [r for r in risk_order if r in df['Risk_Level'].unique()]
    
    sns.countplot(x='Risk_Level', data=df, order=present_risks, palette='coolwarm', ax=ax)
    plt.title("Student Count by Risk Level")
    return fig

def plot_anomalies(df, x_col, y_col):
    """
    Scatter plot highlighting anomalies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot normal points
    normal = df[df['Anomaly_Label'] == 'Normal']
    sns.scatterplot(data=normal, x=x_col, y=y_col, color='lightgray', label='Normal', alpha=0.6, ax=ax)
    
    # Plot anomalies
    anomalies = df[df['Anomaly_Label'] == 'Anomaly']
    sns.scatterplot(data=anomalies, x=x_col, y=y_col, color='red', label='Anomaly', s=50, ax=ax)
    
    plt.title(f"Anomaly Detection: {x_col} vs {y_col}")
    return fig
