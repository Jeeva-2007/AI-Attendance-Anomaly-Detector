from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def load_model(path=None):
    """
    Load a trained model or return a new instance.
    """
    # For now, we are training on the fly, so this might just return None or load a saved pickle.
    # Leaving as placeholder for future model persistence.
    pass

def train_model(X, contamination=0.25):
    """
    Train the Isolation Forest model.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    return model

def predict_anomalies(model, X):
    """
    Predict anomalies in the data.
    Returns a dictionary with 'predictions', 'labels', and 'scores'.
    """
    predictions = model.predict(X)
    # -1 is anomaly, 1 is normal
    labels = ["Anomaly" if x == -1 else "Normal" for x in predictions]
    scores = model.decision_function(X)
    
    return {
        "predictions": predictions,
        "labels": labels,
        "scores": scores
    }

def train_kmeans(X, k=3):
    """
    Train K-Means Clustering model.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

def predict_kmeans(model, X):
    """
    Predict clusters and calculate distance to centroids.
    """
    clusters = model.predict(X)
    
    # Calculate distance to assigned centroid
    # transform(X) returns distances to all centroids
    all_distances = model.transform(X)
    # Select the distance to the assigned cluster centroid for each point
    distances = all_distances[np.arange(len(all_distances)), clusters]
    
    return {
        "clusters": clusters,
        "distances": distances
    }

def calculate_risk_level(anomaly_labels, centroid_distances, distance_threshold_percentile=75):
    """
    Combine Isolation Forest and K-Means outputs to determine Risk Level.
    
    Rules:
    - Both models flag -> High Risk
    - One model flags -> Medium Risk
    - No model flags -> Low Risk
    
    Args:
        anomaly_labels (list): 'Anomaly' or 'Normal' from Isolation Forest.
        centroid_distances (list): Distances to K-Means centroids.
        distance_threshold_percentile (int): Percentile to flag K-Means outlier.
        
    Returns:
        list: Risk levels (High, Medium, Low)
    """
    # Determine K-Means threshold
    threshold = np.percentile(centroid_distances, distance_threshold_percentile)
    
    risk_levels = []
    for label, dist in zip(anomaly_labels, centroid_distances):
        # Isolation Forest Flag
        if_flag = (label == "Anomaly")
        
        # K-Means Flag
        km_flag = (dist > threshold)
        
        if if_flag and km_flag:
            risk_levels.append("High")
        elif if_flag or km_flag:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")
            
    return risk_levels
