import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise e

def validate_data(df):
    """
    Validate the dataset: check shape, columns, missing values, and data types.
    """
    validation_results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum(),
        "numeric_check": df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
    }
    return validation_results

from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the attendance data:
    1. Remove non-ML columns (student_id).
    2. Select numeric behavioral features.
    3. Apply StandardScaler.
    
    Returns:
        X_scaled (numpy array): Scaled feature matrix.
        df_processed (DataFrame): Processed dataframe with selected features.
    """
    # 1. Remove non-ML columns
    if 'student_id' in df.columns:
        df_processed = df.drop(columns=['student_id'])
    else:
        df_processed = df.copy()

    # 2. Select only numeric features
    df_processed = df_processed.select_dtypes(include=[np.number])

    # 3. Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed)

    return X_scaled, df_processed
