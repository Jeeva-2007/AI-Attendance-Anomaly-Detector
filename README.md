# AI Attendance Anomaly Detector

## 1. Project Title
**AI Attendance Anomaly Detector**

## 2. Problem Statement
In academic institutions, maintaining accurate attendance records is critical for student evaluation and discipline. However, traditional manual attendance systems are prone to errors and manipulation, such as proxy attendance (students marking attendance for absent peers) or inconsistent record-keeping. Detecting these irregularities manually across large batches of students is time-consuming, inefficient, and often inaccurate. There is a need for an automated system that can analyze attendance patterns to identify potential anomalies without human bias.

## 3. Project Objective
The primary objective of this project is to develop a machine learning-based system that automatically detects irregular attendance patterns and categorizes students based on their risk levels. This system aims to provide faculty and administrators with actionable insights, enabling early intervention for students with abnormal attendance behaviors.

## 4. Approach
This project utilizes **Unsupervised Machine Learning** techniques, as labeled data for "anomalous" attendance is rarely available in real-world scenarios. The system employs a dual-model approach:

1.  **Isolation Forest**: A robust anomaly detection algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value. It effectively identifies outliers (anomalies) that differ significantly from the norm.
2.  **K-Means Clustering**: A clustering algorithm that groups students into distinct behavioral categories (e.g., Regular, Irregular, At-Risk) based on feature similarity.
3.  **Risk-Level Fusion**: A custom logic layer that combines the outputs of both models to assign a final **Risk Level** to each student:
    -   **High Risk**: Flagged as an anomaly by Isolation Forest AND identified as an outlier by K-Means distance.
    -   **Medium Risk**: Flagged by only one of the models.
    -   **Low Risk**: Considered normal by both models.

## 5. Dataset Description
The system operates on a realistic synthetic dataset designed to mimic actual student attendance records. The dataset includes numeric features that capture various aspects of student presence, such as:
-   **Attendance Percentage**: The overall percentage of classes attended.
-   **Absent Days**: Total count of days the student was absent.
-   **Late Arrivals**: Frequency of late entries (if applicable).
-   **Period-wise Presence**: specific metrics tracking attendance in different sessions (e.g., period_1, period_3, period_5, period_7) to detect patterns like skipping specific classes.
-   **Variance**: Statistical measures to detect inconsistent attendance behavior.

## 6. Tech Stack
The project is built using a modern Python-based technology stack:
-   **Python**: Primary programming language.
-   **Pandas & NumPy**: For efficient data manipulation and numerical analysis.
-   **Scikit-learn**: For implementing the Isolation Forest and K-Means algorithms.
-   **Streamlit**: For creating the interactive web-based user interface.
-   **Matplotlib & Seaborn**: For generating data visualizations and statistical plots.

## 7. System Workflow
1.  **Data Upload**: The user uploads a CSV file containing attendance records via the web interface.
2.  **Preprocessing**: The system cleans the data, handles missing values, removes non-numeric identifiers (like Student IDs), and scales the features using Standard Scaler.
3.  **Model Training**: The preprocessed data is fed into the Isolation Forest and K-Means models, which are trained on the fly.
4.  **Anomaly Detection**: The models predict anomalies and calculate clustering distances.
5.  **Risk Analysis**: The fusion logic determines the final risk level for each student.
6.  **Visualization**: The results are presented through interactive charts and tables, allowing for detailed exploration.

## 8. How to Run the Project
To run this project locally, follow these steps:

1.  **Clone the Repository**:
    Download the project files to your local machine.

2.  **Install Dependencies**:
    Navigate to the project directory and install the required Python packages using:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    Launch the Streamlit web interface:
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser (typically at http://localhost:8501).

## 9. Results
The system outputs a comprehensive analysis report including:
-   **Summary Metrics**: Total number of students, count of high-risk individuals, and detected anomalies.
-   **Visualizations**: Distribution of risk levels, histograms of anomaly scores, and cluster scatter plots.
-   **Detailed Tables**: A searchable list of students with their computed Risk Level (High/Medium/Low).
-   **Downloadable Report**: A CSV file containing the final analysis, which faculty can use to target specific students for counseling or disciplinary action.

## 10. Ethical Considerations
It is important to note that this system is designed as a **decision-support tool** for faculty and administrators. It does not replace human judgment. An "anomaly" or "high risk" flag indicates a statistical deviation that warrants further investigation, not immediate proof of misconduct. Administrators should verify the data and context before taking action.

## 11. Future Enhancements
-   **Biometric Integration**: Incorporating fingerprint or facial recognition data for more accurate inputs.
-   **Real-time Alerts**: Sending automated emails or SMS to faculty when a high-risk pattern is detected.
-   **Dashboard Improvements**: Adding historical trend analysis to track student improvement over time.
