"""
Module for training and evaluating fraud detection models.
"""
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import joblib, os

def train_isolation_forest(X, y, test_size=0.2, random_state=42):
    """
    Train an IsolationForest model and log metrics to MLflow.

    Args:
        X: Feature matrix array.
        y: True labels array (0 for legit, 1 for fraud).
    Returns:
        Trained IsolationForest model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=random_state)

    mlflow.sklearn.autolog()
    with mlflow.start_run():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        # convert to binary fraud labels: -1 -> 1 (fraud), 1 -> 0 (legit)
        y_pred = (y_pred == -1).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Print metrics
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")

        # Visualize anomaly scores
        anomaly_scores = model.decision_function(X_test)
        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_scores, bins=50, kde=True)
        plt.title('Distribution of Anomaly Scores on Test Set')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        # Ensure artifacts directory exists before saving plot
        os.makedirs('artifacts', exist_ok=True)
        plot_path = os.path.join('artifacts', 'anomaly_scores_distribution.png')
        plt.savefig(plot_path)
        print(f"Anomaly score distribution plot saved to {plot_path}")
        mlflow.log_artifact(plot_path)

    return model

def save_artifacts(model, preprocessor, path: str):
    """
    Save trained model and preprocessor to disk.
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, 'model.joblib'))
    joblib.dump(preprocessor, os.path.join(path, 'preprocessor.joblib'))
