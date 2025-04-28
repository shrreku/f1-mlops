"""
Module for training and evaluating fraud detection models.
"""
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
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
        X, y, test_size=test_size, random_state=random_state, stratify=y # Stratify for imbalanced data
    )

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'contamination': [0.001, 0.01, 0.05, 'auto'], # Include 'auto' and smaller values
        'max_features': [0.5, 0.75, 1.0]
    }

    # Use recall as the primary scoring metric due to imbalance
    # Note: IsolationForest predicts -1 for anomalies (fraud), 1 for normal.
    # We need a custom scorer because default recall assumes 1 is the positive class.
    # Our conversion maps -1 -> 1 (fraud), 1 -> 0 (normal), so standard recall works.
    scoring = {'recall': make_scorer(recall_score), 'precision': make_scorer(precision_score)}

    # Initialize Isolation Forest
    iso_forest = IsolationForest(random_state=random_state)

    # Setup GridSearchCV
    # Refit using recall to maximize finding fraud cases
    grid_search = GridSearchCV(iso_forest, param_grid, scoring=scoring, refit='recall', cv=3, n_jobs=-1)

    mlflow.sklearn.autolog(log_models=False) # Autolog parameters and metrics, but handle model logging manually
    with mlflow.start_run() as run:
        print("Starting GridSearchCV for Isolation Forest...")
        grid_search.fit(X_train, y_train) # Fit on training data
        print("GridSearchCV finished.")

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best Parameters found: {best_params}")
        mlflow.log_params(best_params)

        # Evaluate the best model found by GridSearchCV on the test set
        y_pred_test = best_model.predict(X_test)
        # Convert predictions: -1 (anomaly/fraud) -> 1, 1 (normal) -> 0
        y_pred_test_binary = np.array([1 if pred == -1 else 0 for pred in y_pred_test])

        test_precision = precision_score(y_test, y_pred_test_binary)
        test_recall = recall_score(y_test, y_pred_test_binary)

        mlflow.log_metric("best_test_precision", test_precision)
        mlflow.log_metric("best_test_recall", test_recall)

        # Log the best score achieved during CV (based on recall)
        mlflow.log_metric("best_cv_recall", grid_search.best_score_)

        # Print final metrics
        print(f"Best Model Test Precision: {test_precision:.4f}")
        print(f"Best Model Test Recall: {test_recall:.4f}")

        # Log the best model manually
        mlflow.sklearn.log_model(best_model, "isolation-forest-best-model")

        # Visualize anomaly scores for the best model
        anomaly_scores = best_model.decision_function(X_test)
        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_scores, bins=50, kde=True)
        plt.title('Distribution of Anomaly Scores (Best Model) on Test Set')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        # Ensure artifacts directory exists before saving plot
        os.makedirs('artifacts', exist_ok=True)
        plot_path = os.path.join('artifacts', 'best_anomaly_scores_distribution.png')
        plt.savefig(plot_path)
        print(f"Anomaly score distribution plot saved to {plot_path}")
        mlflow.log_artifact(plot_path)

    return best_model

def save_artifacts(model, preprocessor, path: str):
    """
    Save trained model and preprocessor to disk.
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, 'model.joblib'))
    joblib.dump(preprocessor, os.path.join(path, 'preprocessor.joblib'))
