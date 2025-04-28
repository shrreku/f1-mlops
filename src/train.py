"""
Script to orchestrate data ingestion, preprocessing, model training, and artifact saving.
"""
import argparse
import pandas as pd
from src.data_ingestion import list_csv_files, load_transaction_data
from src.preprocessing import preprocess_transactions
from src.model_training import train_isolation_forest, save_artifacts

def main(args):
    # Load and concatenate CSV files
    csv_files = list_csv_files(args.data_dir)
    if not csv_files:
        raise ValueError(f"No CSV files found in {args.data_dir}")
    df_list = [load_transaction_data(fp) for fp in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # Separate features and label
    if 'Class' not in df.columns:
        raise KeyError("Expected 'Class' column for labels.")
    y = df['Class']
    X = df.drop(columns=['Class'])

    # Preprocess
    X_proc, preprocessor = preprocess_transactions(X)

    # Train model
    model = train_isolation_forest(X_proc, y, test_size=args.test_size)

    # Save artifacts
    save_artifacts(model, preprocessor, args.artifact_dir)
    print(f"Artifacts saved to {args.artifact_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory with CSV files')
    parser.add_argument('--artifact-dir', type=str, default='artifacts', help='Location to save model artifacts')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    args = parser.parse_args()
    main(args)
