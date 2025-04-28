"""
Module for loading transaction data for fraud detection.
"""
import pandas as pd
from typing import List

def load_transaction_data(file_path: str) -> pd.DataFrame:
    """
    Load transaction CSV data into a DataFrame.

    Args:
        file_path: Path to the CSV file.
    Returns:
        DataFrame with raw transactions.
    """
    df = pd.read_csv(file_path)
    return df


def list_csv_files(directory: str) -> List[str]:
    """
    List CSV files in a directory.

    Args:
        directory: Directory path to search.
    Returns:
        List of CSV file paths.
    """
    import os
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
