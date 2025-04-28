"""
Module for preprocessing transaction data: imputing, encoding, and scaling.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Any

def preprocess_transactions(df: pd.DataFrame) -> Tuple[Any, ColumnTransformer]:
    """
    Preprocess raw transaction DataFrame.

    Args:
        df: Raw transaction DataFrame.
    Returns:
        Tuple of transformed array (features) and fitted ColumnTransformer.
    """
    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    X = preprocessor.fit_transform(df)
    return X, preprocessor
