import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split_features_target(df: pd.DataFrame):
    """
    Split target (Churn) from features.
    Convert target to binary: Yes=1, No=0.
    """
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame):
    """
    Build a preprocessing pipeline for:
    - One-hot encoding categorical columns
    - Scaling numeric columns
    """

    # Identify columns by type
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor


def prepare_datasets(X, y, test_size=0.2, random_state=42):
    """
    Train-test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
