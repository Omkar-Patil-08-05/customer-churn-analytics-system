import pandas as pd
import joblib

from src.data.preprocessing import load_raw_data, preprocess
from src.data.feature_engineering import (
    split_features_target,
    build_preprocessing_pipeline,
    prepare_datasets
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def train_baseline_models():

    # Step 1: Load and preprocess
    df = load_raw_data("data/raw/telco_churn.csv")
    df = preprocess(df)

    # Step 2: Feature/target split
    X, y = split_features_target(df)

    # Step 3: Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(X)

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = prepare_datasets(X, y)

    # MODEL 1: Logistic Regression
    logreg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    logreg_model.fit(X_train, y_train)

    print("\n=== Logistic Regression Report ===")
    y_pred_logreg = logreg_model.predict(X_test)
    print(classification_report(y_test, y_pred_logreg))

    # MODEL 2: Random Forest
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200))
    ])

    rf_model.fit(X_train, y_train)

    print("\n=== Random Forest Report ===")
    y_pred_rf = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

    # Save models
    joblib.dump(logreg_model, "models/logistic_regression.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")

    print("\nModels saved in: models/")
