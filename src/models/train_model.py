import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.data.preprocessing import load_raw_data, preprocess
from src.data.feature_engineering import (
    split_features_target,
    build_preprocessing_pipeline,
    prepare_datasets
)


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(model, X_test, y_test, title):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def train_baseline_models():

    # Step 1: Load & preprocess raw data
    df = load_raw_data("data/raw/telco_churn.csv")
    df = preprocess(df)

    # Step 2: Split features/target
    X, y = split_features_target(df)

    # Step 3: Build preprocessing transformer
    preprocessor = build_preprocessing_pipeline(X)

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = prepare_datasets(X, y)

    # ----------------------------------------
    # MODEL 1 — Logistic Regression
    # ----------------------------------------
    logreg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    logreg_model.fit(X_train, y_train)

    print("\n=== Logistic Regression Report ===")
    y_pred_logreg = logreg_model.predict(X_test)
    print(classification_report(y_test, y_pred_logreg))

    # ----------------------------------------
    # MODEL 2 — Random Forest
    # ----------------------------------------
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200))
    ])
    rf_model.fit(X_train, y_train)

    print("\n=== Random Forest Report ===")
    y_pred_rf = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

    # ----------------------------------------
    # MODEL 3 — XGBoost
    # ----------------------------------------
    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8
        ))
    ])
    xgb_model.fit(X_train, y_train)

    print("\n=== XGBoost Report ===")
    y_pred_xgb = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred_xgb))

    # ----------------------------------------
    # SAVE MODELS
    # ----------------------------------------
    joblib.dump(logreg_model, "models/logistic_regression.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")

    print("\nModels saved in /models")

    # ----------------------------------------
    # SAVE METRICS
    # ----------------------------------------
    metrics = {
        "logistic_regression": {
            "precision_1": classification_report(y_test, y_pred_logreg, output_dict=True)["1"]["precision"],
            "recall_1": classification_report(y_test, y_pred_logreg, output_dict=True)["1"]["recall"],
            "f1_1": classification_report(y_test, y_pred_logreg, output_dict=True)["1"]["f1-score"]
        },
        "random_forest": {
            "precision_1": classification_report(y_test, y_pred_rf, output_dict=True)["1"]["precision"],
            "recall_1": classification_report(y_test, y_pred_rf, output_dict=True)["1"]["recall"],
            "f1_1": classification_report(y_test, y_pred_rf, output_dict=True)["1"]["f1-score"]
        },
        "xgboost": {
            "precision_1": classification_report(y_test, y_pred_xgb, output_dict=True)["1"]["precision"],
            "recall_1": classification_report(y_test, y_pred_xgb, output_dict=True)["1"]["recall"],
            "f1_1": classification_report(y_test, y_pred_xgb, output_dict=True)["1"]["f1-score"]
        }
    }

    best_model_name = max(metrics, key=lambda m: metrics[m]["f1_1"])
    print(f"\nBest model based on F1: {best_model_name}")

    with open("reports/model_performance.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nReport saved to /reports/model_performance.json")


if __name__ == "__main__":
    train_baseline_models()
