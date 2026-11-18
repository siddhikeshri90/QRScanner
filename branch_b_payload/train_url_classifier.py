"""
Train and evaluate URL phishing classifier.

Input:
data/urls.csv
Columns: url, label
0 = legitimate
1 = malicious

Model selection uses training cross-validation.
Test set is kept separate for final evaluation.
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from url_features import extract_url_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_CSV = os.path.join(PROJECT_DIR, "data", "urls.csv")
MODEL_OUT = os.path.join(BASE_DIR, "url_model.joblib")
FEATURE_ORDER_OUT = os.path.join(BASE_DIR, "url_feature_order.joblib")


def build_feature_table(df):
    rows = []

    total = len(df)

    for index, url in enumerate(df["url"].astype(str), start=1):
        rows.append(extract_url_features(url))

        if index % 100 == 0 or index == total:
            print(f"Extracted features: {index}/{total}")

    return pd.DataFrame(rows)


def evaluate(model, X_test, y_test, name):
    predictions = model.predict(X_test)

    print(f"\n========== {name} TEST RESULTS ==========")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))


def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    required_columns = {"url", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain 'url' and 'label' columns.")

    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both labels 0 and 1.")

    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    print("Dataset rows:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts())

    print("\nExtracting URL features...")
    X = build_feature_table(df)
    y = df["label"].astype(int)

    feature_order = list(X.columns)

    print("\nFeatures used:")
    print(feature_order)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_name = None
    best_cv_f1 = -1

    print("\nComparing models using training 5-fold cross-validation...")

    for name, model in models.items():
        print(f"\nTraining: {name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )

        mean_f1 = scores.mean()
        print(f"CV F1: {mean_f1:.4f} (+/- {scores.std():.4f})")

        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_model = model
            best_name = name

    print(f"\nBest model by CV: {best_name}")
    print(f"Mean CV F1: {best_cv_f1:.4f}")

    print("\nFitting selected model on training data...")
    best_model.fit(X_train, y_train)

    # Evaluate once on the untouched test set.
    evaluate(best_model, X_test, y_test, best_name)

    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(feature_order, FEATURE_ORDER_OUT)

    print("\nModel saved:", MODEL_OUT)
    print("Feature order saved:", FEATURE_ORDER_OUT)
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()