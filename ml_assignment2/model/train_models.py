"""
train_models.py
---------------------------------------------------------------------------
Machine Learning Assignment 2 — model training pipeline.

Dataset : UCI Breast Cancer Wisconsin (Diagnostic)  [569 rows x 30 features]
Task    : Binary classification  (0 = malignant, 1 = benign)

For each of the 5 required classifiers this script:
  1. trains the model (inside a Pipeline so preprocessing travels with it),
  2. evaluates Accuracy, AUC, Precision, Recall, F1 and MCC on a held-out test set,
  3. serialises the fitted Pipeline to model/<name>.pkl (via joblib).

It also writes:
  - test_data.csv   (the held-out test split used by the Streamlit app)
  - model/metadata.json (feature order, class labels, and the metrics table)

Run:  python model/train_models.py     (from the project root)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

HERE      = os.path.dirname(os.path.abspath(__file__))   # .../model
ROOT      = os.path.dirname(HERE)                         # project root
RANDOM    = 42
TARGET    = "target"

# --------------------------------------------------------------------------- #
# 1. Load data as a DataFrame (keeps human-readable feature names)
# --------------------------------------------------------------------------- #
data = load_breast_cancer(as_frame=True)
X = data.data.copy()                        # 30 numeric features
y = data.target.copy()                      # 0 = malignant, 1 = benign
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM)

# Persist the test split so the deployed app evaluates on the SAME data
test_df = X_test.copy()
test_df[TARGET] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

# --------------------------------------------------------------------------- #
# 2. Define the 5 models. Scale-sensitive models (LogReg, kNN) get a
#    StandardScaler inside their Pipeline; tree / NB models do not need one.
# --------------------------------------------------------------------------- #
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM)),
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM)),
    ]),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ]),
    "Naive Bayes": Pipeline([
        ("clf", GaussianNB()),
    ]),
    "Random Forest (Ensemble)": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM)),
    ]),
}

# file-name slug for each model
slug = {
    "Logistic Regression":       "logistic_regression",
    "Decision Tree":             "decision_tree",
    "kNN":                       "knn",
    "Naive Bayes":               "naive_bayes",
    "Random Forest (Ensemble)":  "random_forest",
}

# --------------------------------------------------------------------------- #
# 3. Train, evaluate, and save
# --------------------------------------------------------------------------- #
def evaluate(model, X_te, y_te):
    """Return the 6 required metrics for a fitted model on the test set."""
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]        # P(class = 1) for AUC
    return {
        "Accuracy":  accuracy_score(y_te, y_pred),
        "AUC":       roc_auc_score(y_te, y_prob),
        "Precision": precision_score(y_te, y_pred),
        "Recall":    recall_score(y_te, y_pred),
        "F1":        f1_score(y_te, y_pred),
        "MCC":       matthews_corrcoef(y_te, y_pred),
    }

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    results[name] = evaluate(pipe, X_test, y_test)
    joblib.dump(pipe, os.path.join(HERE, f"{slug[name]}.pkl"))
    print(f"saved {slug[name]}.pkl")

# comparison table (rounded for display)
table = pd.DataFrame(results).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
print("\n=== Comparison table (test set) ===")
print(table.round(4).to_string())

# --------------------------------------------------------------------------- #
# 4. Metadata for the app (feature order, class labels, cached metrics)
# --------------------------------------------------------------------------- #
metadata = {
    "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
    "target": TARGET,
    "class_labels": {"0": "malignant", "1": "benign"},
    "feature_names": feature_names,
    "slug": slug,
    "metrics": {k: {m: float(v) for m, v in d.items()} for k, d in results.items()},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(os.path.join(HERE, "metadata.json"), "w") as fh:
    json.dump(metadata, fh, indent=2)
print("\nsaved metadata.json and test_data.csv")
