"""
app.py — Streamlit front-end for ML Assignment 2
============================================================================
Interactive demo of 5 classification models trained on the
UCI Breast Cancer Wisconsin (Diagnostic) dataset.

Features (assignment Step 6):
  a. CSV upload for the test data
  b. Model-selection dropdown
  c. Display of the 6 evaluation metrics
  d. Confusion matrix + classification report

Run locally :  streamlit run app.py
Deployed on :  Streamlit Community Cloud
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# --------------------------------------------------------------------------- #
# Config + cached loaders
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="ML Assignment 2 — Classifier Demo",
                   page_icon="🧪", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


@st.cache_resource
def load_metadata():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as fh:
        return json.load(fh)


@st.cache_resource
def load_models(slug):
    return {name: joblib.load(os.path.join(MODEL_DIR, f"{s}.pkl"))
            for name, s in slug.items()}


meta = load_metadata()
models = load_models(meta["slug"])
FEATURES = meta["feature_names"]
TARGET = meta["target"]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Breast Cancer Classification — Model Explorer")
st.caption(f"Dataset: **{meta['dataset']}**  ·  "
           f"{len(FEATURES)} features  ·  "
           f"train/test = {meta['n_train']}/{meta['n_test']}  ·  "
           f"target: 0 = malignant, 1 = benign")


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Sidebar — (a) CSV upload  +  (b) model dropdown
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Use the test_data.csv from the repo. It must include the "
         "'target' column plus the 30 feature columns.")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("Tip: the repo's `test_data.csv` is the held-out split "
                   "the models were evaluated on.")

# --------------------------------------------------------------------------- #
# Comparison table across all models (from training)
# --------------------------------------------------------------------------- #
with st.expander("📊 Model comparison table (computed at training time)", expanded=True):
    comp = (pd.DataFrame(meta["metrics"]).T
            [["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]])
    st.dataframe(comp.style.format("{:.4f}")
                 .highlight_max(axis=0, color="#c9f7c9"),
                 use_container_width=True)

# --------------------------------------------------------------------------- #
# Main panel — needs an uploaded file
# --------------------------------------------------------------------------- #
if uploaded is None:
    st.info("⬅️ Upload `test_data.csv` from the sidebar to evaluate a model on it.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview of uploaded data")
st.dataframe(df.head(), use_container_width=True)

if TARGET not in df.columns:
    st.error(f"The uploaded CSV must contain a '{TARGET}' column to compute metrics.")
    st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing {len(missing)} expected feature column(s): "
             f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    st.stop()

# Align feature order exactly with what the pipelines were trained on
X = df[FEATURES]
y = df[TARGET].astype(int)

model = models[model_name]
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]
metrics = compute_metrics(y, y_pred, y_prob)

# --- (c) Display of evaluation metrics --------------------------------------
st.subheader(f"Evaluation metrics — {model_name}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, key in zip([c1, c2, c3, c4, c5, c6],
                    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    col.metric(key, f"{metrics[key]:.4f}")

# --- (d) Confusion matrix + classification report ---------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["malignant (0)", "benign (1)"],
                yticklabels=["malignant (0)", "benign (1)"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, y_pred, target_names=["malignant (0)", "benign (1)"],
        output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"),
                 use_container_width=True)

st.success(f"Evaluated **{model_name}** on {len(df)} uploaded rows.")
