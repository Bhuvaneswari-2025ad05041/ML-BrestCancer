# Machine Learning Assignment 2 — Multi-Model Classification + Streamlit App

End-to-end classification workflow: train five models on one dataset, compare them
across six metrics, and serve the results through an interactive Streamlit app
deployed on Streamlit Community Cloud.

---

## a. Problem Statement

Given diagnostic measurements taken from digitised images of breast-mass cell
nuclei, predict whether a tumour is **malignant (0)** or **benign (1)**. This is a
**binary classification** problem. The goal is to train several classifiers on the
same dataset, evaluate them on a common held-out test set using six metrics, and
identify the best model for this data.

## b. Dataset Description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository — *Breast Cancer Wisconsin (Diagnostic)* |
| Instances | 569  (≥ 500 required) |
| Features | 30 numeric  (≥ 12 required) |
| Target | `target` — 0 = malignant, 1 = benign (binary) |
| Class balance | 212 malignant / 357 benign |
| Train / Test split | 455 / 114  (80 / 20, stratified, `random_state=42`) |

The 30 features are ten cell-nucleus measurements — *radius, texture, perimeter,
area, smoothness, compactness, concavity, concave points, symmetry, fractal
dimension* — each reported as **mean**, **standard error (se)**, and **worst**
value. `test_data.csv` in this repo is the exact held-out test split; it is what the
Streamlit app evaluates.

## c. GitHub Repository Link

https://github.com/Bhuvaneswari-2025ad05041/ML-BrestCancer/tree/main/ml_assignment2

Contents:

```
project-folder/
├── app.py                 # Streamlit front-end
├── requirements.txt       # dependencies for deployment
├── README.md              # this file
├── test_data.csv          # held-out test split (uploaded in the app)
└── model/
    ├── train_models.py    # training + evaluation pipeline
    ├── metadata.json       # feature order, class labels, cached metrics
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## d. Models Used — Comparison Table

All five models were trained on the **same** train split and evaluated on the
**same** 114-row test split. Metrics are for the positive class (benign = 1);
best value in each column is **bold**.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | 0.9861 | **0.9861** | **0.9623** |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN (k = 7) | 0.9737 | 0.9884 | 0.9600 | **1.0000** | 0.9796 | 0.9442 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |


*Metrics: Accuracy, AUC (ROC), Precision, Recall, F1, and Matthews Correlation
Coefficient (MCC). Reproduce with `python model/train_models.py`.*

### Observations on Model Performance

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | **Best overall** — tops Accuracy, AUC, F1 and MCC. After standardisation the two classes are almost linearly separable, which perfectly suits a linear decision boundary; probabilities are well-calibrated, giving the highest AUC (0.995). |
| Decision Tree | **Weakest here** — lowest Accuracy (0.921) and, notably, lowest AUC (0.916). A single depth-limited tree produces coarse, step-like probability estimates and has high variance on 30 continuous features, which hurts ranking quality (AUC) the most. |
| kNN (k = 7) | Very strong (Accuracy 0.974) with **perfect recall (1.000)** — it caught every benign case. Relies heavily on the `StandardScaler` in its pipeline; slightly lower precision (0.960) means a few malignant cases leaked into the benign predictions. |
| Naive Bayes (Gaussian) | High AUC (0.988) but lower Accuracy (0.939). Its core assumption of feature independence is violated — radius, perimeter and area are strongly correlated — which caps accuracy even though class ranking stays good. |
| Random Forest (Ensemble) | Strong and stable (Accuracy 0.947, AUC 0.994). The ensemble slashes the single tree's variance (MCC 0.887 vs 0.834), making it the most robust *non-linear* model, though the clean, near-linear data lets Logistic Regression edge it out. |
| **Overall Winner** | **Logistic Regression** — highest Accuracy, AUC, F1 and MCC. A useful lesson: on a clean, well-behaved, near-linearly-separable dataset a simple linear model can beat heavier ensembles. |

---

## Streamlit App Features (Step 6)

The deployed app (`app.py`) provides:

- **a. CSV upload** — upload `test_data.csv` (only test data, per free-tier limits).
- **b. Model-selection dropdown** — choose any of the 5 trained models.
- **c. Evaluation metrics** — all six metrics shown for the selected model.
- **d. Confusion matrix + classification report** — rendered for the selected model.

It also shows the full cross-model comparison table computed at training time.

## How to Run

**Locally**
```bash
pip install -r requirements.txt
python model/train_models.py     # (re)generates .pkl models + test_data.csv
streamlit run app.py
```

**Deploy on Streamlit Community Cloud**
1. Push this repo to GitHub.
2. Go to <https://streamlit.io/cloud> → sign in with GitHub → **New App**.
3. Select the repo, branch `main`, main file `app.py` → **Deploy**.
4. Open the app, upload `test_data.csv`, pick a model, and view the results.

My Streamlit cloud repo
https://ml-brestcancer-vyrv6pww7talw6pscxfxnk.streamlit.app

## Notes

- `scikit-learn` is pinned to the training version so the saved `.pkl` pipelines
  unpickle safely on the cloud (mismatched versions are a common deployment issue).
- Each `.pkl` is a full `Pipeline`; scale-sensitive models (Logistic Regression,
  kNN) carry their own `StandardScaler`, so raw feature values can be passed
  straight to `predict`.
