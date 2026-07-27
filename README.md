# 🫀 Heart Disease Prediction & Classification

An enterprise-grade Data Science and Machine Learning project predicting the presence of heart disease using clinical patient attributes from the Cleveland dataset. 

Given the critical nature of medical diagnosis, the primary objective of this project is to **maximize Recall**, ensuring high-risk patients are accurately identified and minimizing life-threatening False Negatives.

👉 **[🚀 View Interactive Live Report](https://davitugl.github.io/heart-disease-prediction/)**

---

## 📊 Key Results & Performance Summary

| Metric | Baseline CV | Tuned Model (GridSearchCV) | Holdout Test Set (`X_test`) |
| :--- | :--- | :--- | :--- |
| **Champion Model** | Logistic Regression | Logistic Regression | **Logistic Regression** |
| **ROC-AUC Score** | 90.64% | 90.64% | **90.04%** |
| **Recall (Default 0.50)** | 87.04% | 91.65% | **87.88%** |
| **Recall (Optimal ~0.474)** | - | **95.42%** | **90.91%** |
| **False Negatives** | - | Reduced | **3** (Out of 33 diseased cases) |

---

## 🛠️ Project Architecture & Workflow

1. **Exploratory Data Analysis (EDA):** Data profiling, target distribution analysis, clinical correlation heatmaps, and domain-specific feature analyses (`cp`, `thalach`, `oldpeak`, `ca`).
2. **Modular Preprocessing Pipeline:** Integrated `ColumnTransformer` with `StandardScaler` for continuous features and `OneHotEncoder(handle_unknown='ignore')` for nominal categorical features.
3. **Cross-Validation Benchmarking:** Evaluated 4 distinct model families (Logistic Regression, SVC, Random Forest, Gradient Boosting) using 5-fold Stratified Cross-Validation.
4. **Hyperparameter Tuning:** Optimized parameters via `GridSearchCV(scoring='recall')`.
5. **Decision Threshold Optimization:** Lowered probability decision threshold from 0.50 to ~0.474 using Precision-Recall curve analysis to achieve **95%+ Recall**.
6. **Model Explainability:** Applied `SHAP` (SHapley Additive exPlanations) for model interpretability, verifying top clinical feature drivers.

---

## 🚀 Environment & Usage

### 1. Install Dependencies with Poetry
```bash
poetry install
```

### 2. Run the Interactive Marimo Notebook
```bash
poetry run marimo edit notebooks/heart_disease_classification.py
```

### 3. Export Standalone HTML Report
```bash
echo y | poetry run marimo export html notebooks/heart_disease_classification.py -o index.html
```

---

## 🌐 Live Interactive Dashboard
View the exported HTML report published on GitHub Pages:  
**https://davitugl.github.io/heart-disease-prediction/**
