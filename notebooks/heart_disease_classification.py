# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.15",
#     "matplotlib>=3.11.1",
#     "numpy>=2.5.1",
#     "pandas>=3.0.5",
#     "scikit-learn>=1.9.0",
#     "seaborn>=0.13.2",
#     "shap>=0.52.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""
    # Heart Disease Prediction Project

    **Project Overview:** Predict the presence of heart disease using clinical patient attributes for early medical diagnosis.

    * **Primary Objective:** Maximize **Recall** to minimize False Negatives (ensuring high-risk patients are not missed).
    * **Workflow:** Data Profiling & EDA -> Modular Preprocessing (One-Hot & Scaling) -> ML Model Evaluation -> Decision Threshold Tuning -> Model Interpretation with SHAP.
    """)
    return


@app.cell
def _():
    # Core Data Processing, Visualization & Explainability
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap

    # Preprocessing, Selection & Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import (
        GridSearchCV,
        cross_val_predict,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Machine Learning Algorithms
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Metrics and Model Evaluation
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    return (
        ColumnTransformer,
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        RandomForestClassifier,
        SVC,
        StandardScaler,
        classification_report,
        confusion_matrix,
        cross_val_predict,
        cross_validate,
        mo,
        np,
        pd,
        plt,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
        shap,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    # IMPORTING DATA
    from pathlib import Path
    data_path = Path("data/heart_disease.csv") if Path("data/heart_disease.csv").exists() else Path("../data/heart_disease.csv")
    heart_df = pd.read_csv(data_path)
    return (heart_df,)


@app.cell
def _(mo):
    # DATA DICTIONARY
    mo.md("""
    | Column | Description | Details |
    | :--- | :--- | :--- |
    | **age** | Age | In years |
    | **sex** | Sex | 1 = male; 0 = female |
    | **cp** | Chest Pain Type | Types 0-3 |
    | **trestbps** | Resting Blood Pressure | In mmHg (on admission to the hospital) |
    | **chol** | Serum Cholesterol | In mg/dl |
    | **fbs** | Fasting Blood Sugar | > 120 mg/dl (1 = true; 0 = false) |
    | **restecg** | Resting ECG Results | Values 0-2 |
    | **thalach** | Max Heart Rate | Maximum heart rate achieved |
    | **exang** | Exercise Induced Angina | 1 = yes; 0 = no |
    | **oldpeak** | ST Depression | ST depression induced by exercise relative to rest |
    | **slope** | ST Slope | The slope of the peak exercise ST segment |
    | **ca** | Major Vessels | Number of major vessels colored by flourosopy (0-3) |
    | **thal** | Thalassemia | 1,3 = normal; 6 = fixed defect; 7 = reversable defect |
    | **target** | Diagnosis | 1 = disease; 0 = no disease |
    """)
    return


@app.cell
def _(heart_df, mo):
    # LOADING & SHOWING DATA
    rows, columns = heart_df.shape
    mo.vstack([
        mo.md("## Heart Disease Prediction"),
        mo.md(f"### Total Records: **{rows}** | Total Columns: **{columns}**"),
        mo.ui.table(
            heart_df,
            label="Anonymous Patient Data",
            selection=None,
            pagination=True,
        )
    ])
    return


@app.cell
def _(heart_df, mo, pd):
    # DATA QUALITY & PROFILING
    # Check for missing values and duplicates, unique values
    missing_values = heart_df.isna().sum()
    duplicate_count = heart_df.duplicated().sum()

    mo.vstack([
        mo.md("## Check Data Quality"),
        mo.md(f""" ### Duplicates: **{duplicate_count}**"""),
        mo.md(f""" ### Missing Values: **{missing_values.sum()}**"""),
        mo.ui.table(pd.DataFrame({"Data Type": heart_df.dtypes.astype(str), "Unique Values": heart_df.nunique() 
    }), selection=None)
    ])
    return


@app.cell
def _(heart_df):
    # Drop duplicate rows
    heart_df_clean = heart_df.drop_duplicates()
    return (heart_df_clean,)


@app.cell
def _(heart_df_clean, mo, pd, plt, sns):
    # TARGET DISTRIBUTION
    target_counts = heart_df_clean['target'].value_counts()
    target_percent = (heart_df_clean['target'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'

    target_summary = pd.DataFrame({
        "Count": target_counts,
        "Percentage": target_percent
    })

    # PLOT
    fig, ax = plt.subplots(figsize=(5, 6))

    sns.countplot(x='target', data=heart_df_clean, palette=['#3498db', '#e74c3c'], hue='target', ax=ax, legend=False)

    ax.set_title("Visual Balance (0 vs 1)")
    ax.set_xlabel("Diagnosis (0=Healthy, 1=Disease)")
    ax.set_ylabel("Count")

    mo.vstack([
        mo.md("## Target Variable Distribution"),
        mo.hstack([
            target_summary,
            fig
        ], justify="start", gap=2)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Target Variable Conclusion & Strategy
    * **Balanced Dataset:** The dataset is well-balanced (~54% Disease vs ~46% Healthy). This is excellent for model training.
    * **No Bias:** We don't need to apply complex resampling techniques (like SMOTE).
    * **Evaluation Metric:** Our primary focus is **Recall**. In medical diagnostics, minimizing False Negatives (missing a sick patient) is far more critical than overall Accuracy.
    """)
    return


@app.cell
def _(heart_df_clean, mo):
    # STATISTICAL SUMMARY
    stats = heart_df_clean.describe().T.round(2)
    mo.vstack([
        mo.md("## Statistical Overview"),
        mo.ui.table(stats, selection=None)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Stats
    * **Demographics:** The average patient age is **~54 years**, ranging from 29 to 77.
    * **Gender Imbalance:** The mean of `sex` is **0.68**, indicating that approximately **68%** of the dataset consists of male patients (assuming 1 = Male).
    * **Potential Outliers:**
      * **Cholesterol (`chol`):** The max value is **564 mg/dl**, which is extremely high compared to the mean (246).
      * **Blood Pressure (`trestbps`):** The max value reaches **200 mm Hg**, indicating hypertensive crisis cases.
    """)
    return


@app.cell
def _(heart_df_clean, mo, plt, sns):
    # Correlation Matrix
    corr_matrix = heart_df_clean.corr()

    _fig, _ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr_matrix, 
        annot=True,         
        fmt=".2f",           
        cmap="coolwarm",     
        linewidths=0.5,
        ax=_ax 
    )

    _ax.set_title("Correlation Matrix of Heart Disease Features", pad=20, fontsize=14, weight='bold')

    mo.vstack([
        mo.md("## Feature Correlation Analysis"),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Correlation Matrix
    * **Strongest Positive Features:** `cp` (Chest Pain, **0.43**) and `thalach` (Max Heart Rate, **0.42**) show the highest positive correlation with the target. As these values increase, the likelihood of heart disease increases.
    * **Strongest Negative Features:** `exang` (Exercise Induced Angina, **-0.44**) and `oldpeak` (ST Depression, **-0.43**) have the strongest inverse relationship.
    * **Multicollinearity:** Notice the strong correlation between `slope` and `oldpeak` (**-0.58**). This indicates some redundancy between these features, but generally, the features are well-distributed.
    """)
    return


@app.cell
def _(heart_df_clean, mo, pd, plt, sns):
    # Target By Sex
    _counts = pd.crosstab(heart_df_clean['sex'], heart_df_clean['target'])
    _percs = pd.crosstab(heart_df_clean['sex'], heart_df_clean['target'], normalize='index') * 100

    sex_target_summary = _counts.astype(str) + " (" + _percs.round(2).astype(str) + "%)"
    sex_target_summary['Total (N)'] = _counts.sum(axis=1)

    sex_target_summary.index = ['Female (0)', 'Male (1)']
    sex_target_summary.columns = ['Healthy (0)', 'Disease (1)', 'Total (N)']

    _fig, _ax = plt.subplots(figsize=(7, 6))

    sns.countplot(
        x='sex', 
        hue='target', 
        data=heart_df_clean,
        palette=['#3498db', '#e74c3c'], 
        ax=_ax 
    )

    _ax.bar_label(_ax.containers[0], labels=sex_target_summary['Healthy (0)'], padding=3)
    _ax.bar_label(_ax.containers[1], labels=sex_target_summary['Disease (1)'], padding=3)

    _ax.set_title("Heart Disease Frequency by Sex", pad=15, weight='bold')
    _ax.set_xlabel("Sex (0 = Female, 1 = Male)")
    _ax.set_ylabel("Amount of Patients")
    _ax.legend(["Healthy", "Disease"], title="Target")
    sns.despine(ax=_ax)

    mo.vstack([
        mo.md("## Heart Disease vs Sex"),
        mo.hstack([
            mo.ui.table(sex_target_summary, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Heart Disease vs Sex
    * **Demographic Imbalance:** The dataset is heavily skewed towards males (**206 males** vs **96 females**).
    * **High Risk in Females:** Interestingly, **75% of females** in this dataset have heart disease (72 out of 96). This suggests that if a patient is female in this specific dataset, the probability of diagnosis is very high.
    * **Male Distribution:** Males are more evenly distributed, with a slight lean towards being healthy (**~55% healthy** vs **~45% disease**).
    * **Conclusion:** Sex is a crucial feature. The model will likely learn that being female increases the probability of a positive diagnosis in this specific context.
    """)
    return


@app.cell
def _(heart_df_clean, mo, pd, plt, sns):
    # CHEST PAIN (CP) vs TARGET
    _cp_counts = pd.crosstab(heart_df_clean['cp'], heart_df_clean['target'])
    _cp_percs = pd.crosstab(heart_df_clean['cp'], heart_df_clean['target'], normalize='index') * 100

    _label_healthy = _cp_counts[0].astype(str) + " (" + _cp_percs[0].round(1).astype(str) + "%)"
    _label_disease = _cp_counts[1].astype(str) + " (" + _cp_percs[1].round(1).astype(str) + "%)"

    cp_summary_df = pd.DataFrame({
        "Healthy (0)": _label_healthy,
        "Disease (1)": _label_disease,
        "Total (N)": _cp_counts.sum(axis=1)
    })
    cp_summary_df.index = ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal (2)', 'Asymptomatic (3)']

    _fig, _ax = plt.subplots(figsize=(9, 8))
    sns.countplot(
        x='cp', 
        hue='target', 
        data=heart_df_clean,
        palette=['#9b59b6', '#e74c3c'],
        ax=_ax              
    )

    _ax.bar_label(_ax.containers[0], labels=_label_healthy, padding=3)
    _ax.bar_label(_ax.containers[1], labels=_label_disease, padding=3)

    _ax.set_title("Heart Disease Rate by Chest Pain Type", pad=15, weight='bold')
    _ax.set_xlabel("Chest Pain Type")
    _ax.set_ylabel("Count of Patients")
    _ax.legend(["Healthy", "Disease"], title="Target", loc='upper left')
    _ax.set_ylim(0, 130) 
    sns.despine(ax=_ax) 

    mo.vstack([
        mo.md("## Chest Pain vs Target (Diagnosis)"),
        mo.hstack([
            mo.ui.table(cp_summary_df, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Chest Pain Type
    * **The "Typical" Paradox:** Surprisingly, **Type 0 (Typical Angina)** is the safest category. **72.7%** of patients with this pain type are healthy. This is a crucial insight: having "typical" pain doesn't guarantee heart disease in this dataset.
    * **The Danger Zone (Type 2):** **Type 2 (Non-anginal pain)** is a massive red flag. Out of 86 patients, **79.1%** have heart disease. This will be a dominant predictor for the model.
    * **High Risk in Types 1 & 3:** Types 1 and 3 also show very high disease rates (82% and 69.6% respectively), making any pain type *other than 0* a strong indicator of risk.
    * **Conclusion:** This variable provides excellent "Separability". If $cp > 0$, the risk skyrockets.
    """)
    return


@app.cell
def _(heart_df_clean, mo, plt, sns):
    # THALACH vs Target
    _thalach_stats = heart_df_clean.groupby('target')['thalach'].describe()[['count', 'mean', '50%', 'max']]

    _thalach_stats.columns = ['Count', 'Mean (Average)', 'Median', 'Max Rate']
    _thalach_stats.index = ['Healthy (0)', 'Disease (1)']
    thalach_stats_df = _thalach_stats.round(1)

    _fig, _ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        x='thalach', 
        hue='target', 
        data=heart_df_clean,
        fill=True, 
        palette=['#2ecc71', '#e74c3c'],
        common_norm=False, 
        alpha=0.4,
        ax=_ax              
    )

    _ax.axvline(thalach_stats_df.loc['Healthy (0)', 'Mean (Average)'], color='#27ae60', linestyle='--', label='Healthy Mean', linewidth=2)
    _ax.axvline(thalach_stats_df.loc['Disease (1)', 'Mean (Average)'], color='#c0392b', linestyle='--', label='Disease Mean', linewidth=2)

    _ax.set_title("Distribution of Max Heart Rate (thalach) by Diagnosis", pad=15, weight='bold')
    _ax.set_xlabel("Maximum Heart Rate Achieved")
    _ax.set_ylabel("Density (Probability)")
    _ax.legend()
    sns.despine(ax=_ax)

    mo.vstack([
        mo.md("## Max Heart Rate (thalach) Analysis"),
        mo.hstack([
            mo.ui.table(thalach_stats_df, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Max Heart Rate (thalach)
    * **Clear Separation:** There is a distinct difference between the two groups. The distributions (bumps) are far apart, which makes `thalach` an excellent predictor.
    * **The Trend:** Patients in the **Disease (1)** group tend to have a **significantly higher** maximum heart rate (Mean: **158.4**) compared to the **Healthy (0)** group (Mean: **139.1**).
    * **Correlation Confirmation:** This aligns with the correlation matrix ($r=0.42$), showing that a higher heart rate is positively associated with the target diagnosis in this specific dataset.
    """)
    return


@app.cell
def _(heart_df_clean, mo, np, plt, sns):
    # AGE vs THALACH
    _fig, _ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x='age', 
        y='thalach', 
        data=heart_df_clean,
        hue='target', 
        palette=['#3498db', '#e74c3c'],
        alpha=0.7, 
        s=70,
        ax=_ax           
    )

    _x_points = np.linspace(heart_df_clean['age'].min(), heart_df_clean['age'].max(), 100)
    _y_points = 220 - _x_points
    _ax.plot(_x_points, _y_points, color='grey', linestyle='--', label='Theoretical Max (220-Age)', alpha=0.6, linewidth=2)

    _ax.set_title("Age vs Max Heart Rate: The Impact of Disease", pad=15, weight='bold')
    _ax.set_xlabel("Age (Years)")
    _ax.set_ylabel("Max Heart Rate (thalach)")

    _handles, _labels = _ax.get_legend_handles_labels()

    _clean_labels = ['Healthy (0)' if l == '0' else 'Disease (1)' if l == '1' else l for l in _labels]

    _ax.legend(_handles, _clean_labels, bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)

    sns.despine(ax=_ax)
    _fig.tight_layout()

    mo.vstack([
        mo.md("## Age vs Heart Rate (Domain Knowledge)"),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Age vs Max Heart Rate
    * **Natural Decline:** The plot clearly shows that as **Age increases**, the **Max Heart Rate decreases**. This follows the natural physiological trend (roughly $220 - Age$).
    * **The "Risk Layer":** Notice the vertical separation. The **Disease (Red)** points tend to be positioned **higher** than the Healthy (Blue) points across most ages.
    * **Combined Power:** While Age alone had a lot of overlap (as seen in the KDE plot), combining it with Heart Rate reveals a clearer pattern. A 60-year-old with a heart rate of 170 is much more likely to be in the "Disease" group than a 60-year-old with a heart rate of 130.
    """)
    return


@app.cell
def _(heart_df_clean, mo, plt, sns):
    # AGE vs target
    _age_stats = heart_df_clean.groupby('target')['age'].describe()[['count', 'mean', '50%', 'max']]

    _age_stats.columns = ['Count', 'Mean Age', 'Median', 'Oldest']
    _age_stats.index = ['Healthy (0)', 'Disease (1)']
    age_stats_df = _age_stats.round(1)

    _fig, _ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        x='age', 
        hue='target', 
        data=heart_df_clean,
        fill=True, 
        palette=['#2ecc71', '#e74c3c'],
        common_norm=False, 
        alpha=0.4,
        ax=_ax         
    )

    _ax.axvline(age_stats_df.loc['Healthy (0)', 'Mean Age'], color='#27ae60', linestyle='--', label='Healthy Mean', linewidth=2)
    _ax.axvline(age_stats_df.loc['Disease (1)', 'Mean Age'], color='#c0392b', linestyle='--', label='Disease Mean', linewidth=2)

    _ax.set_title("Age Distribution by Diagnosis", pad=15, weight='bold')
    _ax.set_xlabel("Age (Years)")
    _ax.set_ylabel("Density")
    _ax.legend() 
    sns.despine(ax=_ax)

    mo.vstack([
        mo.md("## Age Distribution Analysis"),
        mo.hstack([
            mo.ui.table(age_stats_df, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Age Distribution vs Diagnosis
    * **Significant Overlap:** Unlike "Chest Pain" or "Heart Rate", the age distributions for Healthy and Disease groups overlap significantly. This means **Age alone is not a strong separator**.
    * **The Shift:** However, there is a visible trend: the **Disease (Red)** curve is shifted to the **right**. The peak risk appears around **58-60 years old**, whereas the healthy group peaks younger (~52-54).
    * **The "Confusion Zone":** Between ages **50 and 65**, the probability is quite mixed. The model will need other features (like `thalach` or `cp`) to make a confident decision in this age range.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data Preprocessing & Train/Test Split

    * **Train/Test Split:** 80/20 stratified split to maintain exact target distribution across sets.
    * **Feature Engineering (`ColumnTransformer`):**
      - **Numeric Scaling:** Standardize continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) using `StandardScaler`.
      - **Categorical Encoding:** One-Hot Encode multi-category nominal features (`cp`, `restecg`, `slope`, `thal`).
      - **Binary Features:** Keep 0/1 binary features (`sex`, `fbs`, `exang`, `ca`) as passthrough.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    heart_df_clean,
    train_test_split,
):
    # Train/Test Split (80/20 stratified)
    X = heart_df_clean.drop('target', axis=1)
    y = heart_df_clean['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature categorization
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['cp', 'restecg', 'slope', 'thal']
    binary_features = ['sex', 'fbs', 'exang', 'ca']

    # Professional preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('pass', 'passthrough', binary_features)
        ],
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform="pandas")
    return X_test, X_train, preprocessor, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## Model Cross-Validation Benchmarking

    * **Cross-Validation Strategy:** 5-fold Stratified Cross-Validation on `X_train` to evaluate baseline generalization.
    * **Model Diversity:** Benchmarking across 4 distinct model algorithms:
      - **Logistic Regression:** Linear probabilistic baseline.
      - **Support Vector Machine (SVC):** Non-linear kernel-based classification.
      - **Random Forest:** Ensemble bagging decision trees.
      - **Gradient Boosting:** Sequential ensemble boosting.
    * **Evaluation Focus:** Primary metric is **Recall (Mean ± Std)** to assess model confidence and stability across folds.
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    SVC,
    X_train,
    cross_validate,
    mo,
    np,
    pd,
    preprocessor,
    y_train,
):
    # Model baseline candidates
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    scoring_metrics = ['recall', 'roc_auc', 'accuracy', 'precision', 'f1']
    model_comparison = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        cv_results = cross_validate(
            pipeline, 
            X_train, 
            y_train, 
            cv=5, 
            scoring=scoring_metrics,
            n_jobs=-1
        )

        model_comparison.append({
            'Model': name,
            'Recall (Mean)': np.mean(cv_results['test_recall']),
            'Recall (Std)': np.std(cv_results['test_recall']),
            'ROC-AUC': np.mean(cv_results['test_roc_auc']),
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'F1-Score': np.mean(cv_results['test_f1'])
        })

    results_df = pd.DataFrame(model_comparison).sort_values('Recall (Mean)', ascending=False)

    mo.vstack([
        mo.md("### Baseline Model Comparison"),
        mo.ui.table(results_df.round(4), selection=None)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Baseline Insights & Next Strategy

    * **Top Performer:** **Logistic Regression** achieves the highest baseline **Recall (~87.0%)** and **ROC-AUC (~90.6%)** with the lowest fold variance (+-6.1%).
    * **Key Takeaway:** On small tabular clinical datasets (~300 samples), linear models with balanced class weights generalize exceptionally well without overfitting.
    * **Next Step Strategy:**
      - Tune **Logistic Regression** and **Random Forest** via `GridSearchCV`.
      - Optimize the **Decision Threshold** (lowering from 0.5 to ~0.35) to boost **Recall beyond 95%** for medical risk minimization.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Hyperparameter Tuning via GridSearchCV

    * **Optimization Goal:** Maximize **Recall** (`scoring='recall'`) using 5-fold Stratified Cross-Validation on `X_train`.
    * **Tuned Models:**
      - **Logistic Regression Pipeline:** Tuning regularization strength (C in [0.01, 0.1, 1.0, 10.0]), solver (`lbfgs`, `liblinear`), and class weighting (`balanced`, `None`).
      - **Random Forest Pipeline:** Tuning number of trees (n in [100, 200]), max depth ([5, 10, None]), min samples split ([2, 5, 10]), and tree class weighting (`balanced`, `balanced_subsample`).
    """)
    return


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    X_train,
    mo,
    preprocessor,
    y_train,
):
    # Logistic Regression GridSearchCV
    logreg_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    logreg_param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'lbfgs'],
        'classifier__class_weight': ['balanced', None]
    }

    grid_logreg = GridSearchCV(
        estimator=logreg_pipe,
        param_grid=logreg_param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )
    grid_logreg.fit(X_train, y_train)

    # Random Forest GridSearchCV
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }

    grid_rf = GridSearchCV(
        estimator=rf_pipe,
        param_grid=rf_param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)

    best_logreg_model = grid_logreg.best_estimator_
    best_rf_model = grid_rf.best_estimator_

    summary_md = mo.md(f"""
    ### GridSearchCV Optimization Results

    * **Logistic Regression Best Recall:** `{grid_logreg.best_score_:.2%}`
      - Best Parameters: `C = {grid_logreg.best_params_['classifier__C']}`, `solver = '{grid_logreg.best_params_['classifier__solver']}'`, `class_weight = {grid_logreg.best_params_['classifier__class_weight']}`
    * **Random Forest Best Recall:** `{grid_rf.best_score_:.2%}`
      - Best Parameters: `n_estimators = {grid_rf.best_params_['classifier__n_estimators']}`, `class_weight = '{grid_rf.best_params_['classifier__class_weight']}'`

    **Conclusion:** **Logistic Regression** achieved the highest cross-validated Recall (**{grid_logreg.best_score_:.2%}**), confirming it as our Champion Model for tuning.
    """)

    mo.vstack([summary_md])
    return (best_logreg_model,)


@app.cell
def _(mo):
    mo.md("""
    ## Decision Threshold Optimization (Recall Optimization)

    * **Clinical Rationale:** In medical diagnosis, missing a sick patient (False Negative) carries a severe risk. Standard binary classification uses a default probability threshold of $0.5$.
    * **Strategy:** Analyze the Precision-Recall curve on `X_train` using 5-fold cross-validated probabilities (`cross_val_predict`), and select an **Optimal Threshold (~0.474)** that guarantees **Recall >= 95%**.
    """)
    return


@app.cell
def _(
    X_train,
    best_logreg_model,
    cross_val_predict,
    mo,
    np,
    plt,
    precision_recall_curve,
    sns,
    y_train,
):
    # Cross-validated predicted probabilities for threshold tuning
    y_probs_cv = cross_val_predict(best_logreg_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, y_probs_cv)

    # Find optimal threshold achieving Recall >= 0.95
    target_recall = 0.95
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    opt_idx = valid_indices[-1] if len(valid_indices) > 0 else 0

    optimal_threshold = float(thresholds[opt_idx])
    opt_recall = float(recalls[opt_idx])
    opt_precision = float(precisions[opt_idx])

    # Plot Precision-Recall vs Threshold
    fig_thresh, ax_thresh = plt.subplots(figsize=(12, 6))

    ax_thresh.plot(thresholds, precisions[:-1], label='Precision', color='#3498db', linewidth=2)
    ax_thresh.plot(thresholds, recalls[:-1], label='Recall', color='#e74c3c', linewidth=2)

    ax_thresh.axvline(optimal_threshold, color='#27ae60', linestyle='--', linewidth=2, label=f'Optimal Threshold ({optimal_threshold:.3f})')
    ax_thresh.axvline(0.5, color='grey', linestyle=':', label='Default Threshold (0.50)')

    ax_thresh.set_title("Decision Threshold Trade-off (Recall vs Precision)", pad=15, weight='bold')
    ax_thresh.set_xlabel("Probability Threshold")
    ax_thresh.set_ylabel("Metric Score")
    ax_thresh.set_ylim(0.4, 1.02)
    ax_thresh.legend(loc='lower left')
    ax_thresh.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax_thresh)

    thresh_summary = mo.md(f"""
    ### Threshold Optimization

    * **Default Threshold (0.50):** Baseline Recall ~91.6%
    * **Optimal Threshold ({optimal_threshold:.3f}):** Pushes **Recall to {opt_recall:.2%}** (Precision: {opt_precision:.2%})
    * **Impact:** Lowering the probability threshold to **{optimal_threshold:.3f}** captures 95%+ of heart disease cases while maintaining high diagnostic precision.
    """)

    mo.vstack([
        thresh_summary,
        fig_thresh
    ])
    return (optimal_threshold,)


@app.cell
def _(mo):
    mo.md("""
    ## Holdout Test Set Evaluation & Confusion Matrix Analysis

    * **Final Holdout Evaluation:** Testing our tuned Champion Model on the unseen 20% holdout test set (`X_test`, `y_test`).
    * **Threshold Comparison:** Comparing performance between Default Threshold ($0.50$) and Optimal Threshold ($0.474$).
    * **Visualizations:** Side-by-side **Confusion Matrices** and **ROC Curve Analysis**.
    """)
    return


@app.cell
def _(
    X_test,
    best_logreg_model,
    classification_report,
    confusion_matrix,
    mo,
    optimal_threshold,
    pd,
    plt,
    roc_auc_score,
    roc_curve,
    sns,
    y_test,
):
    # Predict probabilities on unseen holdout test set
    y_test_probs = best_logreg_model.predict_proba(X_test)[:, 1]

    # Predictions at Default vs Optimal Thresholds
    y_pred_default = (y_test_probs >= 0.50).astype(int)
    y_pred_optimal = (y_test_probs >= optimal_threshold).astype(int)

    cm_def = confusion_matrix(y_test, y_pred_default)
    cm_opt = confusion_matrix(y_test, y_pred_optimal)

    # Side-by-Side Confusion Matrices Plot
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_def, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0], cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    axes_cm[0].set_title(f"Default Threshold (0.50)\nFalse Negatives: {cm_def[1, 0]}", pad=12, weight='bold')
    axes_cm[0].set_xlabel("Predicted Label")
    axes_cm[0].set_ylabel("True Label")
    axes_cm[0].set_xticklabels(['Healthy (0)', 'Disease (1)'])
    axes_cm[0].set_yticklabels(['Healthy (0)', 'Disease (1)'])

    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes_cm[1], cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    axes_cm[1].set_title(f"Optimal Threshold ({optimal_threshold:.3f})\nFalse Negatives: {cm_opt[1, 0]}", pad=12, weight='bold')
    axes_cm[1].set_xlabel("Predicted Label")
    axes_cm[1].set_ylabel("True Label")
    axes_cm[1].set_xticklabels(['Healthy (0)', 'Disease (1)'])
    axes_cm[1].set_yticklabels(['Healthy (0)', 'Disease (1)'])

    plt.tight_layout()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    test_auc = roc_auc_score(y_test, y_test_probs)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    ax_roc.plot(fpr, tpr, color='#2980b9', linewidth=2.5, label=f'Tuned Logistic Regression (AUC = {test_auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.50)')

    ax_roc.set_title("ROC Curve on Holdout Test Set", pad=15, weight='bold')
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax_roc)

    # Classification metrics table on holdout test set
    rep_def = classification_report(y_test, y_pred_default, output_dict=True)['1']
    rep_opt = classification_report(y_test, y_pred_optimal, output_dict=True)['1']

    metrics_df = pd.DataFrame([
        {
            'Threshold': 'Default (0.50)',
            'Test Recall': f"{rep_def['recall']:.2%}",
            'Test Precision': f"{rep_def['precision']:.2%}",
            'Test F1-Score': f"{rep_def['f1-score']:.2%}",
            'False Negatives': cm_def[1, 0]
        },
        {
            'Threshold': f"Optimal ({optimal_threshold:.3f})",
            'Test Recall': f"{rep_opt['recall']:.2%}",
            'Test Precision': f"{rep_opt['precision']:.2%}",
            'Test F1-Score': f"{rep_opt['f1-score']:.2%}",
            'False Negatives': cm_opt[1, 0]
        }
    ])

    summary_test_md = mo.md(f"""
    ### Holdout Test Set Results Summary

    * **ROC-AUC Score on Unseen Test Data:** `{test_auc:.4f}` (**~{test_auc:.1%}**)
    * **False Negative Reduction:** Lowering the threshold from $0.50$ to ${optimal_threshold:.3f}$ reduced **False Negatives from {cm_def[1, 0]} down to {cm_opt[1, 0]}**, pushing Test Recall to **{rep_opt['recall']:.2%}**.
    """)

    mo.vstack([
        summary_test_md,
        mo.md("#### Holdout Metrics Comparison"),
        mo.ui.table(metrics_df, selection=None),
        mo.md("#### Confusion Matrices (Default vs Optimal)"),
        fig_cm,
        mo.md("#### ROC Curve Analysis"),
        fig_roc
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Model Explainability with SHAP (SHapley Additive exPlanations)

    * **Explainability Rationale:** Machine Learning models in healthcare must not act as a "black box". Clinicians require clear feature contribution explanations for every prediction.
    * **SHAP Framework:** Using Shapley values from game theory to quantify the exact contribution (impact on log-odds of heart disease) for each patient feature (`cp`, `thalach`, `oldpeak`, `ca`, etc.).
    * **Visualizations:**
      - **SHAP Summary Beeswarm Plot:** Overall feature importance and direction of feature impact on heart disease risk.
      - **Top 10 Feature Importance Bar Plot:** Absolute mean Shapley impact ranking.
    """)
    return


@app.cell
def _(X_train, best_logreg_model, mo, np, pd, plt, shap):
    # Extract fitted preprocessor and transformed training features
    fitted_preprocessor = best_logreg_model.named_steps['preprocessor']
    X_train_transformed = fitted_preprocessor.transform(X_train)
    feature_names = list(X_train_transformed.columns)

    # Extract fitted classifier from pipeline
    classifier = best_logreg_model.named_steps['classifier']

    # Initialize SHAP Linear Explainer
    explainer = shap.LinearExplainer(classifier, X_train_transformed)
    shap_values = explainer(X_train_transformed)

    # SHAP Summary Beeswarm Plot
    fig_shap, ax_shap = plt.subplots(figsize=(12, 6))
    plt.title("SHAP Summary Plot (Feature Impact on Heart Disease Prediction)", fontsize=13, weight='bold', pad=15)
    shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()

    # Top Feature Importance Bar Chart
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP| Value': mean_abs_shap
    }).sort_values('Mean |SHAP| Value', ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(12, 6))
    ax_imp.barh(shap_importance_df['Feature'][::-1], shap_importance_df['Mean |SHAP| Value'][::-1], color='#2980b9')
    ax_imp.set_title("Top 10 Most Influential Features (SHAP Importance)", pad=15, weight='bold')
    ax_imp.set_xlabel("Mean |SHAP Value| (Impact on Model Output)")
    ax_imp.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    final_conclusion_md = mo.md("""
    ### Executive Project Conclusion

    1. **Data Preprocessing & Modular Pipeline:** Engineered a robust `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` and `StandardScaler` to prevent data leakage and handle rare categorical levels.
    2. **Model Evaluation:** **Logistic Regression** demonstrated superior generalization on small clinical tabular data ($AUC = 0.9004$), outperforming complex tree ensembles.
    3. **Clinical Optimization:** Lowering the probability threshold to **~0.474** pushed **Recall to 95%+**, dramatically reducing critical False Negatives.
    4. **Feature Interpretability:** SHAP analysis confirmed that chest pain type (`cp`), maximum heart rate (`thalach`), ST depression (`oldpeak`), and major vessels colored by fluoroscopy (`ca`) are the primary clinical drivers of heart disease diagnosis.
    """)

    mo.vstack([
        mo.md("### SHAP Feature Interpretability"),
        fig_shap,
        mo.md("### Top 10 Feature Importance"),
        fig_imp,
        final_conclusion_md
    ])
    return


if __name__ == "__main__":
    app.run()
