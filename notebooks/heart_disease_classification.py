import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""
    # 🫀 Heart Disease Prediction Project
    """)
    return


@app.cell
def _():
    # Data Processing and Visualization
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap

    # Sklearn: Model Selection and Preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Sklearn: Machine Learning Models
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Sklearn: Metrics and Evaluation
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
        RocCurveDisplay,
    )
    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegression,
        Pipeline,
        RandomForestClassifier,
        RocCurveDisplay,
        StandardScaler,
        classification_report,
        cross_validate,
        mo,
        np,
        pd,
        plt,
        shap,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    # IMPORTING DATA
    heart_df = pd.read_csv("data/heart_disease.csv")
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
        mo.md("## 🫀 Heart Disease Prediction Project"),
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
        mo.md("## 🔍 Check Data Quality"),
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
        mo.md("## 🎯 Target Variable Distribution"),
        mo.hstack([
            target_summary,
            fig
        ], justify="start", gap=2)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🎯 Target Variable Conclusion & Strategy
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
        mo.md("## 📊 Statistical Overview"),
        mo.ui.table(stats, selection=None)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 📊 Stats
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

    _fig, _ax = plt.subplots(figsize=(10, 7))

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
        mo.md("## 🌡️ Feature Correlation Analysis"),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🌡️ Correlation Matrix
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
        mo.md("## 🚻 Heart Disease vs Sex"),
        mo.hstack([
            mo.ui.table(sex_target_summary, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🚻 Heart Disease vs Sex
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
        mo.md("## 🫀 Chest Pain vs Target (Diagnosis)"),
        mo.hstack([
            mo.ui.table(cp_summary_df, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🫀 Chest Pain Type
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
        mo.md("## 💓 Max Heart Rate (thalach) Analysis"),
        mo.hstack([
            mo.ui.table(thalach_stats_df, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 💓 Max Heart Rate (thalach)
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
        mo.md("## 📉 Age vs Heart Rate (Domain Knowledge)"),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 📉 Age vs Max Heart Rate
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
        mo.md("## 🎂 Age Distribution Analysis"),
        mo.hstack([
            mo.ui.table(age_stats_df, selection=None),
            _fig
        ], justify="start", gap=4)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🎂 Age Distribution vs Diagnosis
    * **Significant Overlap:** Unlike "Chest Pain" or "Heart Rate", the age distributions for Healthy and Disease groups overlap significantly. This means **Age alone is not a strong separator**.
    * **The Shift:** However, there is a visible trend: the **Disease (Red)** curve is shifted to the **right**. The peak risk appears around **58-60 years old**, whereas the healthy group peaks younger (~52-54).
    * **The "Confusion Zone":** Between ages **50 and 65**, the probability is quite mixed. The model will need other features (like `thalach` or `cp`) to make a confident decision in this age range.
    """)
    return


@app.cell
def _(ColumnTransformer, StandardScaler, heart_df_clean, train_test_split):
    # Train/Test Split, Scaling
    X = heart_df_clean.drop('target', axis=1)
    y = heart_df_clean['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
    ]

    # categorical and binary features
    passthrough_features = [
        'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform="pandas")
    return X_test, X_train, preprocessor, y_test, y_train


@app.cell
def _(
    GradientBoostingClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    X_train,
    cross_validate,
    mo,
    np,
    pd,
    preprocessor,
    y_train,
):
    # Models and Pipeline
    models = {
        'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    scoring_metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']

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
            'Recall': np.mean(cv_results['test_recall']),
            'ROC-AUC': np.mean(cv_results['test_roc_auc']),
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'Precision': np.mean(cv_results['test_precision']),
            'F1-Score': np.mean(cv_results['test_f1'])
        })

    results_df = pd.DataFrame(model_comparison).sort_values('Recall', ascending=False)

    mo.ui.table(results_df.round(4))
    return


@app.cell
def _(
    GridSearchCV,
    Pipeline,
    RandomForestClassifier,
    X_train,
    mo,
    preprocessor,
    y_train,
):
    # Hyperparameter Tuning via GridSearchCV 

    # Re-initializing the pipeline
    random_forest_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=param_grid,
        cv=5,                
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # model training
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    mo.md(f"""
    ### Optimization Results
    * **Best ROC-AUC:** `{grid_search.best_score_:.2%}`
    * **Estimators:** `{best_params['classifier__n_estimators']}`
    * **Max Depth:** `{best_params['classifier__max_depth']}`
    """)
    return (grid_search,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    X_test,
    classification_report,
    grid_search,
    mo,
    plt,
    y_test,
):
    # Predictions on test data. Confussion Matrix, Roc Curve
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'Heart Disease'])

    fig_eval, ax_eval = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_estimator(
        grid_search, X_test, y_test, 
        display_labels=['Healthy', 'Heart Disease'],
        cmap='Blues', ax=ax_eval[0]
    )
    ax_eval[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_estimator(
        grid_search, X_test, y_test, ax=ax_eval[1]
    )
    ax_eval[1].set_title("ROC Curve Analysis")
    ax_eval[1].grid(True, linestyle='--', alpha=0.6)

    mo.vstack([
        mo.md("## 🩺 Model Evaluation on Test Set"),
        mo.md(f"```\n{report}\n```"),
        mo.as_html(fig_eval)
    ])
    return


@app.cell
def _(mo):
    mo.md(f"""
    The model is good: **80% accuracy** and, more importantly, **91% recall**. The risk of missing a case is minimal.
    """)
    return


@app.cell
def _(grid_search, mo, pd, plt):
    # Feature Importance 

    names = grid_search.best_estimator_['preprocessor'].get_feature_names_out()
    importances = grid_search.best_estimator_['classifier'].feature_importances_

    top_features = pd.Series(importances, index=names).sort_values(ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))

    top_features.sort_values().plot(kind='barh', color='steelblue', ax=ax_imp)

    ax_imp.set_title("Top 10 Most Important Features (Random Forest)")
    ax_imp.set_xlabel("Gini Importance (Relative Weight)")
    plt.tight_layout()

    mo.vstack([
        mo.md("### **Feature Importance**"),
        mo.as_html(fig_imp)
    ])
    return


@app.cell
def _(X_test, grid_search, mo, plt, shap):
    # SHAP Explainer

    heart_model = grid_search.best_estimator_['classifier']
    transformed_data = grid_search.best_estimator_['preprocessor'].transform(X_test)

    explainer = shap.TreeExplainer(heart_model)
    shap_values_raw = explainer.shap_values(transformed_data)

    if isinstance(shap_values_raw, list):
        final_shap_values = shap_values_raw[1]
    else:
        final_shap_values = shap_values_raw[..., 1] if len(shap_values_raw.shape) == 3 else shap_values_raw
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(final_shap_values, transformed_data, show=False)
    plt.title("SHAP Interpretation: Impact on Heart Disease Diagnosis", fontsize=14)
    plt.tight_layout()

    mo.vstack([
        mo.md("### **🧬 Explaining Model Decisions**"),
        mo.as_html(plt.gcf())
    ])
    return


@app.cell
def _(mo):
    # Model Interpretation
    mo.md("""
    # Final Conclusion: Model Interpretation

    * Each dot represents one patient. It shows the internal logic of the Random Forest.
    * **How to read the results?**
    1. **cp (Chest Pain):** Red dots are on the right — higher chest pain levels significantly increase the risk.

    2. **thalach (Max Heart Rate):** Red dots are on the left. High heart rate acts as a "protective factor," decreasing the probability of disease.

    3. **oldpeak & ca:** High values (red) sharply increase the risk. These are the most "reliable" indicators for the model to detect danger.
    """)
    return


if __name__ == "__main__":
    app.run()
