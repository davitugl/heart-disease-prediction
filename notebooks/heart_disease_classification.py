import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # Data Processing and Visualization
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Sklearn: Model Selection and Preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Sklearn: Machine Learning Models
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    # Sklearn: Metrics and Evaluation
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
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
        RandomForestClassifier,
        SVC,
        StandardScaler,
        classification_report,
        confusion_matrix,
        mo,
        np,
        pd,
        plt,
        roc_auc_score,
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
        mo.md(f"### Total Records: {rows} | Total Columns: {columns}"),
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
        mo.md(f""" ### Duplicates: {duplicate_count}"""),
        mo.md(f""" ### Missing Values: {missing_values.sum()}"""),
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

    sns.countplot(x='target', data=heart_df_clean, palette=['#3498db', '#e74c3c'], hue='target', ax=ax)

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
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    heart_df_clean,
    mo,
    train_test_split,
):
    # Data Preprocessing Pipeline: Train/Test Split, Scaling, One-Hot Encoding (Preventing Data Leakage)
    X = heart_df_clean.drop('target', axis=1)
    y = heart_df_clean['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y 
    )

    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
    binary_features = ['sex', 'fbs', 'exang'] 

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('bin', 'passthrough', binary_features)
        ]
    )

    # Fit only on Train!
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    mo.md(f"""
    ### ⚙️ Data Preprocessing Pipeline!
    * **Steps:** Train/Test Split -> StandardScaler -> OneHotEncoder
    * **Training Data:** {X_train_scaled.shape}
    * **Testing Data:** {X_test_scaled.shape}
    """)
    return X_test_scaled, X_train_scaled, preprocessor, y_test, y_train


@app.cell
def _(classification_report, confusion_matrix, mo, plt, roc_auc_score, sns):
    # Create a universal function to evaluate any model!
    def evaluate_model(model, name, X_train, X_test, y_train, y_test):
        # Training (Fit)
        model.fit(X_train, y_train)

        # Predict on Test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate statistics
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        _fig, _ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=_ax,
                    xticklabels=['Predicted Healthy (0)', 'Predicted Disease (1)'], 
                    yticklabels=['Actual Healthy (0)', 'Actual Disease (1)'],
                    annot_kws={"size": 14, "weight": "bold"})
        _ax.set_title(f'Confusion Matrix: {name}', pad=15, weight='bold')
        plt.tight_layout()

        # Marimo UI
        return mo.vstack([
            mo.md(f"## Model Evaluation: {name}"),
            mo.md(f"**ROC-AUC Score:** `{roc_auc:.3f}` Closer to 1 is better"),
            mo.md(f"### 📊 Classification Report:\n```text\n{report}\n```"),
            _fig
        ])
    return (evaluate_model,)


@app.cell
def _(
    LogisticRegression,
    X_test_scaled,
    X_train_scaled,
    evaluate_model,
    y_test,
    y_train,
):
    # Logistic Regression
    log_reg = LogisticRegression(random_state=42, class_weight='balanced')

    # Call the func
    log_reg_results = evaluate_model(
        model=log_reg, 
        name="Logistic Regression (Baseline)", 
        X_train=X_train_scaled, 
        X_test=X_test_scaled, 
        y_train=y_train, 
        y_test=y_test
    )

    log_reg_results
    return


@app.cell
def _(
    GradientBoostingClassifier,
    RandomForestClassifier,
    SVC,
    X_test_scaled,
    X_train_scaled,
    evaluate_model,
    mo,
    y_test,
    y_train,
):
    # Random Forest
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_results = evaluate_model(
        model=rf_model, 
        name="Random Forest", 
        X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test
    )

    # Support Vector Machine (SVC)
    svm_model = SVC(probability=True, random_state=42, class_weight='balanced')
    svm_results = evaluate_model(
        model=svm_model, 
        name="Support Vector Machine (SVC)", 
        X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test
    )

    # Gradient Boosting 
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_results = evaluate_model(
        model=gb_model, 
        name="Gradient Boosting", 
        X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test
    )

    # All three model results
    mo.vstack([
        mo.md("## Baseline Comparison"),
        rf_results,
        mo.md("---"),
        svm_results,
        mo.md("---"),
        gb_results
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Baseline Conclusion: The Power of Simplicity
    * **Observation:** Against expectations, the simplest model (Logistic Regression) outperformed complex ensemble methods (Random Forest, Gradient Boosting) and SVC out-of-the-box, achieving the highest Recall (82%) and ROC-AUC (0.887).
    * **The "Why" (Occam's Razor):** Our dataset is relatively small (~300 records). In such cases, highly complex models tend to overfit the training data and generalize poorly on unseen data when using default parameters. Linear models like Logistic Regression are much more robust and stable here.
    * **Next Step (Hypothesis):** Can hyperparameter tuning "wake up" the complex models? We will now use `GridSearchCV` to optimize both Logistic Regression and Random Forest specifically for **Recall**, to see if we can catch those remaining 6 false negative patients.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    RandomForestClassifier,
    X_test_scaled,
    X_train_scaled,
    evaluate_model,
    mo,
    y_test,
    y_train,
):
    # Hyperparameter Tuning via GridSearchCV (Optimizing for Recall)

    # Logistic Regression
    log_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'class_weight': ['balanced']
    }

    # Init GridSearch (Focus on Recall!)
    log_grid = GridSearchCV(
        LogisticRegression(random_state=42), 
        log_param_grid, 
        cv=5,
        scoring='recall',
        n_jobs=-1
    )

    # Training
    log_grid.fit(X_train_scaled, y_train)
    best_log_model = log_grid.best_estimator_

    # Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        rf_param_grid, 
        cv=5, 
        scoring='recall',
        n_jobs=-1
    )

    rf_grid.fit(X_train_scaled, y_train)
    best_rf_model = rf_grid.best_estimator_

    # Evaluating the Tuned models
    tuned_log_results = evaluate_model(
        model=best_log_model, 
        name=f"Tuned Logistic Regression (Best C: {log_grid.best_params_['C']})", 
        X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test
    )

    tuned_rf_results = evaluate_model(
        model=best_rf_model, 
        name=f"Tuned Random Forest (Best Depth: {rf_grid.best_params_['max_depth']})", 
        X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test
    )

    mo.vstack([
        mo.md("## Hyperparameter Tuning Results (Optimized for Recall)"),
        tuned_log_results,
        mo.md("---"),
        tuned_rf_results
    ])
    return (best_log_model,)


@app.cell
def _(mo):
    mo.md("""
    ### Final Conclusion: Winning with GridSearchCV

    * **Tuned Logistic Regression** (with `C=10`) is the absolute winner. By optimizing for Recall, we successfully increased our primary metric from 82% to **85%** and pushed the ROC-AUC score to an excellent **0.900**.
    * **Clinical Impact:** In a medical context, this improvement is critical. We successfully identified an additional complex disease case, reducing our False Negatives from 6 down to 5.
    * **The "Small Data" Reality:** Despite hyperparameter tuning, the Random Forest classifier could not surpass the baseline Recall (staying at 82%) and performed worse on the ROC-AUC metric (0.852). This perfectly validates our earlier hypothesis: on small tabular datasets (~300 records), well-tuned linear models often outperform complex tree-based ensembles by preventing overfitting.
    """)
    return


@app.cell
def _(best_log_model, mo, pd, plt, preprocessor, sns):
    # Extracting feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Extracting coefficients from our champion model
    coefficients = best_log_model.coef_[0]

    # Creating a DataFrame and sorting descending
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    }).sort_values(by='Importance', ascending=False)

    # Plotting
    _fig, _ax = plt.subplots(figsize=(8, 6))

    # Color logic: Positive (Red/Danger), Negative (Blue/Safe)
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in feature_importance_df['Importance']]

    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=colors, hue='Feature', legend=False, ax=_ax)
    _ax.set_title('Feature Importance (Tuned Logistic Regression)', weight='bold', size=14, pad=15)
    _ax.set_xlabel('Coefficient Value (Impact on Disease Probability)', weight='bold')
    _ax.set_ylabel('Patient Features', weight='bold')
    _ax.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()

    # Output in Marimo
    mo.vstack([
        mo.md("""
        ## 🧠 How Does the Model Think?
    
    
        * **Red Bars (Positive > 0):** These features **increase** the probability of heart disease. The longer the bar, the more dangerous the symptom.
        * **Blue Bars (Negative < 0):** These features **decrease** the probability (indicate a healthier patient).
        """),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Project Conclusion & Business Value

    We extracted the underlying logic of our champion model (Tuned Logistic Regression). The model aligns perfectly with medical intuition:

    * **Primary Risk Factors:** Different types of chest pain (`cp_3`, `cp_2`, `cp_1`) are the strongest indicators pushing the model to predict the presence of heart disease.
    * **Health Indicators:** Certain fluoroscopy results (e.g., `ca_2`) act as the strongest negative coefficients, heavily reducing the probability of disease.
    """)
    return


if __name__ == "__main__":
    app.run()
