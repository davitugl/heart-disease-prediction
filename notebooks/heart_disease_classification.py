import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # IMPORTING LIBRARIES
    import marimo as mo

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
    return (
        ConfusionMatrixDisplay,
        GridSearchCV,
        KNeighborsClassifier,
        LogisticRegression,
        Pipeline,
        RandomForestClassifier,
        RepeatedStratifiedKFold,
        StandardScaler,
        classification_report,
        confusion_matrix,
        mo,
        np,
        pd,
        plt,
        roc_auc_score,
        roc_curve,
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
            pagination=True
        )
    ])
    return


@app.cell
def _(heart_df, mo, pd):
    # DATA QUALITY & PROFILING
    # Check for missing values and duplicates
    missing_values = heart_df.isna().sum()
    duplicate_count = heart_df.duplicated().sum()

    mo.vstack([
        mo.md("## 🔍 Data Quality Check"),
        mo.md(f""" ### Duplicates: {duplicate_count}, Missing Values: {missing_values.sum()}"""),
        pd.DataFrame({
            "Data Type": heart_df.dtypes, 
            "Null Count": missing_values
        })
    ])
    return


@app.cell
def _(heart_df):
    # Drop duplicate rows
    heart_df.drop_duplicates(inplace=True)
    return


@app.cell
def _(heart_df, mo, pd, plt, sns):
    # TARGET DISTRIBUTION
    target_counts = heart_df['target'].value_counts()
    target_percent = (heart_df['target'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'

    target_summary = pd.DataFrame({
        "Count": target_counts,
        "Percentage": target_percent
    })

    # PLOT CREATION
    plt.figure(figsize=(5, 6))
    sns.countplot(x='target', data=heart_df, palette=['#3498db', '#e74c3c'], hue='target')
    plt.title("Visual Balance (0 vs 1)")
    plt.xlabel("Diagnosis (0=Healthy, 1=Disease)")
    plt.ylabel("Count")

    mo.vstack([
        mo.md("## 🎯 Target Variable Distribution"),
        mo.hstack([
            target_summary,        
            plt.gca()
        ], justify="start", gap=2)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🎯 Target Variable
    * **Balanced Dataset:** The dataset is well-balanced (~54% Disease vs ~46% Healthy). This is excellent for training.
    * **No Bias:** We don't need to apply complex resampling techniques (like SMOTE) because the model won't be biased toward one class.
    """)
    return


@app.cell
def _(heart_df, mo):
    # STATISTICAL SUMMARY
    stats = heart_df.describe().T.round(2)
    mo.vstack([
        mo.md("## 📊 Statistical Overview"),
        stats
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 📊 Stats
    * **Demographics:** The average patient age is **~54 years**, ranging from 29 to 77.
    * **Gender Imbalance:** The mean of `sex` is **0.68**, indicating that approximately **68%** of the dataset consists of male patients (assuming 1 = Male).
    * **Potential Outliers:** * **Cholesterol (`chol`):** The max value is **564 mg/dl**, which is extremely high compared to the mean (246).
    * **Blood Pressure (`trestbps`):** The max value reaches **200 mm Hg**, indicating hypertensive crisis cases.
    """)
    return


@app.cell
def _(heart_df, mo, plt, sns):
    # CORRELATION MATRIX
    corr_matrix = heart_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix, 
        annot=True,         
        fmt=".2f",           
        cmap="coolwarm",     
        linewidths=0.5    
    )

    plt.title("Correlation Matrix of Heart Disease Features")

    # Display
    mo.vstack([
        mo.md("## 🌡️ Feature Correlation Analysis"),
        plt.gca()
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
def _(heart_df, mo, pd, plt, sns):
    # TARGET BY SEX
    counts = pd.crosstab(heart_df['sex'], heart_df['target'])
    percs = pd.crosstab(heart_df['sex'], heart_df['target'], normalize='index') * 100
    sex_target = counts.astype(str) + " (" + percs.round(2).astype(str) + "%)"

    sex_target['Total (N)'] = counts.sum(axis=1)
    sex_target.index, sex_target.columns = ['Female', 'Male'], ['Healthy', 'Disease', 'Total (N)']

    plt.figure(figsize=(6, 6))
    ax_sex = sns.countplot(x='sex', hue='target', data=heart_df, palette=['#3498db', '#e74c3c'])

    ax_sex.bar_label(ax_sex.containers[0], labels=sex_target['Healthy'], padding=3)
    ax_sex.bar_label(ax_sex.containers[1], labels=sex_target['Disease'], padding=3)

    plt.title("Heart Disease Frequency by Sex")
    plt.xlabel("Sex (0 = Female, 1 = Male)")
    plt.ylabel("Amount")
    plt.legend(["Healthy", "Disease"])

    mo.vstack([
        mo.md("## 🚻 Heart Disease vs Sex"),
        mo.hstack([
            sex_target,
            plt.gca()
        ], justify="start", gap=2)
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
def _(heart_df, mo, pd, plt, sns):
    # CHEST PAIN (CP) vs TARGET
    cp_counts = pd.crosstab(heart_df['cp'], heart_df['target'])
    cp_percs = pd.crosstab(heart_df['cp'], heart_df['target'], normalize='index') * 100

    label_healthy = cp_counts[0].astype(str) + " (" + cp_percs[0].round(1).astype(str) + "%)"
    label_disease = cp_counts[1].astype(str) + " (" + cp_percs[1].round(1).astype(str) + "%)"

    cp_summary_df = pd.DataFrame({
        "Healthy (0)": label_healthy,
        "Disease (1)": label_disease,
        "Total (N)": cp_counts.sum(axis=1)
    })

    cp_summary_df.index = ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal (2)', 'Asymptomatic (3)']

    # PLOT 
    plt.figure(figsize=(9, 8))
    ax_cp = sns.countplot(x='cp', hue='target', data=heart_df, palette=['#9b59b6', '#e74c3c'])

    ax_cp.bar_label(ax_cp.containers[0], labels=label_healthy, padding=3)
    ax_cp.bar_label(ax_cp.containers[1], labels=label_disease, padding=3)

    plt.title("Heart Disease Rate by Chest Pain Type")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Count")
    plt.legend(["Healthy", "Disease"], loc='upper left')
    plt.ylim(0, 130)

    # MARIMO UI
    mo.vstack([
        mo.md("## 🫀 Chest Pain vs Target (Diagnosis)"),
        mo.hstack([
            cp_summary_df,
            plt.gca()
        ], justify="start", gap=2)
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
def _(heart_df, mo, plt, sns):
    # THALACH vs Target
    thalach_stats = heart_df.groupby('target')['thalach'].describe()[['count', 'mean', '50%', 'max']]

    thalach_stats.columns = ['Count', 'Mean (Average)', 'Median', 'Max Rate']
    thalach_stats.index = ['Healthy (0)', 'Disease (1)']
    thalach_stats = thalach_stats.round(1)

    plt.figure(figsize=(10, 6))

    sns.kdeplot(x='thalach', hue='target', data=heart_df, fill=True, palette=['green', 'red'], common_norm=False, alpha=0.3)

    plt.axvline(thalach_stats.loc['Healthy (0)', 'Mean (Average)'], color='green', linestyle='--', label='Healthy Mean')
    plt.axvline(thalach_stats.loc['Disease (1)', 'Mean (Average)'], color='red', linestyle='--', label='Disease Mean')

    plt.title("Distribution of Max Heart Rate (thalach) by Target")
    plt.xlabel("Max Heart Rate")
    plt.legend(["Healthy", "Disease"])

    mo.vstack([
        mo.md("## 💓 Max Heart Rate (thalach) Analysis"),
        mo.hstack([
            thalach_stats,
            plt.gca()
        ], justify="start", gap=2)
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
def _(heart_df, mo, np, plt, sns):
    # AGE vs THALACH
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='thalach', data=heart_df, hue='target', palette=['blue', 'red'], alpha=0.7, s=70)

    x_points = np.linspace(29, 77, 100)
    y_points = 220 - x_points
    plt.plot(x_points, y_points, color='grey', linestyle='--', label='Theoretical Max (220-Age)', alpha=0.5)

    plt.title("Age vs Max Heart Rate: The Impact of Disease")
    plt.xlabel("Age (Years)")
    plt.ylabel("Max Heart Rate (thalach)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # MARIMO UI
    mo.vstack([
        mo.md("## 📉 Age vs Heart Rate (Correlation Analysis)"),
        plt.gca()
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 📉 Age vs Max Heart Rate
    * **Natural Decline:** The plot clearly shows that as **Age increases**, the **Max Heart Rate decreases**. This follows the natural physiological trend (roughly $220 - Age$).
    * **The "Risk Layer":** Notice the vertical separation. The **Disease (Red)** points tend to be positioned **higher** than the Healthy (Green) points across most ages.
    * **Combined Power:** While Age alone had a lot of overlap (as seen in the KDE plot), combining it with Heart Rate reveals a clearer pattern. A 60-year-old with a heart rate of 170 is much more likely to be in the "Disease" group than a 60-year-old with a heart rate of 130.
    """)
    return


@app.cell
def _(heart_df, mo, plt, sns):
    # AGE vs target
    age_stats = heart_df.groupby('target')['age'].describe()[['count', 'mean', '50%', 'max']]

    age_stats.columns = ['Count', 'Mean Age', 'Median', 'Oldest']
    age_stats.index = ['Healthy (0)', 'Disease (1)']
    age_stats = age_stats.round(1)

    # PLOT
    plt.figure(figsize=(10, 6))

    # Distribution
    sns.kdeplot(x='age', hue='target', data=heart_df, fill=True, palette=['green', 'red'], common_norm=False, alpha=0.3)

    plt.axvline(age_stats.loc['Healthy (0)', 'Mean Age'], color='green', linestyle='--', label='Healthy Mean')
    plt.axvline(age_stats.loc['Disease (1)', 'Mean Age'], color='red', linestyle='--', label='Disease Mean')

    plt.title("Age Distribution by Target (Diagnosis)")
    plt.xlabel("Age (Years)")
    plt.legend(["Healthy", "Disease"], loc='upper left')

    # MARIMO UI
    mo.vstack([
        mo.md("## 🎂 Age Distribution"),
        mo.hstack([
            age_stats,
            plt.gca()
        ], justify="start", gap=2)
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
def _(heart_df, mo, pd):
    # PREPROCESSING: ONE-HOT ENCODING
    # Copy original data
    df_processed = heart_df.copy()
    essential_cats = ['cp', 'thal', 'slope']

    df_processed = pd.get_dummies(heart_df, columns=essential_cats, drop_first=True)

    cols_old = heart_df.shape[1]
    cols_new = df_processed.shape[1]

    # MARIMO UI
    mo.vstack([
        mo.md(f"## 🛠️ Data Preprocessing: Encoding Applied"),
        mo.ui.table(df_processed.head())
    ])
    return (df_processed,)


@app.cell
def _(df_processed, mo, train_test_split):
    # Split data into features (X) and target (y)
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']

    # Split into Train/Test sets (80/20) with stratification to keep class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Output the size of resulting sets
    mo.md(f"Training samples - {X_train.shape[0]}, Test samples - {X_test.shape[0]}")
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    Pipeline,
    RepeatedStratifiedKFold,
    StandardScaler,
    X_train,
    y_train,
):
    # Modelling: Logistic Regression & Grid Search

    # Create Pipeline: Combines StandardScaler and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    # hyperparameter grid
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'lbfgs'] 
    }

    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # GridSearchCV focusing on 'recall' metric
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='recall', 
        n_jobs=-1
    )

    # Fitting model
    grid_search.fit(X_train, y_train)
    return cv_strategy, grid_search


@app.cell
def _(grid_search, mo):
    # Extract the best model, parameters, and score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mo.vstack([
        mo.md(f"### 🏆 Best Model Parameters"),
        mo.md(f"**C:** `{best_params['model__C']}` | **Solver:** `{best_params['model__solver']}`"),
        mo.md(f"**Mean Recall (CV):** `{best_score:.2%}`")])
    return (best_model,)


@app.cell
def _(X_train, best_model, mo, pd):
    # Features Importance
    coefs = best_model.named_steps['model'].coef_[0]
    features = X_train.columns

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': coefs
    }).sort_values(by='Importance', ascending=False)

    mo.vstack([
        mo.md("### 🔍 Feature Importance"),
        mo.ui.table(importance_df)
    ])
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    classification_report,
    confusion_matrix,
    grid_search,
    mo,
    pd,
    y_test,
):
    # predict on test data (LogisticRegression)
    y_pred = grid_search.predict(X_test)

    test_report = classification_report(y_test, y_pred, output_dict=True)
    test_report_df = pd.DataFrame(test_report).transpose()

    # Conf Matrix
    cm = confusion_matrix(y_test, y_pred)

    mo.vstack([
        mo.md("## Final Evaluation on Test Data"),
        mo.md(f"**Final Recall (Test):** `{test_report['1']['recall']:.2%}`"),
        mo.md("### Metrics"),
        mo.ui.table(test_report_df),    
        mo.md("### Confusion Matrix"),
        mo.as_html(ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Sick"]).plot().figure_)
    ])
    return (y_pred,)


@app.cell
def _(X_test, grid_search, mo, plt, roc_auc_score, roc_curve, y_test):
    y_scores = grid_search.predict_proba(X_test)[:, 1]

    # Calc ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc_value = roc_auc_score(y_test, y_scores)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_value:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    mo.vstack([
        mo.md(f"AUC-ROC Score: `{auc_value:.4f}`"),
        mo.as_html(fig)
    ])
    return


@app.cell
def _(X_test, mo, y_pred, y_test):
    # Error Analysis
    comp_df = X_test.copy()
    comp_df['Actual'] = y_test
    comp_df['Predicted'] = y_pred

    # Filter out rows where the prediction does not match the actual target
    errors = comp_df[comp_df['Actual'] != comp_df['Predicted']]

    mo.vstack([
        mo.md(f"## Error Analysis: Where did the model fail?"),

        # Display the number of errors vs total test samples
        mo.md(f"The model made **{len(errors)}** errors out of {len(y_test)} samples."),
        mo.ui.table(errors),

        mo.md("Look for patterns in these patients. For example, do they all have low `thalach` or high `age`? This shows what kind of data the model struggles with."),
    ])
    return


@app.cell
def _(
    GridSearchCV,
    KNeighborsClassifier,
    Pipeline,
    RandomForestClassifier,
    StandardScaler,
    X_train,
    cv_strategy,
    mo,
    pd,
    y_train,
):
    # Try other models
    other_models = {
        "Random_Forest": {
            "instance": RandomForestClassifier(random_state=42),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None]
            }
        },
        "k-NN": {
            "instance": KNeighborsClassifier(),
            "grid": {
                "model__n_neighbors": [5, 7, 11],
                "model__weights": ["uniform", "distance"]
            }
        }
    }

    results = []

    for name, config in other_models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", config["instance"])
        ])

        search = GridSearchCV(
        estimator=pipe,
        param_grid=config["grid"],
        cv=cv_strategy, 
        scoring="recall",
        n_jobs=-1
        )

        search.fit(X_train, y_train)

        results.append({
            "model": name,
            "best_Recall": f"{search.best_score_:.2%}",
            "parameters": search.best_params_
        })

    final_comparison = pd.DataFrame(results)
    mo.ui.table(final_comparison)
    return


@app.cell
def _(mo):
    mo.md(f"""
    # 🏆 Project Verdict
    After rigorous testing, **Logistic Regression** emerged as the champion model for this project!
    * Despite experimenting with more complex algorithms like Random Forest and k-NN, Logistic Regression achieved the highest **Recall (88.94%)**.
    * This proves that for small medical datasets, simpler linear models often provide better stability and generalization, effectively avoiding overfitting.
    """)
    return


if __name__ == "__main__":
    app.run()
