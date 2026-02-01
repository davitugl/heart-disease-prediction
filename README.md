## About Project
This project aims to predict the presence of **heart disease** using a clinical dataset. Given the medical nature of the problem, the primary goal was to maximize **Recall**, ensuring that as few sick patients as possible are missed (minimizing False Negatives). Through rigorous testing of multiple models, **Logistic Regression** was identified as the champion due to its stability on small datasets and linear interpretability.


## Dataset from Kaggle
**Link:** (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Install Dependencies
Inside the project directory, install all dependencies using Poetry:

```bash
poetry install
```
## Run the Marimo notebook:

```bash
poetry run marimo edit notebooks/heart_disease_classification.py
```


## Live Report
(https://davitugl.github.io/heart-disease-prediction/)


