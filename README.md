# 🚛 Fleet Partnership Prediction

A Machine Learning project for predicting the likelihood of prospective companies becoming successful fleet management partners.

The project uses **XGBoost**, **Stratified K-Fold Cross Validation**, a custom preprocessing pipeline, and an ensemble of multiple models to improve prediction performance.

---

## 📌 Project Overview

A fleet management company evaluates prospective companies for potential partnerships.

The goal of this project is to build a Machine Learning model that predicts the probability of a successful partnership based on anonymized company characteristics.

The dataset contains approximately:

- **38,400 training samples**
- **100 anonymized numerical features**
- A binary target variable
- Separate testing data for generating predictions

The evaluation metric for the competition is **ROC-AUC Score**.

---

## 🎯 Objective

Build a classification model that can accurately predict the probability of a prospective company becoming a successful partner.

The final model predictions are evaluated using:

**ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)**

---

## ⚙️ Project Workflow

The complete Machine Learning workflow:

```text
Load Data
    ↓
Handle Missing Values
    ↓
Handle Outliers
    ↓
Stratified K-Fold Cross Validation
    ↓
Train XGBoost Models
    ↓
Evaluate using ROC-AUC
    ↓
Ensemble Predictions
    ↓
Generate Submission File

---

## 🧹 Data Preprocessing

### Missing Values

Missing values are handled using:

`SimpleImputer(strategy="median")`

The median was chosen because it is more robust to extreme values compared to the mean.

### Outlier Handling

A custom Scikit-learn transformer called `OutlierClipper` was implemented.

Outliers are detected using the **Interquartile Range (IQR)** method.

Instead of removing outliers, extreme values are clipped using the calculated lower and upper bounds.

---

## 🔄 Cross Validation

The project uses **Stratified K-Fold Cross Validation**.

For each fold:

1. Split the data into training and validation sets.
2. Fit the preprocessing pipeline on the training fold.
3. Transform the validation fold using the same fitted preprocessor.
4. Train an XGBoost model.
5. Generate probability predictions.
6. Calculate the ROC-AUC score.

This approach helps prevent **Data Leakage** during preprocessing.

---

## 🤖 Model

The main model used in this project is **XGBoost Classifier**.

Final configuration:

```text
max_depth = 3
learning_rate = 0.1
n_estimators = 1200
subsample = 0.8
colsample_bytree = 0.8


🧠 Ensemble Learning  
A separate XGBoost model was trained for each Cross Validation fold.

Each model generated probability predictions for the test dataset.

The final prediction was calculated by averaging the predictions from all models.

This ensemble approach helps produce more stable predictions.

📊 Results
Cross Validation Performance

Mean ROC-AUC: ~0.929

Kaggle Leaderboard Score

🏆 ROC-AUC Score: 0.9384