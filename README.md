# Heart Disease Prediction Using Machine Learning

## Project Overview

This project implements and compares five machine learning classification algorithms to predict the presence of heart disease. It includes Exploratory Data Analysis (EDA), data preprocessing, model training, performance evaluation, and an interactive web application developed using Streamlit.

## Dataset

**File:** `heart.csv`

The dataset contains 1,025 records, 13 input features, and one binary target variable.

- **Target = 0:** No heart disease
- **Target = 1:** Heart disease
- **Missing values:** 0
- **Duplicate records:** 205
- **Unique records:** 820

The dataset includes features such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, and other clinical measurements.

## Machine Learning Models

The following classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The data was divided into training and testing sets using a group-aware split to reduce data leakage from duplicate feature records. Preprocessing was performed using scikit-learn pipelines.

## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8627 | 0.9434 | 0.8173 | 0.9043 | 0.8586 | 0.7294 |
| Decision Tree | **0.9755** | **0.9750** | **0.9785** | **0.9681** | **0.9733** | **0.9507** |
| KNN | 0.8971 | 0.9515 | 0.9011 | 0.8723 | 0.8865 | 0.7927 |
| Naive Bayes | 0.8431 | 0.9435 | 0.7981 | 0.8830 | 0.8384 | 0.6901 |
| SVM | 0.9265 | 0.9732 | 0.9072 | 0.9362 | 0.9215 | 0.8527 |

### Best Performing Model

The **Decision Tree classifier** achieved the highest overall performance, with:

- Accuracy: **97.55%**
- AUC: **0.9750**
- F1-score: **0.9733**
- MCC: **0.9507**

These results were obtained on a held-out test set of 204 records.

## Streamlit Application

An interactive Streamlit application (`app.py`) was developed to evaluate the trained models.

**Application features:**

- Upload a CSV dataset.
- Select any of the five trained classification models.
- Generate predictions on uploaded data.
- Display Accuracy, AUC, Precision, Recall, F1-score, and MCC when true labels are available.
- View the confusion matrix and classification report.
- Download model predictions as a CSV file.
- Switch between models without uploading the dataset again during the same session.

## Project Structure

```text
Heart_Disease_ML_Assignment/
├── app.py
├── heart.csv
├── test_data.csv
├── requirements.txt
├── README.md
├── notebooks/
│   └── Heart_Disease_EDA.ipynb
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## Installation and Execution

### 1. Clone the Repository

### 2. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### 4. Test the Application

1. Open the Streamlit application in your browser.
2. Upload `test_data.csv`.
3. Select a classification model.
4. View the predictions and evaluation metrics.
5. Change the model to compare results.


## Project Links
**Live Streamlit Application:** https://ga7v3aub5gxcxdr5mxfsxz.streamlit.app/

## Conclusion

This project demonstrates an end-to-end machine learning classification workflow for heart disease prediction. Five models were trained and evaluated using six performance metrics.

The Decision Tree classifier achieved the best results on the selected test set. The Streamlit application makes it possible to interactively test and compare all five models.

**Disclaimer:** This project is intended for academic and educational purposes only. Its predictions should not be used as medical diagnoses.
