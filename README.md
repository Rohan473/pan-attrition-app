# Palo Alto Networks — Employee Attrition Intelligence

A machine learning–based employee attrition prediction and risk scoring system built with Streamlit. The app enables HR teams to identify high-risk employees, understand key drivers of attrition, and take targeted retention action.

## Overview

This project trains and compares multiple classification models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) on Palo Alto Networks HR data, applies SMOTE to address class imbalance, and surfaces results through an interactive dashboard with SHAP-based explainability.

## Project Structure

```
P2/
├── README.md
├── PRD.md                         # Product requirements document
├── executive_summary.md           # Executive summary
├── research_paper.tex             # LaTeX research paper
├── pan_attrition_app/
│   ├── app.py                     # Main Streamlit application
│   ├── Palo_Alto_Networks.csv     # HR dataset
│   └── requirements.txt           # Python dependencies
└── main/
    ├── app.py                     # Alternate entry point
    ├── Palo_Alto_Networks.csv
    └── requirements.txt
```

## Features

- **Model Comparison** — Train and evaluate Logistic Regression, Random Forest, Gradient Boosting, and XGBoost side by side
- **SMOTE Oversampling** — Handles class imbalance in the attrition label
- **Risk Scoring** — Per-employee attrition probability scores
- **SHAP Explainability** — Feature importance and individual prediction explanations
- **Interactive Dashboard** — Filterable visualizations built with Plotly and Streamlit

## Streamlit Dashboard
-https://pan-attrition-app-oadspf3hy6uek2dksvy4dm.streamlit.app/

## Dataset

`Palo_Alto_Networks.csv` — HR records including demographic, compensation, satisfaction, and performance fields. The target variable is `Attrition` (0 = stayed, 1 = left).

Key features: `Age`, `Department`, `JobRole`, `MonthlyIncome`, `OverTime`, `YearsAtCompany`, `JobSatisfaction`, `EnvironmentSatisfaction`, and more.

## Dependencies

| Package | Version |
|---|---|
| streamlit | 1.55.0 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| shap | 0.51.0 |
| plotly | 6.6.0 |
| imbalanced-learn | 0.14.1 |
| pandas | 2.3.3 |
| numpy | 2.4.2 |
