# Lab 7: MLOps Pipeline Documentation

## 1. Objective

To document the end-to-end machine learning pipeline for the Telco Customer Churn Prediction project.

## 2. Project Overview

This project predicts whether a telecommunications customer is likely to leave the service. A Random Forest classification model is used, and MLflow is used for experiment tracking and model registration.

## 3. Pipeline Stages

1. Data Loading
2. Data Validation
3. Data Preprocessing
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. MLflow Experiment Tracking
8. Model Registration

## 4. Lab 6 Model Results

| Metric | Result |
|---|---:|
| Accuracy | 0.7559 |
| Precision | 0.5276 |
| Recall | 0.7674 |
| F1-score | 0.6253 |
| ROC-AUC | 0.8433 |

## 5. Registered Model

- Model name: `Telco_Churn_RandomForest`
- Model version: 1
- Registry status: READY
- MLflow experiment: `Telco_Churn_Prediction`

## 6. Important Files

- `churn.csv` — original dataset
- `preprocess_pipeline.py` — preprocessing script
- `run_lab5_pipeline.py` — Lab 5 script
- `run_lab6_registry.py` — Lab 6 model registration script
- `data/processed/` — processed datasets
- `models/preprocessor.pkl` — saved preprocessing object
- `mlflow.db` — MLflow database

## 7. Conclusion

The project has completed preprocessing and successfully registered a Random Forest model using MLflow. This document records the current workflow and model results.