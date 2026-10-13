
import os
import mlflow
import mlflow.sklearn
import numpy as np

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==================================================
# LAB 6: MLFLOW MODEL REGISTRY
# ==================================================

EXPERIMENT_NAME = "Telco_Churn_Prediction"
REGISTERED_MODEL_NAME = "Telco_Churn_RandomForest"

# 1. Configure MLflow tracking and registry
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

print("\n========== LAB 6: MODEL REGISTRY ==========")

# 2. Load processed data
X_train = np.load("data/processed/X_train_final.npy")
X_test = np.load("data/processed/X_test_final.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("[INFO] Data loaded successfully.")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 3. Define model parameters
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "class_weight": "balanced",
}

# 4. Train, evaluate, and track the model
with mlflow.start_run(
    run_name="Lab6_RandomForest_Registry"
) as run:

    mlflow.log_params(params)
    mlflow.log_param("model_family", "RandomForest")
    mlflow.log_param("lab", "Lab 6 - Model Registry")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    mlflow.log_metrics(metrics)

    # Save model in MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ],
    )

    run_id = run.info.run_id

# 5. Display evaluation results
print("\n========== MODEL EVALUATION ==========")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print(f"\n[INFO] MLflow Run ID: {run_id}")
print(f"[INFO] Model URI: {model_info.model_uri}")

# 6. Register model using the URI returned by MLflow
print("\n[INFO] Registering model...")

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=REGISTERED_MODEL_NAME,
)

print("\n[SUCCESS] Model registered successfully!")
print(f"Model name: {registered_model.name}")
print(f"Model version: {registered_model.version}")

# 7. Verify registration
client = MlflowClient()

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

print("\n========== REGISTERED MODEL VERSIONS ==========")

for version in versions:
    print(
        f"Version: {version.version} | "
        f"Status: {version.status} | "
        f"Run ID: {version.run_id}"
    )

# 8. Final result
print("\n========== LAB 6 COMPLETED ==========")
