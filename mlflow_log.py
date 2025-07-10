import mlflow
import mlflow.sklearn
import joblib
import os

# --- Step 1: Set MLflow config ---
mlflow.set_tracking_uri("http://localhost:5001")  # adjust if different
mlflow.set_experiment("vehicle_model_tracking")

# --- Step 2: Load Trained Models ---
model_busy = joblib.load("models/busy_model.pkl")
model_free = joblib.load("models/free_model.pkl")
model_battery = joblib.load("models/battery_model.pkl")

# --- Step 3: Log models under one run ---
with mlflow.start_run(run_name="vehicle_model_run"):

    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)

    mlflow.sklearn.log_model(model_busy, artifact_path="busy_model", registered_model_name="BusyModel")
    mlflow.sklearn.log_model(model_free, artifact_path="free_model", registered_model_name="FreeModel")
    mlflow.sklearn.log_model(model_battery, artifact_path="battery_model", registered_model_name="BatteryModel")

    print("✅ Models logged and registered to MLflow")


# import dagshub
# dagshub.init(repo_owner='diya91410', repo_name='mlflow_demo', mlflow=True)

# import mlflow
# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)