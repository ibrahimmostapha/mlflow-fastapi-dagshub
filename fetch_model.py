import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "RandomForestDigits")
OUTPUT_DIR = os.getenv("MODEL_DIR", "model")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'")

latest = max(versions, key=lambda v: int(v.version))
model_uri = f"models:/{MODEL_NAME}/{latest.version}"

print(f"Downloading latest model: {model_uri} -> {OUTPUT_DIR}")
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=OUTPUT_DIR)
print("Download complete.")