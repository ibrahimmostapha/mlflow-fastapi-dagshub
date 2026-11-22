import os
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv("MODEL_DIR", "model")
model = mlflow.sklearn.load_model(MODEL_DIR)

app = FastAPI(title="Digits Classifier API")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=64, max_length=64)


@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.features])[0]
    return {"prediction": int(prediction)}