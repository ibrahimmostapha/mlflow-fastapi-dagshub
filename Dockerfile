FROM python:3.11-slim

WORKDIR /app

# Default to your public DagsHub MLflow tracking URI
ARG MLFLOW_TRACKING_URI=https://dagshub.com/ibrahimmostapha/mlflow-fastapi-dagshub.mlflow
ARG MLFLOW_MODEL_NAME=RandomForestDigits

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_MODEL_NAME=${MLFLOW_MODEL_NAME}
ENV MODEL_DIR=/app/model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fetch_model.py main.py ./

# Fetch the latest trained model from DagsHub via MLflow during image build
RUN python fetch_model.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]