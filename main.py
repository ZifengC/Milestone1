import joblib
import numpy as np
from fastapi import FastAPI
from schema import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0"
)

# Load model once at startup (Cloud Run / Container lifecycle)
model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features).max()

    return PredictionResponse(
        prediction=int(pred),
        confidence=float(prob),
        model_version="1.0.0"
    )
