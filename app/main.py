from fastapi import FastAPI, HTTPException
from app.schemas import BatchInput, PredictionResponse
from app.predictor import BatchPredictor
from app.config import API_VERSION

app = FastAPI(
    title="AI Batch Intelligence API",
    version=API_VERSION,
    description="Real-time multi-target manufacturing forecasting system",
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": API_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict_batch(batch: BatchInput):
    try:
        result = BatchPredictor.predict(batch.dict())
        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
