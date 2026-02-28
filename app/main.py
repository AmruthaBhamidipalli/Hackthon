from fastapi import FastAPI, HTTPException
from app.schemas import BatchInput, PredictionResponse
from app.predictor import BatchPredictor
from app.config import API_VERSION


app = FastAPI(
    title="AI Batch Intelligence API",
    version=API_VERSION,
    description="Real-time multi-target manufacturing forecasting system"
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": API_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict_batch(batch: BatchInput):
    try:
        predictions, intervals, inference_time = (
            BatchPredictor.predict(batch.dict())
        )

        return PredictionResponse(
            predictions=predictions,
            prediction_intervals=intervals,
            inference_time_ms=round(inference_time, 2),
            model_version=API_VERSION
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))