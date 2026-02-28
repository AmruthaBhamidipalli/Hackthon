from pydantic import BaseModel, Field
from typing import Dict

class BatchInput(BaseModel):
    Drying_Temp: float = Field(..., gt=0)
    Drying_Time: float = Field(..., gt=0)
    Compression_Force: float
    Machine_Speed: float
    Binder_Amount: float
    Flow_Rate_LPM: float
    Motor_Speed_RPM: float
    Vibration_mm_s: float

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    prediction_intervals: Dict[str, Dict[str, float]]
    inference_time_ms: float
    model_version: str