import joblib
from app.config import (
    QUALITY_MODEL_PATH,
    ENERGY_MODEL_PATH,
    SCALER_PATH,
    PLS_PATH
)


class ModelLoader:

    def __init__(self):
        self.quality_model = joblib.load(QUALITY_MODEL_PATH)
        self.energy_model = joblib.load(ENERGY_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.pls = joblib.load(PLS_PATH)


model_loader = ModelLoader()