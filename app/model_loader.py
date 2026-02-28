import joblib
import os

MODEL_PATH = os.path.join("models", "quality_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

class ModelLoader:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

model_loader = ModelLoader()