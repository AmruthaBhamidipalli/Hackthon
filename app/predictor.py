import pandas as pd
import time
from app.model_loader import model_loader

MODEL_VERSION = "1.0.0"


class BatchPredictor:

    @staticmethod
    def predict(data: dict):

        start_time = time.time()

        input_df = pd.DataFrame([data])
        scaled = model_loader.scaler.transform(input_df)
        preds = model_loader.model.predict(scaled)

        inference_time = (time.time() - start_time) * 1000

        return {
            "predictions": {
                "Tablet_Weight": float(preds[0][0]),
                "Hardness": float(preds[0][1]),
                "Friability": float(preds[0][2]),
                "Disintegration_Time": float(preds[0][3]),
                "Dissolution_Rate": float(preds[0][4]),
                "Content_Uniformity": float(preds[0][5]),
            },
            "inference_time_ms": round(inference_time, 2),
            "model_version": MODEL_VERSION,
        }
