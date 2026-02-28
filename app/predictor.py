import numpy as np
import pandas as pd
import time
from app.model_loader import model_loader

class BatchPredictor:

    @staticmethod
    def preprocess(data_dict):
        df = pd.DataFrame([data_dict])

        # Feature Engineering
        df["Drying_Energy_Index"] = df["Drying_Temp"] * df["Drying_Time"]
        df["Compression_Power"] = df["Compression_Force"] * df["Machine_Speed"]
        df["Binder_Amount_Squared"] = df["Binder_Amount"] ** 2

        scaled = model_loader.scaler.transform(df)
        transformed = model_loader.pls.transform(scaled)

        return transformed

    @staticmethod
    def predict(data_dict):
        start_time = time.time()

        processed = BatchPredictor.preprocess(data_dict)

        quality_preds = model_loader.quality_model.predict(processed)
        energy_pred = model_loader.energy_model.predict(processed)

        inference_time = (time.time() - start_time) * 1000

        results = {
            "Tablet_Weight": float(quality_preds[0][0]),
            "Hardness": float(quality_preds[0][1]),
            "Friability": float(quality_preds[0][2]),
            "Disintegration_Time": float(quality_preds[0][3]),
            "Dissolution_Rate": float(quality_preds[0][4]),
            "Content_Uniformity": float(quality_preds[0][5]),
            "Energy_Consumption": float(energy_pred[0])
        }

        intervals = BatchPredictor.compute_intervals(results)

        return results, intervals, inference_time

    @staticmethod
    def compute_intervals(predictions):
        intervals = {}
        for key, value in predictions.items():
            margin = 0.05 * value  # 5% interval placeholder
            intervals[key] = {
                "lower": float(value - margin),
                "upper": float(value + margin)
            }
        return intervals