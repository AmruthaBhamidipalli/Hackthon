import pandas as pd
from app.model_loader import model_loader


class BatchPredictor:

    @staticmethod
    def predict(data: dict):
        input_df = pd.DataFrame([data])

        scaled = model_loader.scaler.transform(input_df)
        preds = model_loader.model.predict(scaled)

        return {
            "Tablet_Weight": float(preds[0][0]),
            "Hardness": float(preds[0][1]),
            "Friability": float(preds[0][2]),
            "Disintegration_Time": float(preds[0][3]),
            "Dissolution_Rate": float(preds[0][4]),
            "Content_Uniformity": float(preds[0][5]),
        }
