import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "models")

QUALITY_MODEL_PATH = f"{MODEL_DIR}/quality_model.pkl"
ENERGY_MODEL_PATH = f"{MODEL_DIR}/energy_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
PLS_PATH = f"{MODEL_DIR}/pls_transformer.pkl"

API_VERSION = "v1.0.0"