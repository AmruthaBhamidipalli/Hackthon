import logging
import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Logging Configuration
# ==============================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"

logger = logging.getLogger("AI-Batch-Intelligence")
logger.setLevel(LOG_LEVEL)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if ENABLE_FILE_LOGGING:
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# ==============================
# Timing Decorator
# ==============================

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000

        if os.getenv("ENABLE_TIMING_LOGS", "false").lower() == "true":
            logger.info(f"{func.__name__} executed in {round(duration, 2)} ms")

        return result
    return wrapper


# ==============================
# Drift Detection
# ==============================

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.15))

def detect_drift(reference_data, new_data):
    """
    Simple Population Stability Index (PSI)-like drift check
    """

    reference_mean = np.mean(reference_data, axis=0)
    new_mean = np.mean(new_data, axis=0)

    drift_score = np.abs(reference_mean - new_mean) / (np.abs(reference_mean) + 1e-8)

    drift_flag = np.any(drift_score > DRIFT_THRESHOLD)

    logger.info(f"Drift score: {drift_score}")
    logger.info(f"Drift detected: {drift_flag}")

    return drift_flag, drift_score.tolist()


# ==============================
# Input Feature Validation
# ==============================

def validate_numeric_range(value, min_value=None, max_value=None):
    if min_value is not None and value < min_value:
        raise ValueError(f"Value {value} is below allowed minimum {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"Value {value} exceeds allowed maximum {max_value}")
    return True


# ==============================
# Safe Error Formatter
# ==============================

def format_exception(e: Exception):
    logger.error(str(e))
    return {
        "error_type": type(e).__name__,
        "message": str(e)
    }