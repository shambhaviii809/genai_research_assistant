import time
import json
import logging
from functools import wraps
from uuid import uuid4
from logging.handlers import RotatingFileHandler

# ============================================================
# LOGGER SETUP (Production Ready)
# ============================================================

LOGGER_NAME = "rag_logger"

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if re-imported
if not logger.handlers:

    handler = RotatingFileHandler(
        "rag_metrics.log",
        maxBytes=5_000_000,   # 5MB per file
        backupCount=3        # Keep last 3 log files
    )

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


# ============================================================
# STRUCTURED METRICS LOGGER
# ============================================================

def log_metrics(metrics_dict):
    """
    Logs structured JSON telemetry for each RAG request.
    Automatically injects timestamp.
    """
    metrics_dict["timestamp"] = time.time()
    logger.info(json.dumps(metrics_dict))


# ============================================================
# TIMING DECORATOR
# ============================================================

def timed(func):
    """
    Decorator to measure execution time of any function.
    Returns: (original_result, duration_in_ms)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start) * 1000
        return result, duration_ms
    return wrapper


# ============================================================
# REQUEST ID GENERATOR
# ============================================================

def generate_request_id():
    """
    Generates a unique ID per RAG request.
    Useful for tracing and distributed logging.
    """
    return str(uuid4())