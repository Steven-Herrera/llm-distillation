"""Secrets management module for loading and validating environment credentials.

This module validates that MLflow and Gmail credentials are present and raises
errors if any are missing.

Functions:
    load_mlflow_credentials: Validates that Mlflow credentials are present
"""

import os
from dotenv import load_dotenv

load_dotenv()


def load_mlflow_credentials() -> None:
    """Validates MLflow environment credentials.

    Raises:
        EnvironmentError: If any MLflow credentials are missing.
    """
    required_vars = [
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise EnvironmentError(f"Missing MLflow credentials: {', '.join(missing)}")
