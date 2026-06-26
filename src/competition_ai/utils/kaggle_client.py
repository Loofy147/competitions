import os
import subprocess
import kagglehub
from typing import Optional

class KaggleClient:
    """
    Utility wrapper for kagglehub and kaggle-cli.
    Provides methods for data/model retrieval and submission.
    """
    def __init__(self):
        # Ensure KAGGLE_CONFIG_DIR or credentials exist
        pass

    @staticmethod
    def download_dataset(dataset_handle: str, path: Optional[str] = None):
        """Downloads a dataset using kagglehub."""
        return kagglehub.dataset_download(dataset_handle, path=path)

    @staticmethod
    def download_model(model_handle: str):
        """Downloads a model using kagglehub."""
        return kagglehub.model_download(model_handle)

    @staticmethod
    def submit(competition: str, file_path: str, message: str):
        """Submits a file using kaggle-cli."""
        cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", file_path, "-m", message]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Submission failed: {result.stderr}")
        return result.stdout

    @staticmethod
    def list_submissions(competition: str):
        """Lists submissions using kaggle-cli."""
        cmd = ["kaggle", "competitions", "submissions", "-c", competition]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
