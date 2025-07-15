# log_anomaly_ui/app/config.py

import os

from pathlib import Path

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "trainInput"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "trainOutput"
