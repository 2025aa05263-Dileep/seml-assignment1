"""ML inference — Microservices pattern (prediction service)."""

from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import StudentDataPipeline


class PassPredictor:
    def __init__(self, model_path: str | Path):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.metrics = artifact.get("metrics", {})

    def predict(self, study_hours: int, attendance_percent: int,
                assignments_done: int, previous_score: int) -> dict:
        row = pd.DataFrame(
            [[study_hours, attendance_percent, assignments_done, previous_score]],
            columns=StudentDataPipeline.FEATURES,
        )
        scaled = self.scaler.transform(row)
        prob = float(self.model.predict_proba(scaled)[0, 1])
        passed = int(prob >= 0.5)
        return {
            "passed": passed,
            "pass_probability": round(prob, 4),
            "risk": "At Risk" if prob < 0.5 else "Likely Pass",
        }
