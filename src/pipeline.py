"""Data pipeline — Pipeline architectural pattern."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StudentDataPipeline:
    FEATURES = [
        "study_hours",
        "attendance_percent",
        "assignments_done",
        "previous_score",
    ]

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.scaler = StandardScaler()

    def extract(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop_duplicates(subset=["student_id"]).copy()
        out[self.FEATURES] = out[self.FEATURES].fillna(out[self.FEATURES].median())
        out["passed"] = out["passed"].astype(int)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.FEATURES] = self.scaler.fit_transform(out[self.FEATURES])
        return out

    def load(self, test_size: float = 0.2, random_state: int = 42):
        prepared = self.transform(self.clean(self.extract()))
        x = prepared[self.FEATURES]
        y = prepared["passed"]
        return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    @staticmethod
    def generate_sample(path: str | Path, n: int = 800) -> Path:
        rng = np.random.default_rng(42)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        study = rng.integers(1, 40, n)
        attendance = rng.integers(40, 100, n)
        assignments = rng.integers(0, 11, n)
        prev_score = rng.integers(20, 100, n)

        # Logistic link — labels align with Logistic Regression (learnable, realistic noise)
        logit = (
            -5.9
            + 0.085 * study
            + 0.038 * attendance
            + 0.22 * assignments
            + 0.048 * prev_score
        )
        pass_prob = 1 / (1 + np.exp(-logit))
        passed = (rng.random(n) < pass_prob).astype(int)

        pd.DataFrame(
            {
                "student_id": [f"S{i:04d}" for i in range(n)],
                "study_hours": study,
                "attendance_percent": attendance,
                "assignments_done": assignments,
                "previous_score": prev_score,
                "passed": passed,
            }
        ).to_csv(path, index=False)
        return path
