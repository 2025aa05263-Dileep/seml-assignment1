"""Model training — Pipeline pattern (training stage)."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.pipeline import StudentDataPipeline


def train_model(data_path: str | Path, model_path: str | Path) -> dict:
    pipeline = StudentDataPipeline(data_path)
    x_train, x_test, y_train, y_test = pipeline.load()

    model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": pipeline.scaler, "metrics": metrics}, model_path)
    return metrics
