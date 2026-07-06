"""
API Gateway — Microservices pattern.
Run: uvicorn src.api:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import PassPredictor

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "pass_model.joblib"

app = FastAPI(title="StudySmart API", version="1.0")
predictor: PassPredictor | None = None


class StudentInput(BaseModel):
    study_hours: int = Field(ge=0, le=60)
    attendance_percent: int = Field(ge=0, le=100)
    assignments_done: int = Field(ge=0, le=10)
    previous_score: int = Field(ge=0, le=100)


@app.on_event("startup")
def load_model():
    global predictor
    if not MODEL.exists():
        raise RuntimeError("Model missing. Run: python setup.py")
    predictor = PassPredictor(MODEL)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return predictor.metrics if predictor else {}


@app.post("/predict")
def predict(student: StudentInput):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    result = predictor.predict(**student.model_dump())
    return {"input": student.model_dump(), "result": result}
