"""Generate data and train model."""

from pathlib import Path

from src.pipeline import StudentDataPipeline
from src.train import train_model

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "students.csv"
MODEL = ROOT / "models" / "pass_model.joblib"


def main():
    print("Creating sample student data...")
    StudentDataPipeline.generate_sample(DATA)
    print(f"  Saved: {DATA}")

    print("Training pass/fail model...")
    metrics = train_model(DATA, MODEL)
    print(f"  Saved: {MODEL}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
