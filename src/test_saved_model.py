
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    df = pd.read_csv(DATA_PATH)

    # Use one existing record only to verify inference.
    X = df.drop(
        columns=["id", "dataset", "num", "heart_disease"]
    )
    sample = X.iloc[[0]]

    model_paths = [
        MODEL_DIR / "logistic_regression_final_pipeline.joblib",
        MODEL_DIR / "random_forest_final_pipeline.joblib",
    ]

    print("Testing saved model pipelines...")

    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model = joblib.load(model_path)

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0, 1]

        print(f"\nModel: {model_path.name}")
        print(f"Prediction: {int(prediction)}")
        print(f"Estimated model score for class 1: {probability:.4f}")

    print("\nSaved model loading and inference checks passed.")
    print("This verifies inference only, not clinical accuracy.")


if __name__ == "__main__":
    main()
