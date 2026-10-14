
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["id", "dataset", "num", "heart_disease"]
    )
    y = df["heart_disease"].astype(int)

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Expected a complete binary target: 0 and 1.")

    models = {
        "logistic_regression": LogisticRegression(
            C=0.1,
            class_weight=None,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X)),
            ("model", estimator),
        ])

        pipeline.fit(X, y)

        output_path = MODEL_DIR / f"{model_name}_final_pipeline.joblib"
        joblib.dump(pipeline, output_path)

        metadata.append({
            "model": model_name,
            "training_records": len(X),
            "features": X.shape[1],
            "target": "heart_disease",
            "target_definition": "1 if original num > 0, otherwise 0",
            "training_note": (
                "Fitted on all available records for deployment; "
                "not used to calculate holdout evaluation metrics."
            ),
            "model_path": str(output_path.relative_to(PROJECT_ROOT)),
        })

        print(f"\nModel: {model_name}")
        print(f"Training records: {len(X)}")
        print(f"Input features: {X.shape[1]}")
        print(f"Saved pipeline: {output_path}")

    metadata_path = REPORT_DIR / "final_model_metadata.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)

    print(f"\nSaved model metadata: {metadata_path}")
    print("\nFinal model training completed.")


if __name__ == "__main__":
    main()
