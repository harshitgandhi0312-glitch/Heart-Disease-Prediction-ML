
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["id", "dataset", "num", "heart_disease"]
    )
    y = df["heart_disease"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Tuned Logistic Regression": Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight=None,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "Tuned Random Forest": Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
    }

    rows = []

    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print("\nFinal holdout evaluation")

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        }
        rows.append(metrics)

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        for metric, value in metrics.items():
            if metric != "Model":
                print(f"{metric}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No disease", "Disease present"],
            zero_division=0,
        ))

        print("Confusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ))

    results = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "tuned_model_test_results.csv"
    results.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("TUNED MODEL TEST COMPARISON")
    print("=" * 60)
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
