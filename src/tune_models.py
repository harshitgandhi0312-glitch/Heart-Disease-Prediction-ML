
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}"
        )

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

    print(f"Training records: {len(X_train)}")
    print(f"Untouched test records: {len(X_test)}")

    return X_train, y_train


def create_pipeline(model):
    return Pipeline([
        ("preprocessor", create_preprocessor(
            pd.read_csv(DATA_PATH).drop(
                columns=["id", "dataset", "num", "heart_disease"]
            )
        )),
        ("model", model),
    ])


def main():
    X_train, y_train = load_data()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    configurations = {
        "Logistic Regression": {
            "pipeline": create_pipeline(
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                )
            ),
            "parameters": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "pipeline": create_pipeline(
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=1,
                )
            ),
            "parameters": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7],
            },
        },
    }

    rows = []

    for name, config in configurations.items():
        print("\n" + "=" * 65)
        print(f"TUNING: {name}")
        print("=" * 65)

        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["parameters"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

        rows.append({
            "Model": name,
            "Best_CV_ROC_AUC": search.best_score_,
            "Best_parameters": str(search.best_params_),
        })

    results = pd.DataFrame(rows).sort_values(
        "Best_CV_ROC_AUC",
        ascending=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "tuning_results.csv"
    results.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("TUNING SUMMARY")
    print("=" * 65)
    print(results.to_string(index=False))
    print(f"\nSaved: {output_path}")
    print(
        "\nThe original test set was not used for hyperparameter selection."
    )


if __name__ == "__main__":
    main()
