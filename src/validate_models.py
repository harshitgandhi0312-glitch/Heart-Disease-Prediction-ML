
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from preprocessing import create_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["id", "dataset", "num", "heart_disease"]
    )
    y = df["heart_disease"].astype(int)

    return X, y, df["dataset"]



def create_preprocessor(X):
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        (
            "numeric_missing",
            MissingIndicator(
                features="all",
                error_on_new=False,
            ),
            numeric_features,
        ),
        (
            "categorical_missing",
            MissingIndicator(
                features="all",
                error_on_new=False,
            ),
            categorical_features,
        ),
    ])

    return preprocessor




def create_models(X):
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", create_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", create_preprocessor(X)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def run_cross_validation(X_train, y_train):
    print("\n" + "=" * 65)
    print("1. FIVE-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 65)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []

    for name, pipeline in create_models(X_train).items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {"Model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        rows.append(row)

    results = pd.DataFrame(rows)

    print(
        results.round(4).to_string(index=False)
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        REPORT_DIR / "cross_validation_results.csv",
        index=False,
    )

    print("\nSaved: reports/cross_validation_results.csv")

    return results


def run_source_holdout(X, y, sources):
    print("\n" + "=" * 65)
    print("2. SOURCE-HOLDOUT EVALUATION")
    print("=" * 65)

    print(
        "Each source is held out completely while the models train "
        "on the other three sources."
    )

    rows = []

    for source in sorted(sources.unique()):
        train_mask = sources != source
        test_mask = sources == source

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        print(
            f"\nHeld-out source: {source} "
            f"(train={len(X_train)}, test={len(X_test)})"
        )

        for name, pipeline in create_models(X_train).items():
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            row = {
                "Held_out_source": source,
                "Model": name,
                "Train_records": len(X_train),
                "Test_records": len(X_test),
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

            rows.append(row)

            print(
                f"{name}: "
                f"accuracy={row['Accuracy']:.3f}, "
                f"recall={row['Recall']:.3f}, "
                f"F1={row['F1_Score']:.3f}, "
                f"ROC-AUC={row['ROC_AUC']:.3f}"
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        REPORT_DIR / "source_holdout_results.csv",
        index=False,
    )

    print("\nSaved: reports/source_holdout_results.csv")

    return results


def main():
    X, y, sources = load_data()

    # Recreate the original baseline split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training-set cross-validation records:", len(X_train))
    print("Original holdout test records:", len(X_test))

    run_cross_validation(X_train, y_train)

    # Separate diagnostic experiment: hold out each complete source.
    run_source_holdout(X, y, sources)

    print("\nValidation experiments completed.")


if __name__ == "__main__":
    main()
