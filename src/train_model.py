
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from preprocessing import create_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# 2. Load data and separate features and target
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    target = "heart_disease"

    # Exclude identifiers, source information, and original target.
    excluded_columns = [
        "id",
        "dataset",
        "num",
        target,
    ]

    X = df.drop(columns=excluded_columns, errors="ignore")
    y = df[target]

    if y.isna().any():
        raise ValueError("The target contains missing values.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target values 0 and 1.")

    print("Dataset loaded successfully.")
    print(f"Total records: {len(df)}")
    print(f"Input features: {X.shape[1]}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y


# --------------------------------------------------
# 3. Create preprocessing pipeline
# --------------------------------------------------


def create_preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import MissingIndicator

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

    numeric_missing = MissingIndicator(
        features="all",
        error_on_new=False,
    )

    categorical_missing = MissingIndicator(
        features="all",
        error_on_new=False,
    )

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("numeric_missing", numeric_missing, numeric_features),
        ("categorical_missing", categorical_missing, categorical_features),
    ])

    return preprocessor


# --------------------------------------------------
# 4. Train and evaluate models
# --------------------------------------------------

def evaluate_model(name, pipeline, X_test, y_test):
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

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No disease", "Disease present"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    return metrics


def main():
    # Load features and target.
    X, y = load_data()

    # Split before fitting any preprocessing transformations.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTRAIN-TEST SPLIT")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Training positive rate: {y_train.mean():.2%}")
    print(f"Testing positive rate: {y_test.mean():.2%}")

    # Create separate pipelines so each model gets its own
    # independently fitted preprocessing transformations.
    logistic_pipeline = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    forest_pipeline = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
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
    ])

    models = {
        "Logistic Regression": logistic_pipeline,
        "Random Forest": forest_pipeline,
    }

    all_metrics = []

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        all_metrics.append(metrics)

    # Save model comparison results.
    results = pd.DataFrame(all_metrics)
    results = results.sort_values("ROC_AUC", ascending=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results.round(4).to_string(index=False))

    # Save both fitted pipelines for later use in Streamlit.
    joblib.dump(
        logistic_pipeline,
        MODEL_DIR / "logistic_regression_pipeline.joblib",
    )

    joblib.dump(
        forest_pipeline,
        MODEL_DIR / "random_forest_pipeline.joblib",
    )

    print("\nTraining completed.")
    print("Comparison saved to reports/model_comparison.csv")
    print("Fitted pipelines saved to models/")


if __name__ == "__main__":
    main()
