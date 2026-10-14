
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)

# Columns that must not be used as model inputs.
EXCLUDED_FEATURES = [
    "id",              # Record identifier
    "dataset",         # Source-specific information
    "num",             # Original target: would leak the answer
    "heart_disease",   # Actual prediction target
]


def load_features():
    """Load prepared data and separate predictors from the target."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=EXCLUDED_FEATURES + ["heart_disease"],
        errors="ignore",
    )
    y = df["heart_disease"].astype(int)

    if y.isna().any():
        raise ValueError("The target contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Expected a binary target containing 0 and 1.")

    return X, y


def create_preprocessor(X):
    """Build preprocessing steps without fitting on the full dataset."""
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                add_indicator=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def main():
    X, y = load_features()

    print("FEATURE ENGINEERING REVIEW")
    print("=" * 55)

    print(f"Records: {len(X)}")
    print(f"Input features before encoding: {X.shape[1]}")

    print("\nNumeric features:")
    print(X.select_dtypes(include="number").columns.tolist())

    print("\nCategorical features:")
    print(
        X.select_dtypes(
            exclude="number"
        ).columns.tolist()
    )

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print("\nMissing values to be handled by the pipeline:")
    print(
        X.isna().sum()
        .loc[lambda counts: counts > 0]
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nExcluded from predictors:")
    print(EXCLUDED_FEATURES)

    preprocessor = create_preprocessor(X)
    print("\nPreprocessing pipeline created successfully.")
    print("It has not been fitted and no data has been transformed.")


if __name__ == "__main__":
    main()
