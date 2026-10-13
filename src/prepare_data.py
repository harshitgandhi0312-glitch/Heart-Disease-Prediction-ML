
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "id", "dataset", "chol", "trestbps", "num"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    # Preserve the original target and source columns.
    df = df.copy()

    print("Original dataset shape:", df.shape)

    # Investigate and correct known placeholder values.
    for column in ["chol", "trestbps"]:
        zero_count = int(df[column].eq(0).sum())
        print(f"{column}: converting {zero_count} zero values to NaN")
        df.loc[df[column].eq(0), column] = float("nan")

    # Create binary target without overwriting the original num column.
    df["heart_disease"] = df["num"].gt(0).astype("int8")

    # Save the prepared working copy, not the original dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nPrepared dataset saved to:", OUTPUT_PATH)
    print("Prepared dataset shape:", df.shape)

    print("\nMissing values after preparation:")
    print(
        df.isna().sum()
        .loc[lambda counts: counts > 0]
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nBinary target distribution:")
    print(
        df["heart_disease"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No disease", 1: "Disease present"})
        .to_string()
    )

    print("\nSource distribution:")
    print(df["dataset"].value_counts().to_string())

    print("\nPreparation complete. No rows were deleted or imputed.")
    print("The original dataset remains unchanged.")


if __name__ == "__main__":
    main()
