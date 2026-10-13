
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 65)
    print("1. RECORDS BY SOURCE DATASET")
    print("=" * 65)

    print(df["dataset"].value_counts(dropna=False).to_string())

   
    columns_to_check = [
        "trestbps", "chol", "fbs", "restecg",
        "thalch", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    print("\n" + "=" * 65)
    print("2. MISSING VALUES BY SOURCE DATASET (%)")
    print("=" * 65)

    missing_by_source = (
        df.groupby("dataset")[columns_to_check]
        .agg(lambda col: round(col.isna().mean() * 100, 2))
    )

    print(missing_by_source.to_string())

   
    print("\n" + "=" * 65)
    print("3. ZERO VALUES BY SOURCE DATASET")
    print("=" * 65)

    for col in ["chol", "trestbps", "thalch"]:
        print(f"\n{col}:")
        zero_counts = (
            df.assign(is_zero=df[col].eq(0))
            .groupby("dataset")["is_zero"]
            .agg(["sum", "count"])
        )

        zero_counts["percentage"] = (
            zero_counts["sum"] / zero_counts["count"] * 100
        ).round(2)

        print(zero_counts.to_string())

   
    print("\n" + "=" * 65)
    print("4. ORIGINAL TARGET DISTRIBUTION BY SOURCE")
    print("=" * 65)

    target_counts = pd.crosstab(
        df["dataset"],
        df["num"],
        margins=True
    )

    print(target_counts.to_string())

   
    print("\n" + "=" * 65)
    print("5. BINARY TARGET PROPORTIONS BY SOURCE")
    print("=" * 65)

    binary_target = df["num"].gt(0).astype(int)

    target_summary = pd.DataFrame({
        "records": df.groupby("dataset").size(),
        "disease_present": binary_target.groupby(df["dataset"]).sum(),
        "disease_rate_pct": (
            binary_target.groupby(df["dataset"]).mean() * 100
        ).round(2),
    })

    print(target_summary.to_string())

    
    print("\n" + "=" * 65)
    print("6. EXAMPLES WITH ZERO CHOLESTEROL")
    print("=" * 65)

    zero_chol = df.loc[
        df["chol"].eq(0),
        ["id", "dataset", "age", "sex", "chol", "num"]
    ]

    print(f"Rows with zero cholesterol: {len(zero_chol)}")
    print(zero_chol.head(15).to_string(index=False))

    print("\nQuality investigation completed.")
    print("No records or values have been modified.")


if __name__ == "__main__":
    main()
