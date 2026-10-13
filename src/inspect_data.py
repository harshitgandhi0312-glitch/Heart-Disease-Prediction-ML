
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please place heart_disease_uci.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())


    print("\n" + "=" * 60)
    print("2. COLUMN NAMES AND DATA TYPES")
    print("=" * 60)

    print(df.dtypes.to_string())

    print("\nDetailed information:")
    df.info()

   
    print("\n" + "=" * 60)
    print("3. MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = pd.DataFrame({
        "Missing_Count": df.isnull().sum(),
        "Missing_Percentage": (
            df.isnull().mean() * 100
        ).round(2),
    })

    missing = missing[missing["Missing_Count"] > 0]
    missing = missing.sort_values(
        by="Missing_Percentage", ascending=False
    )

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    print(f"\nTotal missing cells: {df.isnull().sum().sum()}")

   
    print("\n" + "=" * 60)
    print("4. DUPLICATE ANALYSIS")
    print("=" * 60)

    print(f"Duplicate rows: {df.duplicated().sum()}")

   
    print("\n" + "=" * 60)
    print("5. NUMERICAL SUMMARY")
    print("=" * 60)

    print(df.describe().round(2).to_string())

    
    print("\n" + "=" * 60)
    print("6. CATEGORICAL SUMMARY")
    print("=" * 60)

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe().to_string())
    else:
        print("No categorical columns detected.")

   
    print("\n" + "=" * 60)
    print("7. TARGET VARIABLE ANALYSIS")
    print("=" * 60)

    if "num" in df.columns:
        target = df["num"]

        print("Target column: num")
        print("\nTarget counts:")
        print(target.value_counts(dropna=False).sort_index())

        print("\nTarget percentages:")
        print(
            (target.value_counts(normalize=True, dropna=False)
             .sort_index() * 100).round(2)
        )
    else:
        print("Target column 'num' was not found.")

   
    print("\n" + "=" * 60)
    print("8. UNIQUE VALUES PER COLUMN")
    print("=" * 60)

    print(df.nunique(dropna=False).to_string())

    
    print("\n" + "=" * 60)
    print("9. INITIAL DATA-QUALITY CHECKS")
    print("=" * 60)

    if "id" in df.columns:
        print("ID column detected. We will investigate its role.")

    if "num" in df.columns:
        print(f"Missing target values: {df['num'].isnull().sum()}")

    for col in ["age", "trestbps", "chol", "thalch", "oldpeak"]:
        if col in df.columns:
            print(f"{col}: zero values = {(df[col] == 0).sum()}")

    print("\nInspection completed. No data has been modified.")


if __name__ == "__main__":
    main()
