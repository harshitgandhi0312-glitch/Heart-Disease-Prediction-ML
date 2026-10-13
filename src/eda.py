
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_prepared.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid", context="notebook")


def save_plot(filename):
    """Save the current chart and close the figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: reports/figures/{filename}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total records: {len(df)}")

    # 1. Target distribution
    print("\n1. TARGET DISTRIBUTION")
    print(df["heart_disease"].value_counts().sort_index())

    target_pct = (
        df["heart_disease"].value_counts(normalize=True).sort_index() * 100
    ).round(2)
    print("\nTarget percentages:")
    print(target_pct)

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df,
        x="heart_disease",
        hue="heart_disease",
        legend=False,
    )
    plt.title("Heart Disease Target Distribution")
    plt.xlabel("Heart Disease (0 = No, 1 = Present)")
    plt.ylabel("Number of Records")
    save_plot("01_target_distribution.png")

    # 2. Numerical feature distributions
    numeric_features = [
        col for col in [
            "age", "trestbps", "chol", "thalch", "oldpeak", "ca"
        ]
        if col in df.columns
    ]

    print("\n2. NUMERICAL FEATURE SUMMARY")
    print(df[numeric_features].describe().round(2).to_string())

    df[numeric_features].hist(
        figsize=(13, 9),
        bins=25,
        edgecolor="black",
    )
    plt.suptitle("Numerical Feature Distributions", y=1.02)
    save_plot("02_numeric_distributions.png")

    # 3. Compare numerical features by target
    print("\n3. NUMERICAL FEATURES BY TARGET")
    print(
        df.groupby("heart_disease")[numeric_features]
        .median()
        .round(2)
        .rename(index={0: "No disease", 1: "Disease present"})
        .to_string()
    )

    for feature in ["age", "trestbps", "chol", "thalch", "oldpeak"]:
        if feature not in df.columns:
            continue

        plt.figure(figsize=(7, 5))
        sns.boxplot(
            data=df,
            x="heart_disease",
            y=feature,
            hue="heart_disease",
            legend=False,
        )
        plt.title(f"{feature} by Heart Disease Target")
        plt.xlabel("Heart Disease (0 = No, 1 = Present)")
        plt.ylabel(feature)
        save_plot(f"03_{feature}_by_target.png")

    # 4. Categorical feature counts
    categorical_features = [
        col for col in [
            "sex", "cp", "fbs", "restecg",
            "exang", "slope", "thal"
        ]
        if col in df.columns
    ]

    print("\n4. CATEGORICAL FEATURE COUNTS")
    for feature in categorical_features:
        print(f"\n{feature}:")
        print(df[feature].value_counts(dropna=False).to_string())

        plt.figure(figsize=(9, 5))
        sns.countplot(
            data=df,
            x=feature,
            hue=feature,
            legend=False,
            order=df[feature].value_counts().index,
        )
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Number of Records")
        plt.xticks(rotation=30, ha="right")
        save_plot(f"04_{feature}_distribution.png")

    # 5. Disease rate by category
    print("\n5. DISEASE RATE BY CATEGORICAL FEATURE")

    for feature in ["sex", "cp", "exang", "slope"]:
        if feature not in df.columns:
            continue

        rates = (
            df.groupby(feature, dropna=False)["heart_disease"]
            .agg(["mean", "count"])
        )
        rates["disease_rate_pct"] = (rates["mean"] * 100).round(2)

        print(f"\n{feature}:")
        print(
            rates[["count", "disease_rate_pct"]].to_string()
        )

        plot_df = rates.reset_index()
        plot_df[feature] = plot_df[feature].fillna("Missing").astype(str)

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=plot_df,
            x=feature,
            y="disease_rate_pct",
            hue=feature,
            legend=False,
        )
        plt.title(f"Disease-Positive Rate by {feature}")
        plt.ylabel("Disease-Positive Rate (%)")
        plt.xlabel(feature)
        plt.xticks(rotation=30, ha="right")
        save_plot(f"05_disease_rate_by_{feature}.png")

    # 6. Missing value analysis
    print("\n6. MISSING VALUE ANALYSIS")

    missing_pct = (
        df.isnull().mean().mul(100)
        .sort_values(ascending=False)
    )
    missing_pct = missing_pct[missing_pct > 0].round(2)

    print(missing_pct.to_string())

    if not missing_pct.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=missing_pct.values,
            y=missing_pct.index,
            hue=missing_pct.index,
            legend=False,
        )
        plt.title("Missing Values by Feature")
        plt.xlabel("Missing Values (%)")
        plt.ylabel("Feature")
        save_plot("06_missing_values.png")

    # 7. Correlation analysis
    print("\n7. NUMERICAL CORRELATIONS")

    correlation_features = [
        col for col in [
            "age", "trestbps", "chol", "thalch",
            "oldpeak", "ca", "heart_disease"
        ]
        if col in df.columns
    ]

    correlation = df[correlation_features].corr()
    print(correlation.round(2).to_string())

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title("Numerical Feature Correlation Matrix")
    save_plot("07_correlation_heatmap.png")

    # 8. Source-specific target rates
    print("\n8. SOURCE DATASET COMPARISON")

    source_summary = (
        df.groupby("dataset")["heart_disease"]
        .agg(["count", "mean"])
    )
    source_summary["disease_rate_pct"] = (
        source_summary["mean"] * 100
    ).round(2)

    print(
        source_summary[["count", "disease_rate_pct"]].to_string()
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df,
        x="dataset",
        y="heart_disease",
        hue="dataset",
        estimator=lambda values: values.mean() * 100,
        errorbar=None,
        legend=False,
    )
    plt.title("Disease-Positive Rate by Source Dataset")
    plt.ylabel("Disease-Positive Rate (%)")
    plt.xlabel("Source Dataset")
    plt.xticks(rotation=20, ha="right")
    save_plot("08_disease_rate_by_source.png")

    print("\nEDA completed.")
    print(f"Charts are saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
