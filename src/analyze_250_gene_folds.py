
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "candidate"
    / "out_of_fold_250_genes"
    / "out_of_fold_predictions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "reports"
    / "candidate"
    / "out_of_fold_250_genes"
    / "fold_metrics.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "actual_subtype",
        "predicted_subtype",
        "fold",
        "patient_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    results = []

    for fold, fold_df in df.groupby("fold", sort=True):
        actual = fold_df["actual_subtype"].astype(str)
        predicted = fold_df["predicted_subtype"].astype(str)

        results.append({
            "fold": int(fold),
            "test_samples": len(fold_df),
            "unique_patients": fold_df["patient_id"].nunique(),
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "macro_f1": f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            ),
        })

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n250-GENE MODEL: FOLD-BY-FOLD PERFORMANCE")
    print("=" * 65)
    print(results_df.round(4).to_string(index=False))

    print("\nMean and standard deviation across folds:")
    print(
        results_df[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")
    print(
        "\nNote: Fold scores measure variation across these "
        "cross-validation splits; they are not independent "
        "external validation."
    )


if __name__ == "__main__":
    main()
