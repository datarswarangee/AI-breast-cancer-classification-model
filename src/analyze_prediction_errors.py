
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = ROOT / "reports" / "candidate" / "out_of_fold"
OUTPUT_DIR = EVAL_DIR / "error_analysis"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = EVAL_DIR / "out_of_fold_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing file: {predictions_path}\n"
            "Run src/evaluate_out_of_fold.py first."
        )

    df = pd.read_csv(predictions_path)

    required = {
        "sample_id",
        "patient_id",
        "actual_subtype",
        "predicted_subtype",
        "fold",
        "correct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Keep only incorrect predictions.
    errors = df.loc[~df["correct"].astype(bool)].copy()

    errors.to_csv(
        OUTPUT_DIR / "misclassified_samples.csv",
        index=False,
    )

    # Count each actual-to-predicted subtype confusion.
    confusion_counts = (
        errors.groupby(
            ["actual_subtype", "predicted_subtype"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    confusion_counts.to_csv(
        OUTPUT_DIR / "subtype_confusion_counts.csv",
        index=False,
    )

    # Summarize the number of errors for each actual subtype.
    subtype_errors = (
        errors.groupby("actual_subtype")
        .size()
        .rename("misclassified_samples")
        .reset_index()
    )

    subtype_errors.to_csv(
        OUTPUT_DIR / "errors_by_actual_subtype.csv",
        index=False,
    )

    # Plot the most common incorrect subtype predictions.
    top_errors = confusion_counts.head(10).copy()

    if not top_errors.empty:
        top_errors["confusion"] = (
            top_errors["actual_subtype"]
            + " → "
            + top_errors["predicted_subtype"]
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.barh(
            top_errors["confusion"].iloc[::-1],
            top_errors["count"].iloc[::-1],
        )

        ax.set_xlabel("Number of misclassified samples")
        ax.set_ylabel("Actual subtype → predicted subtype")
        ax.set_title("My Med AI: Most Common Classification Errors")

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "top_subtype_confusions.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("=" * 60)
    print("MY MED AI - PREDICTION ERROR ANALYSIS")
    print("=" * 60)

    print(f"Total samples: {len(df)}")
    print(f"Correct predictions: {int(df['correct'].astype(bool).sum())}")
    print(f"Misclassified samples: {len(errors)}")

    print("\nMost common subtype confusions:")
    print(confusion_counts.head(10).to_string(index=False))

    print("\nErrors by actual subtype:")
    print(subtype_errors.to_string(index=False))

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
