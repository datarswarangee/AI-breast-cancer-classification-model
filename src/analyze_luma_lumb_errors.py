
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = ROOT / "reports" / "candidate" / "out_of_fold"
DATA_DIR = ROOT / "data" / "processed_candidate"
OUTPUT_DIR = EVAL_DIR / "luma_lumb_analysis"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(
        EVAL_DIR / "out_of_fold_predictions.csv"
    )

    expression = pd.read_csv(
        DATA_DIR / "expression_matrix.csv",
        index_col=0,
    )

    labels = pd.read_csv(
        DATA_DIR / "subtype_labels.csv",
        index_col=0,
    )

    # Focus on actual LumA/LumB samples.
    relevant = predictions[
        predictions["actual_subtype"].isin(["LumA", "LumB"])
    ].copy()

    relevant["correct"] = (
        relevant["actual_subtype"]
        == relevant["predicted_subtype"]
    )

    # Summarize errors in the two directions.
    confusion = (
        relevant[
            relevant["actual_subtype"]
            != relevant["predicted_subtype"]
        ]
        .groupby(["actual_subtype", "predicted_subtype"])
        .size()
        .reset_index(name="count")
    )

    confusion.to_csv(
        OUTPUT_DIR / "luma_lumb_confusions.csv",
        index=False,
    )

    # Summarize observed correctness by actual subtype.
    performance = (
        relevant.groupby("actual_subtype")["correct"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "count": "samples",
            "sum": "correct_predictions",
            "mean": "accuracy",
        })
    )

    performance["error_rate"] = 1 - performance["accuracy"]

    performance.to_csv(
        OUTPUT_DIR / "luma_lumb_performance.csv",
        index=False,
    )

    # Compare expression values for LumA and LumB samples
    # using only genes from the existing candidate matrix.
    subtype_labels = labels["subtype"].astype(str)

    common_ids = expression.index.intersection(
        subtype_labels.index
    )

    expression = expression.loc[common_ids]
    subtype_labels = subtype_labels.loc[common_ids]

    lum_samples = subtype_labels.isin(["LumA", "LumB"])

    X = expression.loc[lum_samples]
    y = subtype_labels.loc[lum_samples]

    # Rank genes by absolute difference in group means,
    # measured in pooled standard-deviation units.
    lum_a = X.loc[y == "LumA"]
    lum_b = X.loc[y == "LumB"]

    mean_a = lum_a.mean(axis=0)
    mean_b = lum_b.mean(axis=0)

    var_a = lum_a.var(axis=0, ddof=1)
    var_b = lum_b.var(axis=0, ddof=1)

    pooled_sd = (
        (
            (len(lum_a) - 1) * var_a
            + (len(lum_b) - 1) * var_b
        )
        / (len(lum_a) + len(lum_b) - 2)
    ) ** 0.5

    effect_size = (mean_b - mean_a) / pooled_sd.replace(0, float("nan"))

    gene_summary = pd.DataFrame({
        "gene": X.columns,
        "mean_LumA": mean_a.values,
        "mean_LumB": mean_b.values,
        "standardized_difference_LumB_minus_LumA": effect_size.values,
    })

    gene_summary["absolute_difference"] = (
        gene_summary["standardized_difference_LumB_minus_LumA"]
        .abs()
    )

    gene_summary = gene_summary.sort_values(
        "absolute_difference",
        ascending=False,
    )

    gene_summary.head(50).to_csv(
        OUTPUT_DIR / "top_50_luma_lumb_expression_differences.csv",
        index=False,
    )

    print("=" * 65)
    print("MY MED AI - LUMA/LUMB ERROR INVESTIGATION")
    print("=" * 65)

    print("\nLumA/LumB prediction errors:")
    print(confusion.to_string(index=False))

    print("\nOut-of-fold performance:")
    print(performance.round(4).to_string(index=False))

    print("\nTop 10 genes by standardized expression difference:")
    print(
        gene_summary[
            [
                "gene",
                "mean_LumA",
                "mean_LumB",
                "standardized_difference_LumB_minus_LumA",
            ]
        ].head(10).round(4).to_string(index=False)
    )

    print(f"\nResults saved in: {OUTPUT_DIR}")
    print(
        "\nExploratory expression comparison only. "
        "The gene ranking is not a validated biomarker analysis."
    )


if __name__ == "__main__":
    main()
