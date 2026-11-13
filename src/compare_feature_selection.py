
from pathlib import Path

import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_candidate"
OUTPUT_DIR = (
    ROOT / "reports" / "candidate" / "feature_selection_comparison"
)

RANDOM_STATE = 42
N_SPLITS = 5
FEATURE_COUNTS = [100, 250, 500, 1000]


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    X = pd.read_csv(
        DATA_DIR / "expression_matrix.csv",
        index_col=0,
    )

    labels = pd.read_csv(
        DATA_DIR / "subtype_labels.csv",
        index_col=0,
    )

    if not X.index.equals(labels.index):
        raise ValueError(
            "Expression and label sample IDs do not match in order."
        )

    if X.index.has_duplicates:
        raise ValueError("Duplicate sample IDs detected.")

    if "subtype" not in labels.columns:
        raise ValueError("The labels file has no subtype column.")

    if labels["subtype"].isna().any():
        raise ValueError("Missing subtype labels detected.")

    X = X.apply(pd.to_numeric, errors="coerce")
    y = labels["subtype"].astype(str)

    groups = pd.Series(
        ["-".join(str(sample_id).split("-")[:3])
         for sample_id in X.index],
        index=X.index,
        name="patient_id",
    )

    if (groups == "").any():
        raise ValueError("Some patient IDs could not be extracted.")

    return X, y, groups


# --------------------------------------------------
# Create a fresh pipeline for each feature count
# --------------------------------------------------

def make_pipeline(k):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("remove_constant_genes", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
        ("scaling", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# --------------------------------------------------
# Compare configurations using grouped CV
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("MY MED AI - FEATURE-SELECTION COMPARISON")
    print("=" * 68)

    X, y, groups = load_data()

    print(f"Samples: {len(X)}")
    print(f"Unique patients: {groups.nunique()}")
    print(f"Available genes: {X.shape[1]}")
    print(f"Feature counts to compare: {FEATURE_COUNTS}")

    if min(y.value_counts()) < N_SPLITS:
        raise ValueError(
            "At least one subtype has too few samples for five-fold CV."
        )

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Reuse identical patient-grouped fold assignments for fair comparison.
    splits = list(cv.split(X, y, groups))

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_patients = set(groups.iloc[train_idx])
        test_patients = set(groups.iloc[test_idx])

        if train_patients.intersection(test_patients):
            raise RuntimeError(
                f"Patient overlap detected in fold {fold}."
            )

    results = []

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": make_scorer(
            f1_score,
            average="weighted",
            zero_division=0,
        ),
    }

    for k in FEATURE_COUNTS:
        print(f"\nEvaluating Logistic Regression with {k} genes...")

        pipeline = make_pipeline(k)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splits,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )

        row = {
            "model": "Logistic Regression",
            "selected_genes": k,
        }

        for metric in scoring:
            values = scores[f"test_{metric}"]

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)

        results.append(row)

        print(
            f"Macro F1: {row['macro_f1_mean']:.4f} "
            f"+/- {row['macro_f1_std']:.4f}"
        )
        print(
            f"Balanced accuracy: "
            f"{row['balanced_accuracy_mean']:.4f} "
            f"+/- {row['balanced_accuracy_std']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "feature_selection_results.csv"
    results_df.to_csv(output_path, index=False)

    # Rank by macro F1, then balanced accuracy.
    ranked = results_df.sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean"],
        ascending=False,
    )

    print("\nComplete results:")
    print(
        results_df[
            [
                "selected_genes",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "weighted_f1_mean",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nBest configuration by mean macro F1:")
    print(ranked.iloc[0].to_string())

    print(f"\nResults saved to: {output_path}")
    print(
        "\nExploratory model selection only. "
        "The best score is not an independent test estimate."
    )


if __name__ == "__main__":
    main()
