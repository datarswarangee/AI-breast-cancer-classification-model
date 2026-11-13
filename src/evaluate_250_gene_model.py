
from pathlib import Path

import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_candidate"
EVAL_DIR = ROOT / "reports" / "candidate" / "out_of_fold"
OUTPUT_DIR = ROOT / "reports" / "candidate" / "out_of_fold_250_genes"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_K_GENES = 250


def load_data():
    X = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    labels = pd.read_csv(DATA_DIR / "subtype_labels.csv", index_col=0)

    if not X.index.equals(labels.index):
        raise ValueError("Sample IDs do not match or are in a different order.")

    if X.index.has_duplicates:
        raise ValueError("Duplicate sample IDs detected.")

    if "subtype" not in labels.columns or labels["subtype"].isna().any():
        raise ValueError("Missing or invalid subtype labels.")

    X = X.apply(pd.to_numeric, errors="coerce")
    y = labels["subtype"].astype(str)

    groups = pd.Series(
        ["-".join(str(sample_id).split("-")[:3]) for sample_id in X.index],
        index=X.index,
    )

    if (groups == "").any():
        raise ValueError("Could not extract every patient ID.")

    return X, y, groups


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("remove_constant_genes", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=TOP_K_GENES)),
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MY MED AI - 250-GENE OUT-OF-FOLD EVALUATION")
    print("=" * 65)

    X, y, groups = load_data()

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = pd.Series(index=X.index, dtype="object")
    fold_numbers = pd.Series(index=X.index, dtype="Int64")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(f"Patient overlap detected in fold {fold}.")

        model = make_pipeline()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions.iloc[test_idx] = model.predict(X.iloc[test_idx])
        fold_numbers.iloc[test_idx] = fold

        print(
            f"Fold {fold}: train={len(train_idx)}, "
            f"test={len(test_idx)}, patient overlap=0"
        )

    if predictions.isna().any():
        raise RuntimeError("Some samples did not receive predictions.")

    prediction_df = pd.DataFrame({
        "sample_id": X.index,
        "patient_id": groups.values,
        "actual_subtype": y.values,
        "predicted_subtype": predictions.values,
        "fold": fold_numbers.astype(int).values,
    })
    prediction_df["correct"] = (
        prediction_df["actual_subtype"] == prediction_df["predicted_subtype"]
    )

    prediction_df.to_csv(
        OUTPUT_DIR / "out_of_fold_predictions.csv",
        index=False,
    )

    metrics = pd.DataFrame([{
        "selected_genes": TOP_K_GENES,
        "samples": len(y),
        "unique_patients": groups.nunique(),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
    }])

    metrics.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)

    class_names = sorted(y.unique())

    report = classification_report(
        y,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "subtype_classification_report.csv"
    )

    matrix = confusion_matrix(y, predictions, labels=class_names)

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).rename_axis("actual_subtype").to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("\n250-gene out-of-fold metrics:")
    print(metrics.round(4).to_string(index=False))

    print("\nSubtype classification report:")
    print(
        pd.DataFrame(report).transpose().round(3).to_string()
    )

    print("\nConfusion matrix:")
    print(pd.DataFrame(matrix, index=class_names, columns=class_names))

    # Compare with the already saved 500-gene out-of-fold results.
    baseline_path = EVAL_DIR / "out_of_fold_predictions.csv"

    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)

        if (
            baseline["sample_id"].nunique() == len(X)
            and set(baseline["sample_id"]) == set(X.index)
        ):
            baseline = baseline.set_index("sample_id").loc[X.index]

            comparison = []

            for name, pred in [
                ("500 genes", baseline["predicted_subtype"].astype(str)),
                ("250 genes", predictions.astype(str)),
            ]:
                comparison.append({
                    "configuration": name,
                    "accuracy": accuracy_score(y, pred),
                    "balanced_accuracy": balanced_accuracy_score(y, pred),
                    "macro_f1": f1_score(
                        y, pred, average="macro", zero_division=0
                    ),
                    "weighted_f1": f1_score(
                        y, pred, average="weighted", zero_division=0
                    ),
                })

            comparison_df = pd.DataFrame(comparison)
            comparison_df.to_csv(
                OUTPUT_DIR / "comparison_250_vs_500_genes.csv",
                index=False,
            )

            print("\n250-gene vs 500-gene comparison:")
            print(comparison_df.round(4).to_string(index=False))
        else:
            print("\n500-gene comparison skipped: sample IDs do not match.")
    else:
        print("\n500-gene comparison skipped: baseline predictions not found.")

    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("Research evaluation only; not clinical validation.")


if __name__ == "__main__":
    main()
