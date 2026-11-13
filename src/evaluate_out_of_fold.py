
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_candidate"
REPORT_DIR = ROOT / "reports" / "candidate"
EVALUATION_DIR = REPORT_DIR / "out_of_fold"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_K_GENES = 500


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

def load_data():
    expression_path = DATA_DIR / "expression_matrix.csv"
    labels_path = DATA_DIR / "subtype_labels.csv"

    X = pd.read_csv(expression_path, index_col=0)
    labels = pd.read_csv(labels_path, index_col=0)

    if not X.index.equals(labels.index):
        raise ValueError(
            "Sample IDs do not match or appear in different orders."
        )

    if X.index.has_duplicates:
        raise ValueError("Duplicate sample IDs detected.")

    if "subtype" not in labels.columns:
        raise ValueError("The labels file must contain a subtype column.")

    if labels["subtype"].isna().any():
        raise ValueError("Missing subtype labels detected.")

    X = X.apply(pd.to_numeric, errors="coerce")
    y = labels["subtype"].astype(str)

    # TCGA patient identifier: first three barcode components.
    groups = pd.Series(
        ["-".join(str(sample_id).split("-")[:3])
         for sample_id in X.index],
        index=X.index,
        name="patient_id",
    )

    if (groups == "").any():
        raise ValueError("Could not extract all patient IDs.")

    if X.isna().all(axis=0).any():
        raise ValueError(
            "At least one gene contains no numeric observations."
        )

    print(f"Samples: {len(X)}")
    print(f"Genes: {X.shape[1]}")
    print(f"Unique patients: {groups.nunique()}")
    print("\nSubtype distribution:")
    print(y.value_counts())

    return X, y, groups


# --------------------------------------------------
# 3. Define the model pipeline
# --------------------------------------------------

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("remove_constant_genes", VarianceThreshold(threshold=0.0)),
        (
            "feature_selection",
            SelectKBest(score_func=f_classif, k=TOP_K_GENES),
        ),
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
# 4. Patient-grouped out-of-fold evaluation
# --------------------------------------------------

def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MY MED AI - OUT-OF-FOLD EVALUATION")
    print("=" * 65)

    X, y, groups = load_data()

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = pd.Series(index=X.index, dtype="object")
    fold_numbers = pd.Series(index=X.index, dtype="Int64")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_patients = set(groups.iloc[train_idx])
        test_patients = set(groups.iloc[test_idx])

        overlap = train_patients.intersection(test_patients)

        if overlap:
            raise RuntimeError(
                f"Patient leakage detected in fold {fold}."
            )

        model = make_pipeline()
        model.fit(X_train, y_train)

        fold_predictions = model.predict(X_test)

        predictions.iloc[test_idx] = fold_predictions
        fold_numbers.iloc[test_idx] = fold

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"patient overlap={len(overlap)}"
        )

    if predictions.isna().any():
        raise RuntimeError(
            "Some samples did not receive out-of-fold predictions."
        )

    # --------------------------------------------------
    # 5. Save sample-level predictions
    # --------------------------------------------------

    prediction_df = pd.DataFrame({
        "sample_id": X.index,
        "patient_id": groups.values,
        "actual_subtype": y.values,
        "predicted_subtype": predictions.values,
        "fold": fold_numbers.astype(int).values,
    })

    prediction_df["correct"] = (
        prediction_df["actual_subtype"]
        == prediction_df["predicted_subtype"]
    )

    prediction_df.to_csv(
        EVALUATION_DIR / "out_of_fold_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. Calculate overall evaluation metrics
    # --------------------------------------------------

    metrics = pd.DataFrame([{
        "samples": len(y),
        "unique_patients": groups.nunique(),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
    }])

    metrics.to_csv(
        EVALUATION_DIR / "overall_metrics.csv",
        index=False,
    )

    # --------------------------------------------------
    # 7. Save subtype-level precision, recall and F1
    # --------------------------------------------------

    report = classification_report(
        y,
        predictions,
        labels=sorted(y.unique()),
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        EVALUATION_DIR / "subtype_classification_report.csv"
    )

    # --------------------------------------------------
    # 8. Save and plot confusion matrix
    # --------------------------------------------------

    class_names = sorted(y.unique())

    matrix = confusion_matrix(
        y,
        predictions,
        labels=class_names,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.index.name = "actual_subtype"
    matrix_df.columns.name = "predicted_subtype"

    matrix_df.to_csv(
        EVALUATION_DIR / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(matrix, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted subtype",
        ylabel="Actual subtype",
        title="My Med AI - Out-of-Fold Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        EVALUATION_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # 9. Display summary
    # --------------------------------------------------

    print("\nOverall out-of-fold metrics:")
    print(metrics.round(4).to_string(index=False))

    print("\nSubtype classification report:")
    print(report_df.round(3).to_string())

    print("\nConfusion matrix:")
    print(matrix_df)

    print(f"\nResults saved in: {EVALUATION_DIR}")
    print("Evaluation complete.")
    print("Research prototype only; not clinical validation.")


if __name__ == "__main__":
    main()
