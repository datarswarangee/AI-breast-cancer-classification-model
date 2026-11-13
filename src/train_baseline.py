
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_candidate"
MODEL_DIR = ROOT / "models" / "candidate"
REPORT_DIR = ROOT / "reports" / "candidate"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_GENES = 500


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MY MED - BASELINE SUBTYPE CLASSIFIER")
    print("=" * 65)

    # 1. Load the prepared files.
    X = pd.read_csv(
        DATA_DIR / "expression_matrix.csv",
        index_col=0,
    )
    y_df = pd.read_csv(
        DATA_DIR / "subtype_labels.csv",
        index_col=0,
    )

    if "subtype" not in y_df.columns:
        raise ValueError(
            "Expected a 'subtype' column in subtype_labels.csv."
        )

    y = y_df["subtype"].astype(str)

    # 2. Verify IDs and data quality.
    if not X.index.equals(y.index):
        raise ValueError(
            "Expression and label sample IDs or order do not match."
        )

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError("Duplicate sample IDs detected.")

    if X.isna().any().any():
        raise ValueError(
            "Missing expression values found. Resolve them first."
        )

    if y.isna().any():
        raise ValueError("Missing subtype labels found.")

    print(f"\nSamples: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print("\nClass distribution:")
    print(y.value_counts().to_string())

    # 3. Split before fitting any preprocessing or feature selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # 4. Build the leakage-safe ML pipeline.
    k = min(TOP_K_GENES, X_train.shape[1])

    model = Pipeline(
    steps=[
        # Remove genes with zero variance in the training data.
        ("remove_constant_genes", VarianceThreshold(threshold=0.0)),

        # Select the top 500 genes using training data only.
        ("feature_selection", SelectKBest(score_func=f_classif, k=500)),

        ("scaling", StandardScaler()),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
                ),
        ),
        ]
    )

    # 5. Train using training data only.
    print("\nTraining logistic regression...")
    model.fit(X_train, y_train)

    # 6. Evaluate once on the held-out test set.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print("\n" + "=" * 65)
    print("TEST-SET RESULTS")
    print("=" * 65)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    report = classification_report(
        y_test,
        predictions,
        labels=sorted(y.unique()),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_DIR / "classification_report.csv")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=sorted(y.unique()),
            zero_division=0,
        )
    )

    # 7. Save confusion matrix with explicit class names.
    classes = sorted(y.unique())
    cm = confusion_matrix(y_test, predictions, labels=classes)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in classes],
        columns=[f"Predicted_{c}" for c in classes],
    )
    cm_df.to_csv(REPORT_DIR / "confusion_matrix.csv")

    # Save scalar metrics separately.
    metrics = pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy},
            {"metric": "balanced_accuracy", "value": balanced_accuracy},
            {"metric": "macro_f1", "value": macro_f1},
            {"metric": "weighted_f1", "value": weighted_f1},
        ]
    )
    metrics.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)

    # Save predictions so each result can be traced to a sample ID.
    predictions_df = pd.DataFrame(
        {
            "sample_id": X_test.index,
            "actual_subtype": y_test.values,
            "predicted_subtype": predictions,
        }
    )
    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv", index=False
    )

    # Save the complete fitted pipeline.
    joblib.dump(model, MODEL_DIR / "baseline_logistic_regression.joblib")

    print("\nSaved files:")
    print("models/baseline_logistic_regression.joblib")
    print("reports/baseline_metrics.csv")
    print("reports/classification_report.csv")
    print("reports/confusion_matrix.csv")
    print("reports/test_predictions.csv")

    print("\nBaseline training completed.")
    print(
        "Research note: this is an exploratory model, not a "
        "clinically validated diagnostic system."
    )


if __name__ == "__main__":
    main()
