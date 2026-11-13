
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_candidate"
REPORT_DIR = ROOT / "reports" / "candidate"

RANDOM_STATE = 42
N_SPLITS = 5
TOP_K_GENES = 500


def load_data():
    expression_path = DATA_DIR / "expression_matrix.csv"
    labels_path = DATA_DIR / "subtype_labels.csv"

    X = pd.read_csv(expression_path, index_col=0)
    labels = pd.read_csv(labels_path, index_col=0)

    if "subtype" not in labels.columns:
        raise ValueError("Labels must contain a 'subtype' column.")

    if not X.index.equals(labels.index):
        raise ValueError("Expression and label sample IDs are not aligned.")

    if X.index.has_duplicates:
        raise ValueError("Duplicate sample IDs found.")

    if labels["subtype"].isna().any():
        raise ValueError("Missing subtype labels found.")

    X = X.apply(pd.to_numeric, errors="coerce")
    y = labels["subtype"].astype(str)

    # TCGA barcode: first three fields identify the patient.
    sample_ids = X.index.astype(str)
    groups = pd.Series(
        ["-".join(sample_id.split("-")[:3]) for sample_id in sample_ids],
        index=X.index,
        name="patient_id",
    )

    if groups.isna().any() or (groups == "").any():
        raise ValueError("Could not derive patient IDs from all sample IDs.")

    if y.value_counts().min() < N_SPLITS:
        raise ValueError("At least five samples per subtype are required.")

    return X, y, groups


def make_pipeline(classifier, scale):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("remove_constant_genes", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=TOP_K_GENES)),
    ]

    if scale:
        steps.append(("scaling", StandardScaler()))

    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MY MED - PATIENT-GROUPED CROSS-VALIDATION")
    print("=" * 65)

    X, y, groups = load_data()

    print(f"\nSamples: {len(X)}")
    print(f"Unique patients: {groups.nunique()}")
    print(f"Genes supplied to pipeline: {X.shape[1]}")
    print("\nSubtype distribution:")
    print(y.value_counts().to_string())

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale=False,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        for metric in scoring:
            values = scores[f"test_{metric}"]

            results.append({
                "model": name,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
            })

            print(
                f"{metric}: {values.mean():.4f} "
                f"+/- {values.std(ddof=1):.4f}"
            )

    results_df = pd.DataFrame(results)

    # Preserve the earlier sample-level CV results.
    output_path = REPORT_DIR / "grouped_cross_validation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nMean patient-grouped cross-validation scores:")
    print(
        results_df.pivot(
            index="model",
            columns="metric",
            values="mean",
        ).round(4).to_string()
    )

    print(f"\nSaved results to: {output_path}")
    print("\nPatient-grouped cross-validation completed.")
    print("Exploratory research only; not clinical validation.")


if __name__ == "__main__":
    main()
