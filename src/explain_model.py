
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed_candidate"
MODEL_DIR = ROOT / "models" / "candidate"
REPORT_DIR = ROOT / "reports" / "candidate"
EXPLAIN_DIR = REPORT_DIR / "explainability"

RANDOM_STATE = 42
TOP_K_GENES = 500
TOP_N_GENES = 20

EXPRESSION_PATH = DATA_DIR / "expression_matrix.csv"
LABELS_PATH = DATA_DIR / "subtype_labels.csv"

MODEL_PATH = MODEL_DIR / "explainable_logistic_regression.joblib"


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    print("1. Loading processed expression data and subtype labels...")

    X = pd.read_csv(EXPRESSION_PATH, index_col=0)
    labels = pd.read_csv(LABELS_PATH, index_col=0)

    if "subtype" not in labels.columns:
        raise ValueError("Labels file must contain a 'subtype' column.")

    if not X.index.equals(labels.index):
        raise ValueError(
            "Expression samples and subtype labels are not aligned."
        )

    if not X.index.is_unique:
        raise ValueError("Duplicate sample IDs found.")

    if labels["subtype"].isna().any():
        raise ValueError("Missing subtype labels found.")

    # Ensure expression values are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove genes that have no usable values at all.
    X = X.dropna(axis=1, how="all")

    if X.shape[1] < TOP_K_GENES:
        raise ValueError(
            f"Only {X.shape[1]} usable genes remain; "
            f"at least {TOP_K_GENES} are required."
        )

    y = labels["subtype"].astype(str)

    if y.nunique() < 2:
        raise ValueError("At least two subtype classes are required.")

    print(f"Samples: {X.shape[0]}")
    print(f"Genes available: {X.shape[1]}")
    print(f"Subtype classes: {y.nunique()}")

    return X, y


# ============================================================
# BUILD AND TRAIN THE FINAL MODEL
# ============================================================

def build_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "remove_constant_genes",
                VarianceThreshold(threshold=0.0),
            ),
            (
                "feature_selection",
                SelectKBest(
                    score_func=f_classif,
                    k=TOP_K_GENES,
                ),
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
        ]
    )


# ============================================================
# GET THE SELECTED GENE NAMES
# ============================================================

def get_selected_gene_names(model, original_gene_names):
    imputer = model.named_steps["imputer"]
    variance_filter = model.named_steps["remove_constant_genes"]
    selector = model.named_steps["feature_selection"]

    # Imputer and variance filtering preserve column order.
    after_imputation = imputer.get_feature_names_out(
        original_gene_names
    )

    after_variance_filter = variance_filter.get_feature_names_out(
        after_imputation
    )

    selected_genes = selector.get_feature_names_out(
        after_variance_filter
    )

    return np.asarray(selected_genes, dtype=str)


# ============================================================
# GLOBAL EXPLAINABILITY
# ============================================================

def save_global_explanations(model, selected_genes):
    print("\n3. Generating global gene-coefficient reports...")

    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_
    classes = classifier.classes_

    if coefficients.shape[1] != len(selected_genes):
        raise ValueError(
            "Coefficient count does not match selected gene count."
        )

    rows = []

    for class_index, subtype in enumerate(classes):
        for gene_index, gene in enumerate(selected_genes):
            coefficient = float(
                coefficients[class_index, gene_index]
            )

            rows.append(
                {
                    "subtype": subtype,
                    "gene": gene,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                    "direction": (
                        "positive"
                        if coefficient > 0
                        else "negative"
                        if coefficient < 0
                        else "zero"
                    ),
                }
            )

    all_coefficients = pd.DataFrame(rows)

    all_coefficients.to_csv(
        EXPLAIN_DIR / "all_gene_coefficients.csv",
        index=False,
    )

    # Save the strongest positive and negative coefficients
    # for each subtype separately.
    top_rows = []

    for subtype in classes:
        subtype_df = all_coefficients[
            all_coefficients["subtype"] == subtype
        ]

        positive = subtype_df.nlargest(
            TOP_N_GENES, "coefficient"
        ).copy()
        positive["explanation"] = "strongest_positive"

        negative = subtype_df.nsmallest(
            TOP_N_GENES, "coefficient"
        ).copy()
        negative["explanation"] = "strongest_negative"

        top_rows.extend(
            positive.to_dict("records")
        )
        top_rows.extend(
            negative.to_dict("records")
        )

    top_coefficients = pd.DataFrame(top_rows)

    top_coefficients.to_csv(
        EXPLAIN_DIR / "top_genes_by_subtype.csv",
        index=False,
    )

    print(
        "Saved:",
        EXPLAIN_DIR / "all_gene_coefficients.csv",
    )
    print(
        "Saved:",
        EXPLAIN_DIR / "top_genes_by_subtype.csv",
    )


# ============================================================
# SAMPLE-LEVEL EXPLAINABILITY
# ============================================================

def save_sample_explanations(
    model,
    X,
    y,
    selected_genes,
):
    print("\n4. Generating sample-level explanations...")

    imputer = model.named_steps["imputer"]
    variance_filter = model.named_steps["remove_constant_genes"]
    selector = model.named_steps["feature_selection"]
    scaler = model.named_steps["scaling"]
    classifier = model.named_steps["classifier"]

    # Apply the exact transformations learned by the fitted pipeline.
    X_imputed = imputer.transform(X)
    X_variable = variance_filter.transform(X_imputed)
    X_selected = selector.transform(X_variable)
    X_scaled = scaler.transform(X_selected)

    predicted = classifier.predict(X_scaled)
    decision_scores = classifier.decision_function(X_scaled)

    classes = classifier.classes_
    coefficients = classifier.coef_
    intercepts = classifier.intercept_

    # For each class, a linear model's decision score is:
    # intercept + sum(scaled_gene_value * gene_coefficient).
    # This is a score decomposition, not a probability decomposition.
    if decision_scores.ndim != 2:
        raise ValueError(
            "Expected a multiclass decision-score matrix."
        )

    sample_rows = []
    contribution_rows = []

    for sample_position, sample_id in enumerate(X.index):
        predicted_subtype = predicted[sample_position]
        actual_subtype = y.loc[sample_id]

        predicted_class_index = np.where(
            classes == predicted_subtype
        )[0][0]

        sample_values = X_scaled[sample_position]
        class_coefficients = coefficients[predicted_class_index]

        contributions = sample_values * class_coefficients

        score_from_contributions = (
            float(intercepts[predicted_class_index])
            + float(contributions.sum())
        )

        # Verify that the contribution decomposition matches
        # the model's decision score for the predicted class.
        expected_score = float(
            decision_scores[
                sample_position, predicted_class_index
            ]
        )

        if not np.isclose(
            score_from_contributions,
            expected_score,
            atol=1e-5,
        ):
            raise ValueError(
                "Contribution scores do not match the model's "
                "decision score. Check the fitted model and "
                "scikit-learn version."
            )

        sample_rows.append(
            {
                "sample_id": sample_id,
                "actual_subtype": actual_subtype,
                "predicted_subtype": predicted_subtype,
                "prediction_correct": (
                    actual_subtype == predicted_subtype
                ),
                "predicted_class_decision_score": expected_score,
                "intercept": float(
                    intercepts[predicted_class_index]
                ),
            }
        )

        # Keep both directions for the predicted subtype.
        # Positive values raise its decision score;
        # negative values lower its decision score.
        strongest_indices = np.argsort(
            np.abs(contributions)
        )[::-1][:TOP_N_GENES]

        for gene_index in strongest_indices:
            contribution = float(contributions[gene_index])

            contribution_rows.append(
                {
                    "sample_id": sample_id,
                    "actual_subtype": actual_subtype,
                    "predicted_subtype": predicted_subtype,
                    "gene": selected_genes[gene_index],
                    "scaled_expression": float(
                        sample_values[gene_index]
                    ),
                    "coefficient": float(
                        class_coefficients[gene_index]
                    ),
                    "contribution": contribution,
                    "direction": (
                        "supports_predicted_subtype"
                        if contribution > 0
                        else "opposes_predicted_subtype"
                        if contribution < 0
                        else "neutral"
                    ),
                }
            )

    predictions_df = pd.DataFrame(sample_rows)
    contributions_df = pd.DataFrame(contribution_rows)

    predictions_df.to_csv(
        EXPLAIN_DIR / "all_sample_predictions.csv",
        index=False,
    )

    contributions_df.to_csv(
        EXPLAIN_DIR / "sample_gene_contributions.csv",
        index=False,
    )

    print(
        "Saved:",
        EXPLAIN_DIR / "all_sample_predictions.csv",
    )
    print(
        "Saved:",
        EXPLAIN_DIR / "sample_gene_contributions.csv",
    )

    return predictions_df


# ============================================================
# TRAINING-SET PERFORMANCE SUMMARY
# ============================================================

def save_training_summary(y, predictions_df):
    print("\n5. Saving training-set summary...")

    y_pred = predictions_df.set_index(
        "sample_id"
    ).loc[y.index, "predicted_subtype"]

    metrics = pd.DataFrame(
        [
            {
                "evaluation": "training_data_resubstitution",
                "accuracy": accuracy_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(
                    y, y_pred
                ),
                "macro_f1": f1_score(
                    y, y_pred, average="macro", zero_division=0
                ),
                "weighted_f1": f1_score(
                    y, y_pred, average="weighted", zero_division=0
                ),
                "sample_count": len(y),
            }
        ]
    )

    metrics.to_csv(
        EXPLAIN_DIR / "training_fit_metrics.csv",
        index=False,
    )

    print(metrics.to_string(index=False))
    print(
        "\nNote: these are training-set scores, not "
        "independent test or cross-validation results."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MY MED - LOGISTIC REGRESSION EXPLAINABILITY")
    print("=" * 65)

    X, y = load_data()

    print("\n2. Fitting final model on all available labelled samples...")

    model = build_model()
    model.fit(X, y)

    selected_genes = get_selected_gene_names(
        model,
        X.columns.to_numpy(dtype=str),
    )

    joblib.dump(model, MODEL_PATH)

    pd.DataFrame(
        {"gene": selected_genes}
    ).to_csv(
        EXPLAIN_DIR / "selected_genes.csv",
        index=False,
    )

    print(f"Final fitted model saved to: {MODEL_PATH}")
    print(f"Selected genes: {len(selected_genes)}")

    save_global_explanations(model, selected_genes)

    predictions_df = save_sample_explanations(
        model,
        X,
        y,
        selected_genes,
    )

    save_training_summary(y, predictions_df)

    print("\n" + "=" * 65)
    print("EXPLAINABILITY REPORTS COMPLETED")
    print("=" * 65)
    print(f"Output folder: {EXPLAIN_DIR}")
    print(
        "\nResearch caution: coefficient-based explanations "
        "describe model behaviour, not biological causation "
        "or clinical validity."
    )


if __name__ == "__main__":
    main()
