
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_candidate"
REPORT_DIR = ROOT / "reports" / "candidate"
EXPLAIN_DIR = REPORT_DIR / "explainability"

OOF_500_DIR = REPORT_DIR / "out_of_fold"
OOF_250_DIR = REPORT_DIR / "out_of_fold_250_genes"
FEATURE_SELECTION_DIR = REPORT_DIR / "feature_selection_comparison"

st.set_page_config(
    page_title="My Med | Genomic Explainability",
    page_icon="🧬",
    layout="wide",
)

st.title("My Med — Genomic Subtype Explorer")
st.caption(
    "Exploratory breast cancer subtype classification "
    "using gene-expression data."
)

st.warning(
    "Research prototype only. This dashboard is not a diagnostic "
    "tool and must not be used to make clinical or treatment decisions."
)


# ============================================================
# DATA LOADING HELPERS
# ============================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_expression_shape(path_string):
    path = Path(path_string)
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0, nrows=1)
    return df.shape[1]


def load_first_existing(*paths):
    """Load the first available CSV from the supplied paths."""
    for path in paths:
        result = load_csv(str(path))
        if result is not None:
            return result
    return None


def show_table(df):
    st.dataframe(df, use_container_width=True, hide_index=False)


# ============================================================
# LOAD EXISTING PROJECT REPORTS
# ============================================================

labels = load_csv(str(DATA_DIR / "subtype_labels.csv"))

# Support the actual alignment report location and the older path.
distribution = load_first_existing(
    ROOT / "reports" / "alignment_candidate" / "subtype_distribution.csv",
    REPORT_DIR / "subtype_distribution.csv",
)

# Prefer the patient-grouped CV results.
grouped_cv = load_first_existing(
    REPORT_DIR / "grouped_cross_validation_results.csv",
    REPORT_DIR / "cross_validation_results.csv",
)

coefficients = load_csv(str(EXPLAIN_DIR / "all_gene_coefficients.csv"))
top_genes = load_csv(str(EXPLAIN_DIR / "top_genes_by_subtype.csv"))
predictions = load_csv(str(EXPLAIN_DIR / "all_sample_predictions.csv"))
contributions = load_csv(str(EXPLAIN_DIR / "sample_gene_contributions.csv"))
training_metrics = load_csv(str(EXPLAIN_DIR / "training_fit_metrics.csv"))

# Out-of-fold evaluation results.
metrics_500 = load_csv(str(OOF_500_DIR / "overall_metrics.csv"))
metrics_250 = load_csv(str(OOF_250_DIR / "overall_metrics.csv"))

comparison_250_500 = load_csv(
    str(OOF_250_DIR / "comparison_250_vs_500_genes.csv")
)

fold_metrics_250 = load_csv(str(OOF_250_DIR / "fold_metrics.csv"))

report_250 = load_csv(
    str(OOF_250_DIR / "subtype_classification_report.csv")
)

report_500 = load_csv(
    str(OOF_500_DIR / "subtype_classification_report.csv")
)

confusion_250 = load_csv(str(OOF_250_DIR / "confusion_matrix.csv"))
confusion_500 = load_csv(str(OOF_500_DIR / "confusion_matrix.csv"))

feature_selection = load_csv(
    str(FEATURE_SELECTION_DIR / "feature_selection_results.csv")
)


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

st.header("1. Dataset overview")

if labels is not None:
    sample_count = len(labels)
    subtype_column = (
        "subtype" if "subtype" in labels.columns else labels.columns[-1]
    )
    subtype_count = labels[subtype_column].nunique()
else:
    sample_count = 0
    subtype_count = 0

gene_count = load_expression_shape(
    str(DATA_DIR / "expression_matrix.csv")
)

m1, m2, m3 = st.columns(3)
m1.metric("Labelled samples", sample_count)
m2.metric(
    "Genes in expression matrix",
    gene_count if gene_count is not None else "Unavailable",
)
m3.metric("Subtype classes", subtype_count)

if distribution is not None:
    st.subheader("Subtype distribution")

    if {"subtype", "count"}.issubset(distribution.columns):
        st.bar_chart(distribution.set_index("subtype")["count"])

    show_table(distribution)
else:
    st.info(
        "Subtype distribution report not found. "
        "Checked reports/alignment_candidate and reports/candidate."
    )


# ============================================================
# 2. PATIENT-GROUPED MODEL COMPARISON
# ============================================================

st.header("2. Patient-grouped cross-validation")

st.caption(
    "These results compare models across five cross-validation folds. "
    "Patient groups are kept separate between training and test sets."
)

if grouped_cv is not None:
    show_table(grouped_cv)

    # Support both wide and long result formats.
    if {"metric", "mean"}.issubset(grouped_cv.columns):
        metric_names = grouped_cv["metric"].dropna().unique().tolist()

        if metric_names:
            chosen_metric = st.selectbox(
                "Cross-validation metric",
                metric_names,
                key="grouped_cv_metric",
            )

            chart_df = grouped_cv[
                grouped_cv["metric"] == chosen_metric
            ].copy()

            if {"model", "mean"}.issubset(chart_df.columns):
                st.bar_chart(chart_df.set_index("model")["mean"])

    elif "model" in grouped_cv.columns:
        possible_metrics = [
            "accuracy_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "weighted_f1_mean",
        ]

        available_metrics = [
            col for col in possible_metrics if col in grouped_cv.columns
        ]

        if available_metrics:
            chosen_metric = st.selectbox(
                "Cross-validation metric",
                available_metrics,
                index=(
                    available_metrics.index("macro_f1_mean")
                    if "macro_f1_mean" in available_metrics
                    else 0
                ),
                key="grouped_cv_wide_metric",
            )

            st.bar_chart(
                grouped_cv.set_index("model")[chosen_metric]
            )
else:
    st.info(
        "Grouped cross-validation report not found. "
        "Expected reports/candidate/grouped_cross_validation_results.csv."
    )


# ============================================================
# 3. OUT-OF-FOLD EVALUATION AND FEATURE-COUNT COMPARISON
# ============================================================

st.header("3. Out-of-fold model evaluation")

st.caption(
    "Every sample receives a prediction from a model that did not train "
    "on that sample. Patient groups are separated within each fold."
)

if comparison_250_500 is not None:
    st.subheader("250 genes versus 500 genes")
    show_table(comparison_250_500)

    if {"configuration", "macro_f1"}.issubset(
        comparison_250_500.columns
    ):
        st.write("Macro F1 comparison")
        st.bar_chart(
            comparison_250_500.set_index("configuration")["macro_f1"]
        )
else:
    st.info(
        "Out-of-fold comparison not found. Run "
        "python src/evaluate_250_gene_model.py"
    )

metric_frames = []

if metrics_250 is not None:
    temp = metrics_250.copy()
    temp["configuration"] = "250 genes"
    metric_frames.append(temp)

if metrics_500 is not None:
    temp = metrics_500.copy()
    temp["configuration"] = "500 genes"
    metric_frames.append(temp)

if metric_frames:
    st.subheader("Overall out-of-fold metrics")

    combined_metrics = pd.concat(metric_frames, ignore_index=True)

    preferred_columns = [
        "configuration",
        "selected_genes",
        "samples",
        "unique_patients",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    display_columns = [
        col for col in preferred_columns if col in combined_metrics.columns
    ]

    show_table(combined_metrics[display_columns])

    st.caption(
        "These are pooled out-of-fold predictions, not independent external "
        "validation results. The feature count was explored using the same "
        "cross-validation framework, so performance may be optimistic."
    )


# ============================================================
# 4. FEATURE-SELECTION COMPARISON
# ============================================================

st.header("4. Feature-selection analysis")

if feature_selection is not None:
    st.caption(
        "Feature counts were compared using patient-grouped "
        "cross-validation. The best mean score is exploratory."
    )

    show_table(feature_selection)

    if {
        "selected_genes",
        "macro_f1_mean",
    }.issubset(feature_selection.columns):
        chart_df = feature_selection.sort_values("selected_genes")

        st.subheader("Macro F1 by number of selected genes")
        st.line_chart(
            chart_df.set_index("selected_genes")["macro_f1_mean"]
        )
else:
    st.info(
        "Feature-selection report not found. Run "
        "python src/compare_feature_selection.py"
    )


# ============================================================
# 5. FOLD-BY-FOLD PERFORMANCE
# ============================================================

st.header("5. Fold-by-fold stability")

if fold_metrics_250 is not None:
    st.caption(
        "250-gene model. Variation across folds indicates sensitivity "
        "to the particular cross-validation split."
    )

    show_table(fold_metrics_250)

    if {"fold", "accuracy"}.issubset(fold_metrics_250.columns):
        chart_df = fold_metrics_250.copy()
        chart_df["fold"] = chart_df["fold"].astype(str)

        st.subheader("Accuracy by fold")
        st.bar_chart(chart_df.set_index("fold")["accuracy"])

    metric_columns = [
        col for col in
        ["accuracy", "balanced_accuracy", "macro_f1"]
        if col in fold_metrics_250.columns
    ]

    if metric_columns:
        st.subheader("Mean and standard deviation across folds")

        fold_summary = fold_metrics_250[metric_columns].agg(["mean", "std"])
        show_table(fold_summary)
else:
    st.info(
        "Fold metrics not found. Run "
        "python src/analyze_250_gene_folds.py"
    )


# ============================================================
# 6. SUBTYPE CLASSIFICATION REPORTS
# ============================================================

st.header("6. Subtype-level performance")

available_reports = {}

if report_250 is not None:
    available_reports["250-gene model"] = report_250

if report_500 is not None:
    available_reports["500-gene model"] = report_500

if available_reports:
    selected_report_name = st.selectbox(
        "Select model report",
        list(available_reports.keys()),
        key="subtype_report_choice",
    )

    selected_report = available_reports[selected_report_name]

    st.caption(
        "Precision measures how often predictions of a subtype are correct. "
        "Recall measures how many actual samples of that subtype are found. "
        "F1 combines precision and recall."
    )

    show_table(selected_report)
else:
    st.info("No out-of-fold subtype classification reports found.")


# ============================================================
# 7. CONFUSION MATRICES
# ============================================================

st.header("7. Confusion matrix")

confusion_options = {}

if confusion_250 is not None:
    confusion_options["250-gene model"] = confusion_250

if confusion_500 is not None:
    confusion_options["500-gene model"] = confusion_500

if confusion_options:
    selected_confusion_name = st.selectbox(
        "Select confusion matrix",
        list(confusion_options.keys()),
        key="confusion_choice",
    )

    selected_confusion = confusion_options[selected_confusion_name]

    st.caption(
        "Rows represent actual subtypes; columns represent predicted "
        "subtypes. Diagonal values are correct classifications."
    )

    show_table(selected_confusion)

    # CSV matrices have actual_subtype stored as the index.
    numeric_matrix = selected_confusion.copy()

    if "actual_subtype" in numeric_matrix.columns:
        numeric_matrix = numeric_matrix.set_index("actual_subtype")

    numeric_matrix = numeric_matrix.apply(pd.to_numeric, errors="coerce")

    if not numeric_matrix.empty and numeric_matrix.notna().all().all():
        st.subheader("Confusion matrix heatmap")
        st.dataframe(
            numeric_matrix.style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )
else:
    st.info("No out-of-fold confusion matrices found.")


# ============================================================
# 8. GLOBAL GENE EXPLAINABILITY
# ============================================================

st.header("8. Global gene explainability")

if top_genes is not None:
    subtypes = sorted(top_genes["subtype"].dropna().unique())

    selected_subtype = st.selectbox(
        "Select subtype",
        subtypes,
        key="global_gene_subtype",
    )

    direction = st.radio(
        "Coefficient direction",
        ["strongest_positive", "strongest_negative"],
        horizontal=True,
        format_func=lambda x: (
            "Supports subtype score"
            if x == "strongest_positive"
            else "Opposes subtype score"
        ),
        key="global_gene_direction",
    )

    gene_view = top_genes[
        (top_genes["subtype"] == selected_subtype)
        & (top_genes["explanation"] == direction)
    ].copy()

    if not gene_view.empty:
        gene_view = gene_view.sort_values(
            "coefficient",
            ascending=(direction == "strongest_negative"),
        )

        st.bar_chart(gene_view.set_index("gene")["coefficient"])

        display_columns = [
            col for col in [
                "gene",
                "coefficient",
                "absolute_coefficient",
                "direction",
            ]
            if col in gene_view.columns
        ]

        show_table(gene_view[display_columns])
else:
    st.info(
        "Gene coefficient reports not found. "
        "Run python src/explain_model.py first."
    )


# ============================================================
# 9. INDIVIDUAL SAMPLE EXPLORER
# ============================================================

st.header("9. Individual sample explorer")

if predictions is not None and contributions is not None:
    if "sample_id" in predictions.columns:
        sample_ids = sorted(
            predictions["sample_id"].dropna().astype(str).unique()
        )

        if sample_ids:
            selected_sample = st.selectbox(
                "Select a sample ID",
                sample_ids,
                key="sample_explorer_id",
            )

            sample_rows = predictions[
                predictions["sample_id"].astype(str) == selected_sample
            ]

            if not sample_rows.empty:
                sample_info = sample_rows.iloc[0]

                a, b, c = st.columns(3)
                a.metric("Actual subtype", sample_info["actual_subtype"])
                b.metric("Predicted subtype", sample_info["predicted_subtype"])

                correct_value = sample_info.get("prediction_correct", False)

                if pd.isna(correct_value):
                    correct_display = "Unavailable"
                else:
                    correct_display = (
                        "Yes" if str(correct_value).lower() in
                        {"true", "1", "yes"} else "No"
                    )

                c.metric("Prediction match", correct_display)

                sample_contributions = contributions[
                    contributions["sample_id"].astype(str) == selected_sample
                ].copy()

                if not sample_contributions.empty:
                    sample_contributions = sample_contributions.sort_values(
                        "contribution",
                        key=lambda values: values.abs(),
                        ascending=False,
                    )

                    st.subheader("Largest gene contributions")
                    st.caption(
                        "Positive contributions increase the predicted "
                        "class's decision score; negative contributions "
                        "decrease it. These are not probabilities or "
                        "causal effects."
                    )

                    if {"gene", "contribution"}.issubset(
                        sample_contributions.columns
                    ):
                        st.bar_chart(
                            sample_contributions.set_index("gene")[
                                "contribution"
                            ]
                        )

                    display_columns = [
                        col for col in [
                            "gene",
                            "scaled_expression",
                            "coefficient",
                            "contribution",
                            "direction",
                        ]
                        if col in sample_contributions.columns
                    ]

                    show_table(sample_contributions[display_columns])
                else:
                    st.info("No gene contributions found for this sample.")
        else:
            st.info("No sample IDs found in the prediction report.")
else:
    st.info(
        "Sample-level reports not found. "
        "Run python src/explain_model.py first."
    )


# ============================================================
# 10. FINAL FITTED MODEL
# ============================================================

st.header("10. Final fitted model")

if training_metrics is not None:
    show_table(training_metrics)

    st.caption(
        "These metrics measure fit on training samples. They are not "
        "an independent estimate of generalisation."
    )
else:
    st.info("Training-fit metrics are not available.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "My Med research prototype | Gene coefficients represent model "
    "associations, not biological causation. PAM50 subtype prediction "
    "in this dataset does not establish clinical validity or support "
    "individual treatment recommendations."
)