
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# MY MED: DAY 2 - DATA ALIGNMENT AND PREPROCESSING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

CLINICAL_FILE = RAW_DIR / "TCGA_BRCA_clinicalMatrix_deidentified.tsv"
EXPRESSION_FILE = RAW_DIR / "TCGA.BRCA.sampleMap_HiSeqV2.tsv"

ID_COLUMN = "_GENOMIC_ID_TCGA_BRCA_PDMarray"
LABEL_COLUMN = "PAM50Call_RNAseq"

EXPECTED_LABELS = {"LumA", "LumB", "Basal", "Her2", "Normal"}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MY MED - DATA PREPARATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load clinical metadata
    # --------------------------------------------------------
    print("\n1. Loading clinical metadata...")

    clinical = pd.read_csv(
        CLINICAL_FILE,
        sep="\t",
        low_memory=False
    )

    required = [ID_COLUMN, LABEL_COLUMN]

    missing_columns = [
        column for column in required
        if column not in clinical.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required clinical columns are missing: {missing_columns}"
        )

    clinical[ID_COLUMN] = (
        clinical[ID_COLUMN].astype("string").str.strip()
    )

    clinical[LABEL_COLUMN] = (
        clinical[LABEL_COLUMN].astype("string").str.strip()
    )

    valid_label = (
        clinical[LABEL_COLUMN].notna()
        & ~clinical[LABEL_COLUMN].str.upper().isin(
            ["", "NA", "N/A", "NAN", "NULL"]
        )
    )

    clinical = clinical.loc[valid_label].copy()

    unexpected = set(clinical[LABEL_COLUMN].unique()) - EXPECTED_LABELS

    if unexpected:
        raise ValueError(f"Unexpected subtype labels: {unexpected}")

    # Check sample identifiers
    
    print("2. Checking sample identifiers...")
    id_col = "_GENOMIC_ID_TCGA_BRCA_PDMarray"
    label_col = "PAM50Call_RNAseq"
    
    # Remove rows without a sample ID
    clinical = clinical.dropna(subset=[id_col]).copy()
    
    # Normalize sample IDs
    clinical[id_col] = clinical[id_col].astype(str).str.strip()
    
    # Remove empty identifiers
    clinical = clinical[clinical[id_col] != ""].copy()
    
    # Check for duplicate IDs among rows with actual identifiers
    duplicates = clinical[
        clinical.duplicated(subset=[id_col], keep=False)
        ].sort_values(id_col)
    
    if not duplicates.empty:
        duplicates.to_csv(
            "reports/duplicate_clinical_ids.csv",
            index=False
              )
        raise ValueError(
             "Duplicate clinical sample IDs found."
             "Inspect reports/duplicate_clinical_ids.csv."
             )
        
        print(f"Clinical rows with sample IDs: {len(clinical)}")
        print(f"Unique clinical sample IDs: {clinical[id_col].nunique()}")

    # --------------------------------------------------------
    # 3. Load expression matrix
    # --------------------------------------------------------
    print("3. Loading expression matrix...")
    print("This step may take some time.")

    expression = pd.read_csv(
        EXPRESSION_FILE,
        sep="\t",
        index_col=0,
        low_memory=False
    )

    print("Expression matrix shape (genes x samples):", expression.shape)

    if not expression.index.is_unique:
        raise ValueError(
            "Duplicate gene identifiers found. "
            "Investigate them before proceeding."
        )

    # --------------------------------------------------------
    # 4. Align expression samples and labels
    # --------------------------------------------------------
    print("4. Matching samples and labels...")

    expression.columns = expression.columns.astype(str).str.strip()

    if not expression.columns.is_unique:
        raise ValueError("Duplicate expression sample IDs found.")

    common_ids = [
        sample_id
        for sample_id in clinical[ID_COLUMN].tolist()
        if sample_id in expression.columns
    ]

    if not common_ids:
        raise ValueError(
            "No common sample IDs found. Check the identifier mapping."
        )

    aligned_clinical = (
        clinical.set_index(ID_COLUMN)
        .loc[common_ids]
    )

    # Samples become rows; genes become columns.
    X = expression.loc[:, common_ids].T

    # Convert every expression feature to numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Reject columns that are entirely non-numeric.
    empty_features = X.columns[X.isna().all()].tolist()

    if empty_features:
        raise ValueError(
            f"{len(empty_features)} genes contain no numeric values. "
            "Inspect the expression file format."
        )

    y = aligned_clinical[LABEL_COLUMN].rename("subtype")

    # Explicitly verify alignment.
    if not X.index.equals(y.index):
        raise ValueError("Expression and label indexes are misaligned.")

    if not X.index.is_unique:
        raise ValueError("Duplicate sample rows remain after alignment.")

    # --------------------------------------------------------
    # 5. Report missing values and class distribution
    # --------------------------------------------------------
    print("5. Inspecting data quality...")

    missing_cells = int(X.isna().sum().sum())
    total_cells = int(X.shape[0] * X.shape[1])

    missing_percent = (
        100 * missing_cells / total_cells if total_cells else 0
    )

    class_counts = y.value_counts().reindex(
        sorted(EXPECTED_LABELS),
        fill_value=0
    )

    print("\nAligned sample count:", len(X))
    print("Gene count:", X.shape[1])
    print("Missing expression cells:", missing_cells)
    print(f"Missing expression percentage: {missing_percent:.4f}%")

    print("\nSubtype distribution:")
    print(class_counts.to_string())

    if (class_counts == 0).any():
        print(
            "\nWARNING: At least one expected class is absent "
            "from the aligned cohort."
        )

    # --------------------------------------------------------
    # 6. Save processed data and provenance
    # --------------------------------------------------------
    print("\n6. Saving processed files...")

    X.to_csv(PROCESSED_DIR / "expression_matrix.csv", index_label="sample_id")

    y.to_csv(PROCESSED_DIR / "subtype_labels.csv", index_label="sample_id")

    summary = pd.DataFrame({
        "sample_id": X.index,
        "subtype": y.values
    })

    summary.to_csv(
        PROCESSED_DIR / "sample_manifest.csv",
        index=False
    )

    class_counts.rename_axis("subtype").reset_index(
        name="sample_count"
    ).to_csv(
        REPORTS_DIR / "subtype_distribution.csv",
        index=False
    )

    print("\nSaved:")
    print("data/processed/expression_matrix.csv")
    print("data/processed/subtype_labels.csv")
    print("data/processed/sample_manifest.csv")
    print("reports/subtype_distribution.csv")

    print("\nPreparation completed.")
    print("No model was trained.")


if __name__ == "__main__":
    main()
