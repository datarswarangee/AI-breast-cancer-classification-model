
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

CLINICAL_FILE = RAW_DIR / "TCGA_BRCA_clinicalMatrix_deidentified.tsv"
EXPRESSION_FILE = RAW_DIR / "TCGA.BRCA.sampleMap_HiSeqV2.tsv"

LABEL_COLUMN = "PAM50Call_RNAseq"
SAMPLE_KEY = "_GENOMIC_ID_TCGA_BRCA_exp_HiSeqV2"


def main():
    print("=" * 65)
    print("MY MED - DAY 1 DATA VERIFICATION")
    print("=" * 65)

    # 1. Confirm that both files exist.
    for file_path in [CLINICAL_FILE, EXPRESSION_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}\n"
                "Download the file from the specified public dataset "
                "and place it in data/raw/."
            )

        print(f"\nFound: {file_path.name}")
        print(f"Size: {file_path.stat().st_size / (1024**2):.2f} MB")

    # 2. Load clinical metadata.
    print("\nLoading clinical metadata...")
    clinical = pd.read_csv(
        CLINICAL_FILE,
        sep="\t",
        low_memory=False
    )

    print("Clinical shape:", clinical.shape)

    # 3. Verify required columns.
    print("\nChecking important clinical columns...")

    for column in [LABEL_COLUMN, SAMPLE_KEY]:
        if column in clinical.columns:
            print(f"[OK] {column}")
        else:
            print(f"[MISSING] {column}")

    # Show related columns if the exact names differ.
    print("\nColumns containing PAM50:")
    print([c for c in clinical.columns if "PAM50" in c.upper()])

    print("\nColumns containing genomic ID:")
    print([c for c in clinical.columns if "GENOMIC_ID" in c.upper()])

    if LABEL_COLUMN not in clinical.columns:
        raise ValueError(
            "Expected RNA-seq PAM50 label column was not found. "
            "Inspect the printed columns before continuing."
        )

    if SAMPLE_KEY not in clinical.columns:
        raise ValueError(
            "Expected expression sample-ID mapping column was not found. "
            "Inspect the printed columns before continuing."
        )

    # 4. Inspect label availability and class balance.
    labels = clinical[LABEL_COLUMN].astype("string").str.strip()

    valid_labels = labels.notna() & ~labels.str.upper().isin(
        ["", "NA", "N/A", "NAN", "NULL"]
    )

    print("\nLabel availability:")
    print("Rows with a label:", int(valid_labels.sum()))
    print("Rows without a label:", int((~valid_labels).sum()))

    print("\nSubtype counts:")
    print(labels[valid_labels].value_counts())

    # 5. Read the expression header without loading the full matrix.
    print("\nReading expression matrix header...")
    expression_header = pd.read_csv(
        EXPRESSION_FILE,
        sep="\t",
        nrows=0
    )

    expression_columns = list(expression_header.columns)

    print("Expression columns:", len(expression_columns))
    print("First five column names:", expression_columns[:5])

    # 6. Match clinical sample IDs to expression sample IDs.
    clinical_ids = (
        clinical.loc[valid_labels, SAMPLE_KEY]
        .astype("string")
        .str.strip()
        .dropna()
    )

    expression_sample_ids = set(expression_columns[1:])
    clinical_sample_ids = set(clinical_ids)

    matched_ids = clinical_sample_ids.intersection(
        expression_sample_ids
    )

    print("\nSample matching:")
    print("Unique clinical sample IDs with labels:", len(clinical_sample_ids))
    print("Unique expression sample IDs:", len(expression_sample_ids))
    print("Matched sample IDs:", len(matched_ids))

    # 7. Summarize what was verified.
    if len(matched_ids) == 0:
        print(
            "\nWARNING: No matching IDs were found. "
            "Do not train a model yet. Inspect sample ID formats."
        )
    else:
        print(
            "\nSample IDs overlap. Next, inspect duplicate IDs, "
            "label consistency, and sample-level alignment."
        )

    print("\nDay 1 preliminary verification complete.")
    print("No model has been trained.")


if __name__ == "__main__":
    main()
