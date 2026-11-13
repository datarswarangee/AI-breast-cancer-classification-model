
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CLINICAL_PATH = ROOT / "data/raw/TCGA_BRCA_clinicalMatrix_deidentified.tsv"
EXPRESSION_PATH = ROOT / "data/raw/TCGA.BRCA.sampleMap_HiSeqV2.tsv"

OUTPUT_DIR = ROOT / "data/processed_candidate"
REPORT_DIR = ROOT / "reports/alignment_candidate"

SAMPLE_COL = "sampleID"
LABEL_COL = "PAM50Call_RNAseq"

VALID_LABELS = {"Basal", "Her2", "LumA", "LumB", "Normal"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading clinical metadata...")
    clinical = pd.read_csv(
        CLINICAL_PATH, sep="\t", low_memory=False
    )

    print("2. Loading expression matrix...")
    expression = pd.read_csv(
        EXPRESSION_PATH, sep="\t", index_col=0
    ).T

    expression.index = expression.index.astype(str).str.strip()
    expression.index.name = "sample_id"

    clinical = clinical[[SAMPLE_COL, LABEL_COL]].copy()
    clinical[SAMPLE_COL] = (
        clinical[SAMPLE_COL].astype("string").str.strip()
    )
    clinical[LABEL_COL] = (
        clinical[LABEL_COL].astype("string").str.strip()
    )

    # Remove rows without a sample identifier.
    clinical = clinical.dropna(subset=[SAMPLE_COL])
    clinical = clinical[clinical[SAMPLE_COL] != ""]

    # Verify uniqueness before joining.
    duplicate_ids = clinical[
        clinical[SAMPLE_COL].duplicated(keep=False)
    ]

    if not duplicate_ids.empty:
        duplicate_ids.to_csv(
            REPORT_DIR / "duplicate_sampleID_rows.csv",
            index=False
        )
        raise ValueError(
            "Duplicate sampleID values found. "
            "See reports/alignment_candidate/"
            "duplicate_sampleID_rows.csv"
        )

    expression_ids = set(expression.index)
    clinical_ids = set(clinical[SAMPLE_COL])
    shared_ids = expression_ids & clinical_ids

    print("\n3. Identifier checks")
    print("Expression samples:", len(expression_ids))
    print("Unique clinical sampleIDs:", len(clinical_ids))
    print("Shared sampleIDs:", len(shared_ids))

    # One-to-one join using the documented TCGA sample barcode.
    aligned = (
        expression.loc[expression.index.isin(shared_ids)]
        .join(
            clinical.set_index(SAMPLE_COL),
            how="inner",
            validate="one_to_one"
        )
    )

    # Save a report of all matched samples, including missing labels.
    aligned[["PAM50Call_RNAseq"]].rename(
        columns={"PAM50Call_RNAseq": "subtype"}
    ).to_csv(REPORT_DIR / "all_matched_labels.csv")

    # Keep only recognised PAM50 subtype labels for modelling.
    labelled = aligned[
        aligned["PAM50Call_RNAseq"].isin(VALID_LABELS)
    ].copy()

    X = labelled.drop(columns=["PAM50Call_RNAseq"])
    y = labelled[["PAM50Call_RNAseq"]].rename(
        columns={"PAM50Call_RNAseq": "subtype"}
    )

    assert X.index.equals(y.index)
    assert X.index.is_unique
    assert y.index.is_unique

    print("\n4. Candidate dataset quality")
    print("Matched samples, with or without labels:", len(aligned))
    print("Matched samples with valid subtype labels:", len(labelled))
    print("Genes:", X.shape[1])
    print("Missing expression values:", int(X.isna().sum().sum()))
    print("Missing subtype labels among matched samples:",
          int(aligned["PAM50Call_RNAseq"].isna().sum()))

    print("\nSubtype distribution:")
    print(y["subtype"].value_counts().to_string())

    X.to_csv(OUTPUT_DIR / "expression_matrix.csv")
    y.to_csv(OUTPUT_DIR / "subtype_labels.csv")

    manifest = pd.DataFrame({
        "sample_id": X.index,
        "subtype": y["subtype"].values
    })
    manifest.to_csv(OUTPUT_DIR / "sample_manifest.csv", index=False)

    y["subtype"].value_counts().rename_axis(
        "subtype"
    ).reset_index(name="count").to_csv(
        REPORT_DIR / "subtype_distribution.csv", index=False
    )

    print("\nCandidate dataset saved to:", OUTPUT_DIR)
    print("Existing processed data has not been modified.")
    print("No model was trained.")


if __name__ == "__main__":
    main()
