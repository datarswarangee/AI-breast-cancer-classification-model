
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CLINICAL_PATH = ROOT / "data/raw/TCGA_BRCA_clinicalMatrix_deidentified.tsv"
EXPRESSION_PATH = ROOT / "data/raw/TCGA.BRCA.sampleMap_HiSeqV2.tsv"
OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports"

BARCODE_COL = "_GENOMIC_ID_TCGA_BRCA_PDMarray"
LABEL_COL = "PAM50Call_RNAseq"

EXPECTED_LABELS = {"LumA", "LumB", "Basal", "Her2", "Normal"}


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
    )

    # The expression matrix must be samples x genes.
    expression = expression.T
    expression.index = expression.index.astype(str).str.strip()
    expression.index.name = "sample_id"

    print("3. Preparing clinical sample identifiers...")
    clinical = clinical[[BARCODE_COL, LABEL_COL]].copy()
    clinical = clinical.dropna(subset=[BARCODE_COL])
    clinical[BARCODE_COL] = (
        clinical[BARCODE_COL].astype(str).str.strip()
    )

    # Exclude duplicate identifiers rather than silently choosing a label.
    duplicate_mask = clinical.duplicated(
        subset=[BARCODE_COL], keep=False
    )
    if duplicate_mask.any():
        clinical.loc[duplicate_mask].to_csv(
            REPORT_DIR / "duplicate_clinical_ids.csv",
            index=False,
        )
        raise ValueError(
            "Duplicate clinical barcodes found; inspect "
            "reports/duplicate_clinical_ids.csv."
        )

    clinical = clinical.rename(
        columns={
            BARCODE_COL: "sample_id",
            LABEL_COL: "subtype",
        }
    )

    print("4. Joining expression samples and labels...")
    expression_ids = set(expression.index)
    clinical_ids = set(clinical["sample_id"])
    shared_ids = expression_ids & clinical_ids

    print(f"Expression samples: {len(expression_ids)}")
    print(f"Clinical barcodes: {len(clinical_ids)}")
    print(f"Shared sample IDs: {len(shared_ids)}")

    aligned = expression.loc[
        expression.index.isin(shared_ids)
    ].join(
        clinical.set_index("sample_id"),
        how="inner",
        validate="one_to_one",
    )

    aligned = aligned[
        aligned["subtype"].isin(EXPECTED_LABELS)
    ].copy()

    labels = aligned[["subtype"]].copy()
    X = aligned.drop(columns=["subtype"])

    # Check the alignment invariant explicitly.
    assert X.index.equals(labels.index)
    assert X.index.is_unique
    assert labels.index.is_unique

    print("\n5. Data quality summary")
    print("Aligned labelled samples:", len(X))
    print("Genes:", X.shape[1])
    print("Missing expression cells:", int(X.isna().sum().sum()))
    print("\nSubtype counts:")
    print(labels["subtype"].value_counts().to_string())

    # Save sample IDs with both output files for auditable alignment.
    X.to_csv(OUTPUT_DIR / "expression_matrix.csv", index=True)
    labels.to_csv(OUTPUT_DIR / "subtype_labels.csv", index=True)

    manifest = pd.DataFrame({
        "sample_id": X.index,
        "subtype": labels["subtype"].values,
    })
    manifest.to_csv(
        OUTPUT_DIR / "sample_manifest.csv", index=False
    )

    labels["subtype"].value_counts().rename_axis(
        "subtype"
    ).reset_index(name="count").to_csv(
        REPORT_DIR / "subtype_distribution.csv",
        index=False,
    )

    print("\nAlignment complete. No model was trained.")


if __name__ == "__main__":
    main()
