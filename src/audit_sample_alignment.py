
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CLINICAL_PATH = (
    ROOT / "data/raw/TCGA_BRCA_clinicalMatrix_deidentified.tsv"
)
EXPRESSION_PATH = (
    ROOT / "data/raw/TCGA.BRCA.sampleMap_HiSeqV2.tsv"
)
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "sample_alignment_audit.txt"

UUID_COL = "_GENOMIC_ID_TCGA_BRCA_exp_HiSeqV2"
BARCODE_COL = "_GENOMIC_ID_TCGA_BRCA_PDMarray"
LABEL_COL = "PAM50Call_RNAseq"


def clean_ids(series):
    return series.dropna().astype(str).str.strip()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clinical metadata...")
    clinical = pd.read_csv(
        CLINICAL_PATH, sep="\t", low_memory=False
    )

    print("Loading expression sample identifiers...")
    expression_header = pd.read_csv(
        EXPRESSION_PATH, sep="\t", nrows=0
    )
    expression_ids = pd.Index(
        expression_header.columns[1:]
    ).astype(str).str.strip()

    required = [UUID_COL, BARCODE_COL, LABEL_COL]
    missing_columns = [
        col for col in required if col not in clinical.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing clinical columns: {missing_columns}"
        )

    uuid_ids = clean_ids(clinical[UUID_COL])
    barcode_ids = clean_ids(clinical[BARCODE_COL])

    # Keep identifier pairs together for auditing.
    pairs = clinical[[UUID_COL, BARCODE_COL, LABEL_COL]].copy()
    pairs[UUID_COL] = pairs[UUID_COL].astype("string").str.strip()
    pairs[BARCODE_COL] = (
        pairs[BARCODE_COL].astype("string").str.strip()
    )

    uuid_set = set(uuid_ids)
    barcode_set = set(barcode_ids)
    expression_set = set(expression_ids)

    uuid_overlap = uuid_set & expression_set
    barcode_overlap = barcode_set & expression_set

    labelled = pairs[
        pairs[LABEL_COL].isin(
            ["Basal", "Her2", "LumA", "LumB", "Normal"]
        )
    ].copy()

    labelled_uuid_match = labelled[
        labelled[UUID_COL].isin(expression_set)
    ]
    labelled_barcode_match = labelled[
        labelled[BARCODE_COL].isin(expression_set)
    ]

    duplicate_uuid = pairs[UUID_COL].duplicated(keep=False)
    duplicate_barcode = pairs[BARCODE_COL].duplicated(keep=False)

    # Compare the UUID and barcode fields for rows where both exist.
    both_present = pairs.dropna(
        subset=[UUID_COL, BARCODE_COL]
    )
    same_value = (
        both_present[UUID_COL] == both_present[BARCODE_COL]
    )

    lines = [
        "MY MED AI — SAMPLE ALIGNMENT AUDIT",
        "=" * 45,
        f"Clinical rows: {len(clinical)}",
        f"Expression sample columns: {len(expression_ids)}",
        f"Unique expression IDs: {len(expression_set)}",
        "",
        "IDENTIFIER OVERLAP",
        "-" * 25,
        f"Unique UUID-field IDs: {len(uuid_set)}",
        f"UUID IDs matching expression headers: {len(uuid_overlap)}",
        f"Unique barcode-field IDs: {len(barcode_set)}",
        f"Barcode IDs matching expression headers: {len(barcode_overlap)}",
        "",
        "LABELLED ROWS",
        "-" * 25,
        f"Rows with recognised subtype labels: {len(labelled)}",
        f"Labelled rows matching by UUID field: {len(labelled_uuid_match)}",
        f"Labelled rows matching by barcode field: {len(labelled_barcode_match)}",
        "",
        "IDENTIFIER QUALITY",
        "-" * 25,
        f"Rows where both identifiers are present: {len(both_present)}",
        f"Rows where both identifier strings are identical: {int(same_value.sum())}",
        f"Rows involved in duplicate UUID values: {int(duplicate_uuid.sum())}",
        f"Rows involved in duplicate barcode values: {int(duplicate_barcode.sum())}",
        "",
        "UUID EXAMPLES",
        "-" * 25,
        pairs[UUID_COL].dropna().head(5).to_string(index=False),
        "",
        "BARCODE EXAMPLES",
        "-" * 25,
        pairs[BARCODE_COL].dropna().head(5).to_string(index=False),
        "",
        "EXPRESSION HEADER EXAMPLES",
        "-" * 25,
        "\n".join(expression_ids[:10]),
        "",
        "INTERPRETATION",
        "-" * 25,
        "This audit checks identifier overlap and uniqueness only.",
        "A matching identifier string does not by itself establish",
        "that the label and expression profile belong to the same",
        "biological specimen. Verify the identifier mapping against",
        "the source dataset documentation before accepting the join.",
    ]

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
