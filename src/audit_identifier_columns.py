
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

clinical_path = ROOT / "data/raw/TCGA_BRCA_clinicalMatrix_deidentified.tsv"
expression_path = ROOT / "data/raw/TCGA.BRCA.sampleMap_HiSeqV2.tsv"
output_path = ROOT / "reports/identifier_column_comparison.csv"

label_col = "PAM50Call_RNAseq"

clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
expression_header = pd.read_csv(expression_path, sep="\t", nrows=0)

expression_ids = set(
    str(c).strip() for c in expression_header.columns[1:]
)

candidate_columns = [
    "sampleID",
    "sample",
    "_patient",
    "bcr_patient_barcode",
    "bcr_sample_barcode",
    "_GENOMIC_ID_TCGA_BRCA_exp_HiSeqV2",
    "_GENOMIC_ID_TCGA_BRCA_PDMarray",
]

results = []

for col in candidate_columns:
    if col not in clinical.columns:
        results.append({
            "column": col,
            "exists": False,
            "non_missing": 0,
            "unique_values": 0,
            "overlap_with_expression": 0,
            "labelled_rows_matching_expression": 0,
            "duplicate_rows": 0,
        })
        continue

    values = clinical[col].astype("string").str.strip()
    valid = values.notna() & values.ne("") & values.ne("NA")
    non_missing_values = values[valid]

    matching_rows = clinical.loc[
        valid & values.isin(expression_ids)
    ]

    labelled_matching = matching_rows[
        matching_rows[label_col].isin(
            ["Basal", "Her2", "LumA", "LumB", "Normal"]
        )
    ]

    results.append({
        "column": col,
        "exists": True,
        "non_missing": int(valid.sum()),
        "unique_values": int(non_missing_values.nunique()),
        "overlap_with_expression": int(
            non_missing_values.isin(expression_ids).sum()
        ),
        "labelled_rows_matching_expression": len(labelled_matching),
        "duplicate_rows": int(
            non_missing_values.duplicated(keep=False).sum()
        ),
    })

result_df = pd.DataFrame(results)
output_path.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(output_path, index=False)

print(result_df.to_string(index=False))
print(f"\nSaved report: {output_path}")
