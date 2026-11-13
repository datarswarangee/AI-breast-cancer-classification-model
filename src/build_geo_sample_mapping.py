
import gzip
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "GSE115577"
OUTPUT_DIR = ROOT / "reports" / "external_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = (
    DATA_DIR / "GSE115577_data_1110_25334probes_GEO.csv.gz"
)
PLATFORMS = ["GPL17585", "GPL22920"]


def clean(value):
    return value.strip().strip('"')


def read_expression_names():
    with gzip.open(
        EXPRESSION_FILE, "rt", encoding="utf-8", newline=""
    ) as f:
        header = next(csv.reader(f))
    return [clean(x).removesuffix(".CEL") for x in header[1:]]


def read_platform(platform):
    path = DATA_DIR / f"GSE115577-{platform}_series_matrix.txt.gz"
    fields = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                fields["gsm"] = [clean(x) for x in line.rstrip().split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                fields["title"] = [clean(x) for x in line.rstrip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                values = [clean(x) for x in line.rstrip().split("\t")[1:]]
                if values and "molecularsubtype pam50 modifiedmedianmethod" in values[0].lower():
                    fields["pam50_modified"] = values
                elif values and "molecularsubtype pam50 subgroupspecificmethod" in values[0].lower():
                    fields["pam50_specific"] = values

    if "gsm" not in fields or "title" not in fields:
        raise ValueError(f"Missing sample metadata in {platform}")

    n = len(fields["gsm"])
    if len(fields["title"]) != n:
        raise ValueError(f"Title count mismatch in {platform}")

    records = []
    for i in range(n):
        title = fields["title"][i]
        match = re.search(r"\[([^\]]+)\]", title)
        stem = match.group(1).removesuffix(".CEL").strip() if match else ""

        modified = fields.get("pam50_modified", [""] * n)[i]
        specific = fields.get("pam50_specific", [""] * n)[i]

        records.append({
            "platform": platform,
            "gsm": fields["gsm"][i],
            "title": title,
            "expression_name": stem,
            "pam50_modified": modified,
            "pam50_specific": specific,
        })

    return records


expression_names = read_expression_names()
if len(expression_names) != len(set(expression_names)):
    raise ValueError("Duplicate expression sample names detected")

expression_set = set(expression_names)
records = []
for platform in PLATFORMS:
    records.extend(read_platform(platform))

by_name = {}
for record in records:
    name = record["expression_name"]
    if name in expression_set:
        by_name.setdefault(name, []).append(record)

rows = []
for name in expression_names:
    matches = by_name.get(name, [])

    if len(matches) == 1:
        record = matches[0]
        status = "unique_match"
    elif len(matches) == 0:
        record = {}
        status = "unmatched"
    else:
        record = {}
        status = "ambiguous_match"

    rows.append({
        "expression_name": name,
        "mapping_status": status,
        "platform": record.get("platform", ""),
        "gsm": record.get("gsm", ""),
        "sample_title": record.get("title", ""),
        "pam50_modified": record.get("pam50_modified", ""),
        "pam50_specific": record.get("pam50_specific", ""),
    })

output = OUTPUT_DIR / "GSE115577_sample_mapping.csv"
with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

from collections import Counter
print("Expression samples:", len(expression_names))
print("Mapping statuses:", dict(Counter(r["mapping_status"] for r in rows)))
print("Output:", output)
print("\nUnmatched samples:")
for row in rows:
    if row["mapping_status"] != "unique_match":
        print(row["expression_name"], "->", row["mapping_status"])
