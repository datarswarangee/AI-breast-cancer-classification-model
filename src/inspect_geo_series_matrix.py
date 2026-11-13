
from pathlib import Path
import gzip

DATA_DIR = Path("data/external/GSE115577")

for path in sorted(DATA_DIR.glob("*.txt.gz")):
    print("\n" + "=" * 75)
    print("FILE:", path.name)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(35):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\r\n"))

    for line in lines:
        # Print platform, sample, and subtype metadata.
        if (
            line.startswith("!Series_title")
            or line.startswith("!Series_summary")
            or line.startswith("!Series_platform_id")
            or line.startswith("!Sample_geo_accession")
            or line.startswith("!Sample_title")
            or line.startswith("!Sample_source_name_ch1")
            or line.startswith("!Sample_characteristics_ch1")
            or line.startswith("!series_matrix_table_begin")
            or line.startswith("!series_matrix_table_end")
        ):
            print(line[:1200])

    # Search the complete compressed file for PAM50 annotations.
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        matches = [
            (number, line.rstrip("\r\n")[:1200])
            for number, line in enumerate(f, start=1)
            if "pam50" in line.lower() or "molecularsubtype" in line.lower()
        ]

    print("\nPAM50-related metadata lines:", len(matches))
    for number, line in matches[:12]:
        print(f"Line {number}: {line}")
