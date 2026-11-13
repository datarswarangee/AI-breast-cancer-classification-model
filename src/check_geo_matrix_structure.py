
from pathlib import Path
import gzip

DATA_DIR = Path("data/external/GSE115577")

for path in sorted(DATA_DIR.glob("*.txt.gz")):
    print("\n" + "=" * 70)
    print("FILE:", path.name)

    in_table = False
    sample_count = 0
    row_count = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()

            if not in_table:
                if stripped.lower().startswith("!series_matrix_table_begin"):
                    in_table = True
                    header = next(f, "").rstrip("\r\n")
                    print("Header:", header[:300])
                    sample_count = max(0, len(header.split("\t")) - 1)
                continue

            if stripped.lower().startswith("!series_matrix_table_end"):
                break

            if not stripped:
                continue

            row_count += 1
            if row_count <= 5:
                print(f"Row {row_count}:", stripped[:500])

    print("Expression rows / probes:", row_count)
    print("Sample columns:", sample_count)

    if not in_table:
        print("WARNING: Expression table start marker was not found.")
