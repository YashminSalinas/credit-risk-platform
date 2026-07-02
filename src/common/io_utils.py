import re
from pathlib import Path
import pandas as pd

BRONZE_PATH = Path("data/bronze")


def _part_number(path: Path) -> int:
    match = re.search(r"part_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_all_accepted():
    files = sorted(
        BRONZE_PATH.glob("accepted_part_*.parquet"),
        key=_part_number
    )

    if not files:
        raise FileNotFoundError("No parquet files found in bronze")

    print(f"Loading {len(files)} bronze partitions...")

    df = pd.concat(
        [pd.read_parquet(f) for f in files],
        ignore_index=True
    )

    print(f"Loaded {len(df):,} rows.")

    return df