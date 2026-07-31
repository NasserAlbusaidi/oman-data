from pathlib import Path

import pandas as pd

AS_OF = "2026-07"  # curation date of names.csv; bump when names change


def parse(raw_path: Path):
    df = pd.read_csv(raw_path, encoding="utf-8")
    return df, AS_OF
