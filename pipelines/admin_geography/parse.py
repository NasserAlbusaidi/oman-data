import shutil
from pathlib import Path

import pandas as pd

AS_OF = "2026-07"  # curation date of names.csv; bump when names change
NAMES_CSV = Path(__file__).parent / "names.csv"
BOUNDARIES_OUT = Path("data/admin_geography/boundaries.geojson")


def parse(raw_path: Path):
    df = pd.read_csv(NAMES_CSV, encoding="utf-8")
    # publish the boundary artifact alongside the table
    BOUNDARIES_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_path, BOUNDARIES_OUT)
    return df, AS_OF
