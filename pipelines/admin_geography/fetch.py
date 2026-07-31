import shutil
from pathlib import Path

NAMES_CSV = Path(__file__).parent / "names.csv"

# Boundary geometry is NOT redistributed: geoBoundaries gbOpen OMN ADM2 carries
# boundaryLicense "Other - Direct Permission", which is not ours to re-publish.
# Consumers fetch it themselves from the endpoint recorded in dataset.yaml.


def fetch(raw_dir: Path) -> Path:
    """Archive the curated source table. Static dataset — no network."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "names.csv"
    shutil.copyfile(NAMES_CSV, out)
    return out
