import json
from datetime import datetime, timezone
from pathlib import Path

from oman_data.run import run_dataset

NOW = datetime(2026, 7, 31, tzinfo=timezone.utc)

CFG_YAML = """\
id: fake
title_ar: "وهمي"
title_en: "Fake"
source_name: "test"
source_url: "https://example.com/"
license: "test"
cadence: monthly
columns:
  - {name: month, dtype: str}
  - {name: value, dtype: float, min: 0}
"""

FETCH_PY = """\
from pathlib import Path
def fetch(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / "raw.csv"
    p.write_text("month,value\\n2026-06,{V}\\n", encoding="utf-8")
    return p
"""

PARSE_PY = """\
from pathlib import Path
import pandas as pd
def parse(raw_path: Path):
    df = pd.read_csv(raw_path, encoding="utf-8")
    return df, str(df["month"].max())
"""


def make_repo(tmp_path: Path, value: str = "1.5") -> Path:
    p = tmp_path / "pipelines" / "fake"
    p.mkdir(parents=True)
    (p / "dataset.yaml").write_text(CFG_YAML, encoding="utf-8")
    (p / "fetch.py").write_text(FETCH_PY.replace("{V}", value), encoding="utf-8")
    (p / "parse.py").write_text(PARSE_PY, encoding="utf-8")
    return tmp_path


def test_happy_path_publishes_and_builds_api(tmp_path):
    repo = make_repo(tmp_path)
    assert run_dataset("fake", repo, NOW) is True
    assert (repo / "data" / "fake" / "fake.csv").exists()
    assert (repo / "raw" / "fake" / "2026-07-31" / "raw.csv").exists()
    catalog = json.loads((repo / "api" / "v1" / "datasets.json").read_text(encoding="utf-8"))
    assert catalog["datasets"][0]["id"] == "fake"


def test_validation_failure_publishes_nothing(tmp_path):
    repo = make_repo(tmp_path, value="-5")  # below min: 0
    assert run_dataset("fake", repo, NOW) is False
    assert not (repo / "data" / "fake").exists()


def test_failure_preserves_last_good(tmp_path):
    repo = make_repo(tmp_path, value="1.5")
    assert run_dataset("fake", repo, NOW) is True
    (repo / "pipelines" / "fake" / "fetch.py").write_text(
        FETCH_PY.replace("{V}", "-5"), encoding="utf-8")
    assert run_dataset("fake", repo, NOW) is False
    df_csv = (repo / "data" / "fake" / "fake.csv").read_text(encoding="utf-8")
    assert "1.5" in df_csv  # last-good untouched
