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

# Same parse plus the runner's optional post-validation hook. It records that it
# ran, next to the pipeline, so a test can tell "never called" from "called".
PARSE_WITH_PERSIST_PY = PARSE_PY + """
CURATED = Path(__file__).parent / "curated.txt"
def persist(df):
    CURATED.write_text(",".join(map(str, df["value"])), encoding="utf-8")
    return list(df["month"])
"""


def make_repo(tmp_path: Path, value: str = "1.5", parse_py: str = PARSE_PY) -> Path:
    p = tmp_path / "pipelines" / "fake"
    p.mkdir(parents=True)
    (p / "dataset.yaml").write_text(CFG_YAML, encoding="utf-8")
    (p / "fetch.py").write_text(FETCH_PY.replace("{V}", value), encoding="utf-8")
    (p / "parse.py").write_text(parse_py, encoding="utf-8")
    return tmp_path


def curated(repo: Path) -> Path:
    return repo / "pipelines" / "fake" / "curated.txt"


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


def test_persist_hook_runs_only_after_validation_passes(tmp_path):
    """The hook writes a pipeline's own curated input — a file no source can
    regenerate. It must run on the happy path...
    """
    repo = make_repo(tmp_path, parse_py=PARSE_WITH_PERSIST_PY)
    assert run_dataset("fake", repo, NOW) is True
    assert curated(repo).read_text(encoding="utf-8") == "1.5"


def test_validation_failure_never_invokes_the_persist_hook(tmp_path):
    """...and must NOT run when validation rejects the frame.

    Otherwise a bad reading becomes the curated truth every later run
    cross-checks against, while the runner prints "last-good preserved" — the
    published data would indeed be untouched, but the pipeline's own history
    would already be poisoned, and the *correct* later reading is then the one
    that gets rejected for disagreeing with it.
    """
    repo = make_repo(tmp_path, value="-5", parse_py=PARSE_WITH_PERSIST_PY)
    assert run_dataset("fake", repo, NOW) is False
    assert not curated(repo).exists(), "persist ran on a validation-failed run"


def test_persist_hook_is_optional(tmp_path):
    """Pipelines with nothing to curate omit it; the runner must not care."""
    repo = make_repo(tmp_path)  # PARSE_PY defines no persist
    assert run_dataset("fake", repo, NOW) is True


def test_failure_preserves_last_good(tmp_path):
    repo = make_repo(tmp_path, value="1.5")
    assert run_dataset("fake", repo, NOW) is True
    (repo / "pipelines" / "fake" / "fetch.py").write_text(
        FETCH_PY.replace("{V}", "-5"), encoding="utf-8")
    assert run_dataset("fake", repo, NOW) is False
    df_csv = (repo / "data" / "fake" / "fake.csv").read_text(encoding="utf-8")
    assert "1.5" in df_csv  # last-good untouched
