from pathlib import Path
import pytest
from oman_data.schema import load_dataset_config, ConfigError

VALID_YAML = """\
id: cpi
title_ar: "الرقم القياسي لأسعار المستهلك"
title_en: "Consumer Price Index"
source_name: "NCSI"
source_url: "https://data.gov.om/"
license: "Open Government License - Oman"
cadence: monthly
columns:
  - {name: month, dtype: str}
  - {name: group, dtype: str}
  - {name: index, dtype: float, min: 30, max: 300}
"""

def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "dataset.yaml"
    p.write_text(text, encoding="utf-8")
    return p

def test_loads_valid_config(tmp_path):
    cfg = load_dataset_config(write(tmp_path, VALID_YAML))
    assert cfg.id == "cpi"
    assert cfg.cadence == "monthly"
    assert cfg.columns[2].min == 30
    assert cfg.title_ar.startswith("الرقم")

@pytest.mark.parametrize("field", [
    "id", "title_ar", "title_en", "source_name", "source_url",
    "license", "cadence", "columns",
])
def test_missing_field_raises(tmp_path, field):
    import yaml
    raw = yaml.safe_load(VALID_YAML)
    del raw[field]
    with pytest.raises(ConfigError):
        load_dataset_config(write(tmp_path, yaml.safe_dump(raw, allow_unicode=True)))

def test_bad_cadence_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_dataset_config(write(tmp_path, VALID_YAML.replace("monthly", "hourly")))

def test_bad_dtype_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_dataset_config(write(tmp_path, VALID_YAML.replace("dtype: float", "dtype: decimal")))

def test_empty_title_ar_raises(tmp_path):
    bad = VALID_YAML.replace('title_ar: "الرقم القياسي لأسعار المستهلك"', 'title_ar: ""')
    with pytest.raises(ConfigError):
        load_dataset_config(write(tmp_path, bad))
