from pathlib import Path

import requests

ADM2_API = "https://www.geoboundaries.org/api/current/gbOpen/OMN/ADM2/"


def fetch(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta = requests.get(ADM2_API, timeout=60)
    meta.raise_for_status()
    gj_url = meta.json()["gjDownloadURL"]
    gj = requests.get(gj_url, timeout=120)
    gj.raise_for_status()
    out = raw_dir / "omn_adm2.geojson"
    out.write_bytes(gj.content)
    return out
