# oman-data

The unofficial developer layer for Oman's open data. Official published
datasets, normalized into tidy bilingual (AR/EN) tables, versioned in git,
served as a static JSON API.

## Use it

Raw JSON (static, CORS-open once hosted; for now served from this repo):

    curl https://raw.githubusercontent.com/NasserAlbusaidi/oman-data/main/api/v1/datasets.json
    curl https://raw.githubusercontent.com/NasserAlbusaidi/oman-data/main/api/v1/cpi/latest.json

Files: every dataset lives in `data/<id>/` as CSV + Parquet with `meta.json`
and a changelog. Every raw source snapshot is archived under `raw/<id>/<date>/`.

## Guarantees

- Bilingual: every dataset carries `title_ar` and `title_en`; Arabic is stored
  unescaped.
- Traceable: JSON → git commit → raw snapshot → official source URL.
- Honest freshness: a dataset past its update window is flagged `"stale": true`,
  never silently outdated.
- No partial publishes: validation failures leave last-good data untouched.

## Add a dataset

`pipelines/<id>/` needs three files: `dataset.yaml` (bilingual metadata +
column schema), `fetch.py` (`fetch(raw_dir) -> Path`), `parse.py`
(`parse(raw_path) -> (DataFrame, as_of)`). Run
`python -m oman_data.run <id>`. Validation gates publication.

Code: MIT. Data: license of each official source (see each dataset's
`dataset.yaml`). Not affiliated with any government entity.
