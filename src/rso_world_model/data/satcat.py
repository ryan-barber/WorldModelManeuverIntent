from __future__ import annotations

import csv
from pathlib import Path

import requests

from rso_world_model.data.schemas import SatelliteMetadata


SATCAT_URL = "https://celestrak.org/pub/satcat.csv"


def download_satcat_csv(output_path: str | Path, timeout_s: float = 30.0) -> Path:
    response = requests.get(SATCAT_URL, timeout=timeout_s)
    response.raise_for_status()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(response.text, encoding="utf-8")
    return target


def load_satcat_metadata(path: str | Path) -> dict[int, SatelliteMetadata]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            int(row["NORAD_CAT_ID"]): SatelliteMetadata.from_satcat_row(row)
            for row in reader
            if row.get("NORAD_CAT_ID")
        }

