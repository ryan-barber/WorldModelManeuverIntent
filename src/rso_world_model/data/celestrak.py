from __future__ import annotations

from pathlib import Path
from typing import Iterable

import requests

from rso_world_model.data.io import ensure_dir, write_json


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"


def fetch_group_gp(group: str, timeout_s: float = 30.0) -> list[dict]:
    response = requests.get(
        CELESTRAK_GP_URL,
        params={"GROUP": group, "FORMAT": "json"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def download_group_gp(groups: Iterable[str], output_dir: str | Path) -> list[Path]:
    output = ensure_dir(output_dir)
    written_paths: list[Path] = []
    for group in groups:
        payload = fetch_group_gp(group)
        target = output / f"{group}.json"
        write_json(target, payload)
        written_paths.append(target)
    return written_paths

