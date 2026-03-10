from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from rso_world_model.data.io import ensure_dir, write_json


SPACE_TRACK_AUTH_URL = "https://www.space-track.org/ajaxauth/login"
SPACE_TRACK_QUERY_ROOT = "https://www.space-track.org/basicspacedata/query"


@dataclass(slots=True)
class SpaceTrackCredentials:
    identity: str
    password: str

    @classmethod
    def from_env(cls) -> "SpaceTrackCredentials":
        identity = os.environ.get("SPACETRACK_ID")
        password = os.environ.get("SPACETRACK_PASSWORD")
        if not identity or not password:
            raise RuntimeError("SPACETRACK_ID and SPACETRACK_PASSWORD must be set.")
        return cls(identity=identity, password=password)


class SpaceTrackClient:
    def __init__(self, credentials: SpaceTrackCredentials, timeout_s: float = 30.0) -> None:
        self.credentials = credentials
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def login(self) -> None:
        response = self.session.post(
            SPACE_TRACK_AUTH_URL,
            data={"identity": self.credentials.identity, "password": self.credentials.password},
            timeout=self.timeout_s,
        )
        response.raise_for_status()

    def fetch_gp_history(self, norad_cat_id: int) -> list[dict]:
        query = (
            f"{SPACE_TRACK_QUERY_ROOT}/class/gp_history/NORAD_CAT_ID/{norad_cat_id}/"
            "orderby/EPOCH asc/format/json"
        )
        response = self.session.get(query, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def download_gp_history_cache(self, norad_cat_ids: Iterable[int], output_dir: str | Path) -> list[Path]:
        ensure_dir(output_dir)
        self.login()
        written_paths: list[Path] = []
        for norad_cat_id in norad_cat_ids:
            payload = self.fetch_gp_history(int(norad_cat_id))
            target = Path(output_dir) / f"{int(norad_cat_id)}.json"
            write_json(target, payload)
            written_paths.append(target)
        return written_paths

