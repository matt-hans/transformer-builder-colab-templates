"""
Gist loader with revision pinning and checksum helper.

In restricted environments, fetch falls back gracefully and returns metadata
with at least the parsed gist_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import hashlib
import json
import re


@dataclass
class GistMetadata:
    gist_id: str
    revision: Optional[str]
    file_names: List[str]
    sha256: Optional[str]
    owner: Optional[str]


_GIST_ID_RE = re.compile(r"([0-9a-f]{8,32})$", re.IGNORECASE)


def _parse_gist_id(gist_url_or_id: str) -> str:
    s = gist_url_or_id.strip().rstrip('/')
    # Typical forms: https://gist.github.com/user/<id> or just <id>
    m = _GIST_ID_RE.search(s)
    if not m:
        raise ValueError(f"Could not parse gist id from: {gist_url_or_id}")
    return m.group(1)


def _compute_dir_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob('**/*')):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def load_gist_model(gist_url_or_id: str, revision: str | None = None, download_dir: str = "./external/gists") -> GistMetadata:
    gid = _parse_gist_id(gist_url_or_id)
    out_base = Path(download_dir) / gid / (revision or 'latest')
    out_base.mkdir(parents=True, exist_ok=True)

    owner = None
    file_names: List[str] = []

    # Try network fetch; degrade gracefully on failure
    try:
        import requests  # type: ignore
        url = f"https://api.github.com/gists/{gid}"
        if revision:
            url += f"/{revision}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        owner = (data.get('owner') or {}).get('login')
        files = data.get('files') or {}
        for name, meta in files.items():
            raw_url = meta.get('raw_url')
            if not raw_url:
                continue
            r = requests.get(raw_url, timeout=10)
            r.raise_for_status()
            (out_base / name).write_bytes(r.content)
            file_names.append(name)
    except Exception:
        # No network; leave directory empty and proceed
        pass

    sha256 = _compute_dir_sha256(out_base) if any(out_base.iterdir()) else None
    return GistMetadata(gist_id=gid, revision=revision, file_names=file_names, sha256=sha256, owner=owner)

