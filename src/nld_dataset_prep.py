"""
Helpers for preparing a local NLD / altorg-style dataset root.

These utilities handle:

- extracting one or more local zip archives
- discovering candidate altorg-style dataset roots
- validating that a root looks compatible with `nle.dataset.add_altorg_directory`
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any


def extract_zip_archives(zip_paths: list[str], output_dir: str) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    extracted_archives: list[str] = []
    for zip_path in zip_paths:
        archive_path = Path(zip_path)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_path)
            extracted_files += len(zf.infolist())
        extracted_archives.append(str(archive_path))
    return {
        "output_dir": str(output_path),
        "archives": extracted_archives,
        "extracted_files": extracted_files,
    }


def _contains_any(path: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if any(path.glob(pattern)):
            return True
    return False


def is_altorg_dataset_root(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    has_blacklist = (candidate / "blacklist.txt").exists()
    has_xlog = _contains_any(candidate, ["xlogfile.*", "xlogfile"])
    has_ttyrec = _contains_any(candidate, ["*/*.ttyrec.bz2", "*/*.ttyrec", "*.ttyrec.bz2", "*.ttyrec"])
    return has_blacklist and has_xlog and has_ttyrec


def discover_altorg_roots(search_root: str) -> list[str]:
    root = Path(search_root)
    candidates: list[str] = []
    for path in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if is_altorg_dataset_root(path):
            candidates.append(str(path))
    deduped = sorted(dict.fromkeys(candidates))
    return deduped


def summarize_altorg_root(root_path: str) -> dict[str, Any]:
    root = Path(root_path)
    xlogfiles = sorted(str(p.relative_to(root)) for p in root.glob("xlogfile.*"))
    if not xlogfiles and (root / "xlogfile").exists():
        xlogfiles = ["xlogfile"]
    ttyrecs = list(root.glob("*/*.ttyrec.bz2")) + list(root.glob("*/*.ttyrec"))
    if not ttyrecs:
        ttyrecs = list(root.glob("*.ttyrec.bz2")) + list(root.glob("*.ttyrec"))
    usernames = sorted({p.parent.name for p in ttyrecs if p.parent != root})
    return {
        "root_path": str(root),
        "valid_altorg_root": is_altorg_dataset_root(root),
        "has_blacklist": (root / "blacklist.txt").exists(),
        "xlogfiles": xlogfiles,
        "ttyrec_count": len(ttyrecs),
        "user_count": len(usernames),
        "sample_users": usernames[:10],
    }
