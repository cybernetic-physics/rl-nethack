from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import fcntl
import torch


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: str | Path, text: str) -> str:
    target = _ensure_parent(path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return str(target)


def atomic_write_json(path: str | Path, payload: dict) -> str:
    return atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_torch_save(path: str | Path, payload: dict) -> str:
    target = _ensure_parent(path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return str(target)


def atomic_copy_file(src_path: str | Path, dst_path: str | Path) -> str:
    source = Path(src_path)
    target = _ensure_parent(dst_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    os.close(fd)
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return str(target)


@contextmanager
def experiment_lock(lock_path: str | Path):
    target = _ensure_parent(lock_path)
    with open(target, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield str(target)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
