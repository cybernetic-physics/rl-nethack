from __future__ import annotations

import importlib
import subprocess
import sys


APPO_BACKEND_PACKAGES = [
    "sample-factory",
    "faster-fifo",
    "signal-slot-mp",
    "colorlog",
    "tensorboardx",
    "pyglet",
    "opencv-python",
    "absl-py",
    "threadpoolctl",
    "tensorboard",
    "tensorboard-data-server",
    "werkzeug",
    "markdown",
    "wandb",
    "platformdirs",
    "gitpython",
    "gitdb",
    "smmap",
]


def sample_factory_available() -> bool:
    try:
        importlib.import_module("sample_factory")
        return True
    except Exception:
        return False


def ensure_sample_factory_backend() -> bool:
    """Install the APPO backend into the current project venv if needed.

    We intentionally install with `--no-deps` because upstream package metadata
    conflicts with the working `nle==1.2.0` stack even though runtime works.
    """
    if sample_factory_available():
        return False

    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-deps",
        *APPO_BACKEND_PACKAGES,
    ]
    subprocess.run(cmd, check=True)
    importlib.invalidate_caches()
    return True
