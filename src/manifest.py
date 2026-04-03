"""
Manifest: Build attested manifest from training artifacts.

Provides:
1. hash_file          -- SHA256 hash of a file
2. build_manifest     -- Compile all hashes and metrics into a JSON manifest
3. save_manifest      -- Write manifest as pretty-printed JSON
4. load_manifest      -- Load and return manifest
5. verify_manifest    -- Verify the manifest's self-hash is correct
"""

import hashlib
import json
import os
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 1. hash_file
# ---------------------------------------------------------------------------

def hash_file(path: str) -> str:
    """Return the SHA256 hex digest of a file.

    Args:
        path: Filesystem path to the file.

    Returns:
        Lowercase hex string of the SHA256 digest.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)  # 64 KiB
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Internal: compute manifest self-hash
# ---------------------------------------------------------------------------

def _compute_manifest_hash(manifest: dict) -> str:
    """Compute SHA256 of the manifest content, excluding manifest_hash itself."""
    copy = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    raw = json.dumps(copy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 2. build_manifest
# ---------------------------------------------------------------------------

def build_manifest(
    base_model: str,
    training_data_path: str,
    adapter_path: str,
    baseline_scores: dict,
    post_training_scores: dict,
    training_config: Optional[dict] = None,
) -> dict:
    """Compile all hashes and metrics into a JSON manifest.

    Args:
        base_model: Name or path of the base model.
        training_data_path: Path to the training data file (JSONL / JSON).
        adapter_path: Path to the adapter directory (LoRA output).
        baseline_scores: Dict of metric scores before training.
        post_training_scores: Dict of metric scores after training.
        training_config: Optional training configuration dict.

    Returns:
        Complete manifest dict with manifest_hash.
    """
    # -- base_model section --
    model_section: Dict = {"name": base_model}
    config_path = os.path.join(base_model, "config.json")
    if os.path.isfile(config_path):
        model_section["sha256"] = hash_file(config_path)

    # -- training_data section --
    training_data_section: Dict = {
        "path": training_data_path,
        "sha256": hash_file(training_data_path),
        "num_lines": sum(1 for _ in open(training_data_path, "rb")),
    }

    # -- adapter section --
    adapter_section: Dict = {"path": adapter_path}
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(safetensors_path):
        adapter_section["sha256"] = hash_file(safetensors_path)
        adapter_section["size_bytes"] = os.path.getsize(safetensors_path)

    # -- improvement: element-wise delta (post - baseline) --
    improvement = {}
    all_keys = set(baseline_scores.keys()) | set(post_training_scores.keys())
    for key in all_keys:
        b = baseline_scores.get(key, 0)
        a = post_training_scores.get(key, 0)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            improvement[key] = a - b

    # -- assemble manifest (without hash first) --
    manifest = {
        "version": "1.0",
        "base_model": model_section,
        "training_data": training_data_section,
        "training_config": training_config if training_config is not None else {},
        "results": {
            "baseline_scores": baseline_scores,
            "post_training_scores": post_training_scores,
            "improvement": improvement,
        },
        "adapter": adapter_section,
    }

    # -- self-hash --
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)

    return manifest


# ---------------------------------------------------------------------------
# 3. save_manifest
# ---------------------------------------------------------------------------

def save_manifest(manifest: dict, path: str) -> None:
    """Write manifest as pretty-printed JSON.

    Args:
        manifest: The manifest dict (should include manifest_hash).
        path: Destination file path.
    """
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# 4. load_manifest
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    """Load and return a manifest from a JSON file.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        Parsed manifest dict.
    """
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 5. verify_manifest
# ---------------------------------------------------------------------------

def verify_manifest(manifest: dict) -> dict:
    """Verify the manifest's self-hash is correct.

    Args:
        manifest: The manifest dict.

    Returns:
        Dict with:
            valid: True if the stored hash matches the computed hash.
            computed_hash: The hash we computed.
            stored_hash: The hash stored in the manifest.
    """
    stored = manifest.get("manifest_hash", "")
    computed = _compute_manifest_hash(manifest)
    return {
        "valid": stored == computed,
        "computed_hash": computed,
        "stored_hash": stored,
    }
