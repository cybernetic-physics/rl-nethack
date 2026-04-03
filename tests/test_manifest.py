"""
Tests for Manifest module.

Tests cover:
- hash_file on a known file
- build_manifest returns dict with all required keys
- verify_manifest returns valid=True for a correct manifest
- verify_manifest returns valid=False if you tamper with a field
- save/load roundtrip preserves content
- improvement is correctly computed as post - baseline
- Test with real files (write temp files, hash them)
"""

import json
import os
import sys
import tempfile

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.manifest import (
    build_manifest,
    hash_file,
    load_manifest,
    save_manifest,
    verify_manifest,
)


# ===========================================================================
# hash_file tests
# ===========================================================================

class TestHashFile:
    """Test hash_file on known content."""

    def test_known_content(self, tmp_path):
        """SHA256 of a file with known content."""
        f = tmp_path / "hello.txt"
        content = b"hello world\n"
        f.write_bytes(content)
        # Pre-computed: echo -n 'hello world\n' | sha256sum  won't work,
        # use Python directly for reference.
        import hashlib
        expected = hashlib.sha256(content).hexdigest()
        assert hash_file(str(f)) == expected

    def test_empty_file(self, tmp_path):
        """SHA256 of an empty file."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        import hashlib
        expected = hashlib.sha256(b"").hexdigest()
        assert hash_file(str(f)) == expected

    def test_returns_hex_string(self, tmp_path):
        """Result should be a 64-char lowercase hex string."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"some data")
        result = hash_file(str(f))
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_nonexistent_file_raises(self):
        """hash_file should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            hash_file("/tmp/this_file_does_not_exist_manifest_test_xyz.json")


# ===========================================================================
# build_manifest tests
# ===========================================================================

class TestBuildManifest:
    """Test build_manifest returns dict with all required keys."""

    def _make_temp_training_data(self, tmp_path):
        """Create a temp training data file and return its path."""
        data_path = tmp_path / "train.jsonl"
        lines = ['{"input": "a", "output": "b"}', '{"input": "c", "output": "d"}']
        data_path.write_text("\n".join(lines))
        return str(data_path)

    def _make_temp_adapter_dir(self, tmp_path):
        """Create a temp adapter dir with a safetensors file."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        sf = adapter_dir / "adapter_model.safetensors"
        sf.write_bytes(b"\x00" * 128)
        return str(adapter_dir)

    def test_returns_dict_with_all_required_keys(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"exact_match_rate": 0.3, "pos_accuracy": 0.5},
            post_training_scores={"exact_match_rate": 0.6, "pos_accuracy": 0.8},
        )

        # Top-level keys
        assert "version" in manifest
        assert manifest["version"] == "1.0"
        assert "base_model" in manifest
        assert "training_data" in manifest
        assert "training_config" in manifest
        assert "results" in manifest
        assert "adapter" in manifest
        assert "manifest_hash" in manifest

    def test_base_model_section(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model="my-model-v1",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        assert manifest["base_model"]["name"] == "my-model-v1"

    def test_base_model_sha256_when_config_exists(self, tmp_path):
        """If base_model dir has a config.json, include its hash."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"vocab_size": 32000}')

        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model=str(model_dir),
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        assert "sha256" in manifest["base_model"]
        assert len(manifest["base_model"]["sha256"]) == 64

    def test_training_data_section(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        td = manifest["training_data"]
        assert td["path"] == data_path
        assert "sha256" in td
        assert td["num_lines"] == 2  # 2 JSONL lines

    def test_training_data_num_lines_single_line(self, tmp_path):
        """Single line file (no trailing newline) = 1 line."""
        data_path = tmp_path / "one.jsonl"
        data_path.write_text('{"a": 1}')
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model="m",
            training_data_path=str(data_path),
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        assert manifest["training_data"]["num_lines"] == 1

    def test_adapter_section_with_safetensors(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        adapter = manifest["adapter"]
        assert adapter["path"] == adapter_path
        assert "sha256" in adapter
        assert adapter["size_bytes"] == 128

    def test_adapter_section_no_safetensors(self, tmp_path):
        """If no safetensors file, sha256 and size_bytes should be absent."""
        data_path = self._make_temp_training_data(tmp_path)
        adapter_dir = tmp_path / "adapter_empty"
        adapter_dir.mkdir()

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=str(adapter_dir),
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        adapter = manifest["adapter"]
        assert "sha256" not in adapter
        assert "size_bytes" not in adapter

    def test_training_config_passed_through(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)
        config = {"lr": 0.001, "epochs": 3, "lora_rank": 8}

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
            training_config=config,
        )
        assert manifest["training_config"] == config
        assert manifest["training_config"]["lr"] == 0.001

    def test_training_config_default_empty(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores={"a": 1},
            post_training_scores={"a": 2},
        )
        assert manifest["training_config"] == {}

    def test_improvement_computed(self, tmp_path):
        """Improvement should be post - baseline for each metric."""
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        baseline = {"exact_match_rate": 0.3, "pos_accuracy": 0.5}
        post = {"exact_match_rate": 0.6, "pos_accuracy": 0.8}

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores=baseline,
            post_training_scores=post,
        )
        improvement = manifest["results"]["improvement"]
        assert improvement["exact_match_rate"] == pytest.approx(0.3)
        assert improvement["pos_accuracy"] == pytest.approx(0.3)

    def test_improvement_negative(self, tmp_path):
        """Improvement can be negative."""
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        baseline = {"score": 0.8}
        post = {"score": 0.5}

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores=baseline,
            post_training_scores=post,
        )
        assert manifest["results"]["improvement"]["score"] == pytest.approx(-0.3)

    def test_improvement_with_extra_keys_in_post(self, tmp_path):
        """Keys only in post_training_scores use 0 as baseline."""
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        baseline = {"score_a": 0.3}
        post = {"score_a": 0.6, "score_b": 0.4}

        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores=baseline,
            post_training_scores=post,
        )
        imp = manifest["results"]["improvement"]
        assert imp["score_a"] == pytest.approx(0.3)
        assert imp["score_b"] == pytest.approx(0.4)

    def test_results_section_structure(self, tmp_path):
        data_path = self._make_temp_training_data(tmp_path)
        adapter_path = self._make_temp_adapter_dir(tmp_path)

        baseline = {"x": 1}
        post = {"x": 2}
        manifest = build_manifest(
            base_model="test-model",
            training_data_path=data_path,
            adapter_path=adapter_path,
            baseline_scores=baseline,
            post_training_scores=post,
        )
        results = manifest["results"]
        assert results["baseline_scores"] == baseline
        assert results["post_training_scores"] == post
        assert "improvement" in results


# ===========================================================================
# verify_manifest tests
# ===========================================================================

class TestVerifyManifest:
    """Test verify_manifest self-hash verification."""

    def _make_manifest(self, tmp_path):
        """Helper: build a valid manifest using temp files."""
        data_path = tmp_path / "train.jsonl"
        data_path.write_text('{"a": 1}\n{"b": 2}\n')
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 64)

        return build_manifest(
            base_model="test-model",
            training_data_path=str(data_path),
            adapter_path=str(adapter_dir),
            baseline_scores={"score": 0.3},
            post_training_scores={"score": 0.7},
        )

    def test_valid_manifest(self, tmp_path):
        """verify_manifest returns valid=True for an untampered manifest."""
        manifest = self._make_manifest(tmp_path)
        result = verify_manifest(manifest)
        assert result["valid"] is True
        assert result["computed_hash"] == result["stored_hash"]

    def test_tampered_field_invalid(self, tmp_path):
        """verify_manifest returns valid=False if a field is changed."""
        manifest = self._make_manifest(tmp_path)
        manifest["version"] = "tampered"
        result = verify_manifest(manifest)
        assert result["valid"] is False
        assert result["computed_hash"] != result["stored_hash"]

    def test_tampered_scores_invalid(self, tmp_path):
        """Tampering with scores should invalidate."""
        manifest = self._make_manifest(tmp_path)
        manifest["results"]["baseline_scores"]["score"] = 999
        result = verify_manifest(manifest)
        assert result["valid"] is False

    def test_tampered_improvement_invalid(self, tmp_path):
        """Tampering with improvement should invalidate."""
        manifest = self._make_manifest(tmp_path)
        manifest["results"]["improvement"]["score"] = 999
        result = verify_manifest(manifest)
        assert result["valid"] is False

    def test_missing_manifest_hash(self, tmp_path):
        """A manifest with no manifest_hash should be invalid."""
        manifest = self._make_manifest(tmp_path)
        del manifest["manifest_hash"]
        result = verify_manifest(manifest)
        assert result["valid"] is False
        assert result["stored_hash"] == ""

    def test_verify_returns_expected_keys(self, tmp_path):
        """verify_manifest should return valid, computed_hash, stored_hash."""
        manifest = self._make_manifest(tmp_path)
        result = verify_manifest(manifest)
        assert "valid" in result
        assert "computed_hash" in result
        assert "stored_hash" in result


# ===========================================================================
# save/load roundtrip tests
# ===========================================================================

class TestSaveLoadRoundtrip:
    """Test save_manifest and load_manifest roundtrip."""

    def _make_manifest(self, tmp_path):
        data_path = tmp_path / "train.jsonl"
        data_path.write_text('{"a": 1}\n')
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        return build_manifest(
            base_model="test-model",
            training_data_path=str(data_path),
            adapter_path=str(adapter_dir),
            baseline_scores={"x": 0.1},
            post_training_scores={"x": 0.9},
        )

    def test_roundtrip_preserves_content(self, tmp_path):
        """save then load should produce identical dict."""
        manifest = self._make_manifest(tmp_path)
        out_path = str(tmp_path / "manifest.json")
        save_manifest(manifest, out_path)
        loaded = load_manifest(out_path)
        assert loaded == manifest

    def test_saved_file_is_valid_json(self, tmp_path):
        """The saved file should be parseable JSON."""
        manifest = self._make_manifest(tmp_path)
        out_path = str(tmp_path / "manifest.json")
        save_manifest(manifest, out_path)

        with open(out_path) as f:
            parsed = json.load(f)
        assert parsed == manifest

    def test_saved_file_is_pretty_printed(self, tmp_path):
        """Saved JSON should use indent=2."""
        manifest = self._make_manifest(tmp_path)
        out_path = str(tmp_path / "manifest.json")
        save_manifest(manifest, out_path)

        text = (tmp_path / "manifest.json").read_text()
        # Pretty-printed should have newlines and indentation
        assert "\n" in text
        assert '  "' in text  # 2-space indent

    def test_load_nonexistent_raises(self):
        """load_manifest should raise for missing file."""
        with pytest.raises(FileNotFoundError):
            load_manifest("/tmp/no_such_manifest_file_xyz.json")

    def test_roundtrip_still_verifies(self, tmp_path):
        """After save/load, verify_manifest should still say valid."""
        manifest = self._make_manifest(tmp_path)
        out_path = str(tmp_path / "manifest.json")
        save_manifest(manifest, out_path)
        loaded = load_manifest(out_path)
        result = verify_manifest(loaded)
        assert result["valid"] is True


# ===========================================================================
# Integration: build + verify + tamper + reload
# ===========================================================================

class TestIntegration:
    """End-to-end scenarios."""

    def test_full_pipeline(self, tmp_path):
        """Build manifest, save, load, verify -> valid."""
        # Setup files
        data_path = tmp_path / "data.jsonl"
        data_path.write_text('{"in": "x"}\n{"in": "y"}\n{"in": "z"}\n')
        adapter_dir = tmp_path / "lora_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"ADAPTER_WEIGHTS")

        manifest = build_manifest(
            base_model="llama-3-8b",
            training_data_path=str(data_path),
            adapter_path=str(adapter_dir),
            baseline_scores={"exact_match_rate": 0.25, "pos_accuracy": 0.4},
            post_training_scores={"exact_match_rate": 0.55, "pos_accuracy": 0.7},
            training_config={"lr": 2e-4, "epochs": 3},
        )

        # Check structure
        assert manifest["version"] == "1.0"
        assert manifest["base_model"]["name"] == "llama-3-8b"
        assert manifest["training_data"]["num_lines"] == 3
        assert manifest["results"]["improvement"]["exact_match_rate"] == pytest.approx(0.3)
        assert manifest["adapter"]["size_bytes"] == len(b"ADAPTER_WEIGHTS")

        # Save & reload
        out = str(tmp_path / "manifest.json")
        save_manifest(manifest, out)
        loaded = load_manifest(out)
        assert loaded == manifest

        # Verify
        result = verify_manifest(loaded)
        assert result["valid"] is True

    def test_tamper_after_save_invalidates(self, tmp_path):
        """Modify file on disk -> load -> verify should fail."""
        data_path = tmp_path / "data.jsonl"
        data_path.write_text('{"a": 1}\n')
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        manifest = build_manifest(
            base_model="m",
            training_data_path=str(data_path),
            adapter_path=str(adapter_dir),
            baseline_scores={"s": 0.0},
            post_training_scores={"s": 1.0},
        )

        out = str(tmp_path / "manifest.json")
        save_manifest(manifest, out)

        # Tamper with the file on disk
        text = (tmp_path / "manifest.json").read_text()
        text = text.replace('"m"', '"tampered"')
        (tmp_path / "manifest.json").write_text(text)

        loaded = load_manifest(out)
        result = verify_manifest(loaded)
        assert result["valid"] is False
