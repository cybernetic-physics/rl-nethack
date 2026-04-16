"""
Tests for the training script (train.py).

All tests run WITHOUT GPU. They validate:
- Data format loading and validation
- Conversation formatting function
- Hash file utility
- Argparse defaults
- CLI --help works

No unsloth, trl, peft, or GPU operations are performed.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import (
    build_curriculum_dataset,
    dataset_has_sample_weights,
    filter_dataset_by_metadata,
    format_conversation_text,
    format_dataset_conversations,
    normalize_training_dataset,
    normalize_training_row,
    parse_curriculum_buckets,
    parse_curriculum_stage_repeats,
    parse_metadata_filters,
    parse_args,
    save_training_metadata,
    truncate_dataset,
    weighted_mean_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sharegpt_jsonl(path, num_examples=3):
    """Write a small ShareGPT-format JSONL file for testing."""
    with open(path, "w") as f:
        for i in range(num_examples):
            conversation = {
                "conversations": [
                    {"role": "system", "content": "Predict the outcome of a NetHack action."},
                    {
                        "role": "user",
                        "content": f"HP:10/10 AC:7 Str:15 Dex:14\n"
                                   f"Pos:(3,5) Gold:0 Depth:1 Turn:{i}\n"
                                   f"Adjacent: north=floor south=wall east=door west=wall\n"
                                   f"Monsters: none\n"
                                   f"Items: none\n"
                                   f"Action: north",
                    },
                    {
                        "role": "assistant",
                        "content": f"pos:(0,-1) | hp:same | gold:same | depth:same | alive:yes | msg:",
                    },
                ]
            }
            f.write(json.dumps(conversation) + "\n")


# ===========================================================================
# Test: Training data loads as HF dataset
# ===========================================================================

class TestTrainingDataLoads:
    """Test that JSONL data can be loaded as a HuggingFace Dataset."""

    def test_training_data_loads_as_hf_dataset(self, tmp_path):
        """Generate JSONL, load with datasets.Dataset, verify it works."""
        from datasets import Dataset

        data_path = tmp_path / "train.jsonl"
        _make_sharegpt_jsonl(str(data_path), num_examples=5)

        # Load via from_json (what the HuggingFace datasets library provides)
        dataset = Dataset.from_json(str(data_path))
        assert len(dataset) == 5
        assert "conversations" in dataset.column_names

    def test_conversations_structure(self, tmp_path):
        """Each row should have a conversations list with 3 messages."""
        from datasets import Dataset

        data_path = tmp_path / "train.jsonl"
        _make_sharegpt_jsonl(str(data_path), num_examples=3)

        dataset = Dataset.from_json(str(data_path))
        for row in dataset:
            convs = row["conversations"]
            assert isinstance(convs, list)
            assert len(convs) == 3
            assert convs[0]["role"] == "system"
            assert convs[1]["role"] == "user"
            assert convs[2]["role"] == "assistant"

    def test_from_json_preserves_content(self, tmp_path):
        """Content should be intact after loading."""
        from datasets import Dataset

        data_path = tmp_path / "train.jsonl"
        _make_sharegpt_jsonl(str(data_path), num_examples=2)

        dataset = Dataset.from_json(str(data_path))
        row0 = dataset[0]
        system_msg = row0["conversations"][0]["content"]
        assert "NetHack" in system_msg
        user_msg = row0["conversations"][1]["content"]
        assert "HP:" in user_msg
        assert "Action:" in user_msg

    def test_load_via_train_module(self, tmp_path):
        """Test load_training_data from train.py."""
        from train import load_training_data

        data_path = tmp_path / "train.jsonl"
        _make_sharegpt_jsonl(str(data_path), num_examples=4)

        dataset = load_training_data(str(data_path))
        assert len(dataset) == 4
        assert "conversations" in dataset.column_names

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines in JSONL should be skipped by load_training_data."""
        from train import load_training_data

        data_path = tmp_path / "train.jsonl"
        # Write JSONL with blank lines interspersed
        lines = []
        for i in range(3):
            conversation = {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"prompt {i}"},
                    {"role": "assistant", "content": f"response {i}"},
                ]
            }
            lines.append(json.dumps(conversation))
        content = "\n\n".join(lines) + "\n"
        with open(str(data_path), "w") as f:
            f.write(content)

        dataset = load_training_data(str(data_path))
        assert len(dataset) == 3


class TestMetadataFiltering:
    def test_parse_metadata_filters(self):
        parsed = parse_metadata_filters(["target_context_bucket=256k", "outcome=win"])
        assert parsed == {"target_context_bucket": "256k", "outcome": "win"}

    def test_filter_dataset_by_metadata(self, tmp_path):
        from train import load_training_data

        data_path = tmp_path / "train.jsonl"
        rows = [
            {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "metadata": {"target_context_bucket": "128k", "outcome": "loss"},
            },
            {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ],
                "metadata": {"target_context_bucket": "256k", "outcome": "win"},
            },
        ]
        with open(data_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        dataset = load_training_data(str(data_path))
        filtered = filter_dataset_by_metadata(dataset, {"target_context_bucket": "256k"})
        assert len(filtered) == 1
        assert filtered[0]["metadata"]["outcome"] == "win"

    def test_truncate_dataset(self, tmp_path):
        from train import load_training_data

        data_path = tmp_path / "train.jsonl"
        _make_sharegpt_jsonl(str(data_path), num_examples=5)
        dataset = load_training_data(str(data_path))
        truncated = truncate_dataset(dataset, 2)
        assert len(truncated) == 2

    def test_normalize_training_row_converts_preference_schema(self):
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "state"},
            ],
            "completion": "east",
            "label": False,
        }
        normalized = normalize_training_row(row, positive_weight=1.0, negative_weight=-0.5)
        assert normalized["conversations"][-1] == {"role": "assistant", "content": "east"}
        assert normalized["sample_weight"] == -0.5

    def test_normalize_training_dataset_adds_sample_weights(self, tmp_path):
        from train import load_training_data

        data_path = tmp_path / "pref.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u1"},
                ],
                "completion": "east",
                "label": True,
            },
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u2"},
                ],
                "completion": "wait",
                "label": False,
            },
        ]
        with open(data_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        dataset = load_training_data(str(data_path))
        normalized = normalize_training_dataset(dataset, positive_weight=1.0, negative_weight=-0.25)
        assert dataset_has_sample_weights(normalized) is True
        assert normalized[0]["sample_weight"] == 1.0
        assert normalized[1]["sample_weight"] == -0.25

    def test_weighted_mean_loss_supports_negative_weights(self):
        torch = pytest.importorskip("torch")

        per_example = torch.tensor([2.0, 6.0], dtype=torch.float32)
        weights = torch.tensor([1.0, -0.5], dtype=torch.float32)
        result = weighted_mean_loss(per_example, weights)
        assert round(float(result.item()), 6) == round((2.0 - 3.0) / 1.5, 6)

    def test_parse_curriculum_helpers(self):
        assert parse_curriculum_buckets("128k,256k, 512k") == ["128k", "256k", "512k"]
        assert parse_curriculum_stage_repeats("", 3) == [1, 1, 1]
        assert parse_curriculum_stage_repeats("2,2,1", 3) == [2, 2, 1]
        with pytest.raises(ValueError):
            parse_curriculum_stage_repeats("2,1", 3)

    def test_build_curriculum_dataset_expands_cumulative_stages(self, tmp_path):
        from train import load_training_data

        data_path = tmp_path / "curriculum.jsonl"
        rows = [
            {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ],
                "metadata": {"target_context_bucket": "128k"},
            },
            {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ],
                "metadata": {"target_context_bucket": "256k"},
            },
            {
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u3"},
                    {"role": "assistant", "content": "a3"},
                ],
                "metadata": {"target_context_bucket": "512k"},
            },
        ]
        with open(data_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        dataset = load_training_data(str(data_path))
        curriculum = build_curriculum_dataset(
            dataset,
            ["128k", "256k", "512k"],
            [2, 1, 1],
            metadata_key="target_context_bucket",
        )
        assert len(curriculum) == 7
        buckets = [curriculum[i]["metadata"]["target_context_bucket"] for i in range(len(curriculum))]
        assert buckets == ["128k", "128k", "128k", "256k", "128k", "256k", "512k",]

    def test_parse_args_supports_curriculum_flags(self):
        args = parse_args([
            "--curriculum-buckets", "128k,256k,512k",
            "--curriculum-stage-repeats", "2,2,1",
            "--curriculum-metadata-key", "target_context_bucket",
        ])
        assert args.curriculum_buckets == "128k,256k,512k"
        assert args.curriculum_stage_repeats == "2,2,1"
        assert args.curriculum_metadata_key == "target_context_bucket"


# ===========================================================================
# Test: Format conversation function
# ===========================================================================

class TestFormatConversation:
    """Test the format_conversation_text function."""

    def test_format_conversation_function_basic(self):
        """Verify formatting produces ChatML-style text from ShareGPT."""
        conversation = {
            "conversations": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        text = format_conversation_text(conversation)
        assert "<|im_start|>system" in text
        assert "You are helpful." in text
        assert "<|im_end|>" in text
        assert "<|im_start|>user" in text
        assert "Hello" in text
        assert "<|im_start|>assistant" in text
        assert "Hi there!" in text

    def test_format_preserves_all_content(self):
        """All message content should appear in output."""
        conversation = {
            "conversations": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USR"},
                {"role": "assistant", "content": "AST"},
            ]
        }
        text = format_conversation_text(conversation)
        assert "SYS" in text
        assert "USR" in text
        assert "AST" in text

    def test_format_ordering(self):
        """System message should come before user, user before assistant."""
        conversation = {
            "conversations": [
                {"role": "system", "content": "AAA"},
                {"role": "user", "content": "BBB"},
                {"role": "assistant", "content": "CCC"},
            ]
        }
        text = format_conversation_text(conversation)
        assert text.index("AAA") < text.index("BBB")
        assert text.index("BBB") < text.index("CCC")

    def test_format_with_nle_data(self):
        """Format a realistic NLE game conversation."""
        conversation = {
            "conversations": [
                {
                    "role": "system",
                    "content": "Predict the outcome of a NetHack action.",
                },
                {
                    "role": "user",
                    "content": (
                        "HP:10/10 AC:7 Str:15 Dex:14\n"
                        "Pos:(3,5) Gold:0 Depth:1 Turn:0\n"
                        "Adjacent: north=floor south=wall east=door west=wall\n"
                        "Monsters: none\n"
                        "Items: none\n"
                        "Action: north"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "pos:(0,-1) | hp:same | gold:same | depth:same | alive:yes | msg:",
                },
            ]
        }
        text = format_conversation_text(conversation)
        assert "HP:10/10" in text
        assert "Action: north" in text
        assert "pos:(0,-1)" in text
        assert "alive:yes" in text

    def test_format_ends_with_newline(self):
        """Formatted text should end with a newline."""
        conversation = {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast"},
            ]
        }
        text = format_conversation_text(conversation)
        assert text.endswith("\n")

    def test_format_dataset_conversations_batch(self):
        """Test format_dataset_conversations with a batch."""
        examples = {
            "conversations": [
                [
                    {"role": "system", "content": "sys1"},
                    {"role": "user", "content": "usr1"},
                    {"role": "assistant", "content": "ast1"},
                ],
                [
                    {"role": "system", "content": "sys2"},
                    {"role": "user", "content": "usr2"},
                    {"role": "assistant", "content": "ast2"},
                ],
            ]
        }
        result = format_dataset_conversations(examples)
        assert "text" in result
        assert len(result["text"]) == 2
        assert "usr1" in result["text"][0]
        assert "usr2" in result["text"][1]


# ===========================================================================
# Test: hash_file works
# ===========================================================================

class TestHashFile:
    """Test the hash_file function from manifest module."""

    def test_hash_file_works(self, tmp_path):
        """hash_file should return correct SHA256 hex digest."""
        from src.manifest import hash_file

        f = tmp_path / "test.bin"
        content = b"test content for hashing"
        f.write_bytes(content)

        import hashlib
        expected = hashlib.sha256(content).hexdigest()
        result = hash_file(str(f))
        assert result == expected

    def test_hash_file_returns_64_char_hex(self, tmp_path):
        """Result should be a 64-char lowercase hex string."""
        from src.manifest import hash_file

        f = tmp_path / "data.jsonl"
        f.write_text('{"test": true}\n')
        result = hash_file(str(f))
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_file_deterministic(self, tmp_path):
        """Same file content should produce same hash."""
        from src.manifest import hash_file

        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        h1 = hash_file(str(f))
        h2 = hash_file(str(f))
        assert h1 == h2


# ===========================================================================
# Test: Argparse defaults
# ===========================================================================

class TestArgparseDefaults:
    """Verify default arg values are sensible."""

    def test_default_model(self):
        args = parse_args([])
        assert args.model == "Qwen/Qwen2.5-3B-Instruct"

    def test_default_data_path(self):
        args = parse_args([])
        assert args.data == "/data/train.jsonl"

    def test_default_output(self):
        args = parse_args([])
        assert args.output == "/data/output/adapter"

    def test_default_lora_rank(self):
        args = parse_args([])
        assert args.lora_rank == 16

    def test_default_lora_alpha(self):
        args = parse_args([])
        assert args.lora_alpha == 32

    def test_default_lr(self):
        args = parse_args([])
        assert args.lr == 2e-4

    def test_default_epochs(self):
        args = parse_args([])
        assert args.epochs == 1

    def test_default_batch_size(self):
        args = parse_args([])
        assert args.batch_size == 4

    def test_default_max_seq_length(self):
        args = parse_args([])
        assert args.max_seq_length == 1024

    def test_default_max_steps(self):
        args = parse_args([])
        assert args.max_steps == -1

    def test_default_eval_data_none(self):
        args = parse_args([])
        assert args.eval_data is None

    def test_default_eval_after_train_false(self):
        args = parse_args([])
        assert args.eval_after_train is False

    def test_custom_args_override(self):
        args = parse_args([
            "--model", "meta-llama/Llama-3-8B",
            "--data", "/tmp/data.jsonl",
            "--output", "/tmp/out",
            "--lora-rank", "32",
            "--lr", "1e-4",
            "--epochs", "3",
        ])
        assert args.model == "meta-llama/Llama-3-8B"
        assert args.data == "/tmp/data.jsonl"
        assert args.output == "/tmp/out"
        assert args.lora_rank == 32
        assert args.lr == 1e-4
        assert args.epochs == 3

    def test_lora_alpha_is_double_rank_by_default(self):
        """Default alpha (32) should be 2x default rank (16)."""
        args = parse_args([])
        assert args.lora_alpha == 2 * args.lora_rank


# ===========================================================================
# Test: CLI --help works
# ===========================================================================

class TestCLIHelp:
    """Test that train.py --help works."""

    def test_cli_help_works(self):
        """Running `python3 train.py --help` should return 0."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=30,
        )
        assert result.returncode == 0
        assert "LoRA" in result.stdout or "lora" in result.stdout.lower()

    def test_help_shows_all_args(self):
        """--help should document all the key arguments."""
        project_root = os.path.join(os.path.dirname(__file__), "..")
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=30,
        )
        assert "--model" in result.stdout
        assert "--data" in result.stdout
        assert "--output" in result.stdout
        assert "--lora-rank" in result.stdout
        assert "--lora-alpha" in result.stdout
        assert "--lr" in result.stdout
        assert "--epochs" in result.stdout
        assert "--batch-size" in result.stdout


# ===========================================================================
# Test: save_training_metadata
# ===========================================================================

class TestSaveTrainingMetadata:
    """Test the metadata saving function."""

    def test_saves_json_file(self, tmp_path):
        """Should create training_meta.json in output dir."""
        from types import SimpleNamespace

        args = SimpleNamespace(
            lora_rank=16,
            lora_alpha=32,
            lr=2e-4,
            epochs=1,
            batch_size=4,
            max_seq_length=1024,
            max_steps=-1,
        )
        meta = save_training_metadata(
            output_dir=str(tmp_path),
            base_model="Qwen/Qwen2.5-3B-Instruct",
            data_path="/data/train.jsonl",
            data_hash="abc123" * 10 + "abcd",
            final_loss=0.5432,
            global_steps=100,
            args=args,
            adapter_hash="def456" * 10 + "defg",
        )
        meta_path = tmp_path / "training_meta.json"
        assert meta_path.exists()

        with open(str(meta_path)) as f:
            loaded = json.load(f)

        assert loaded["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert loaded["data_path"] == "/data/train.jsonl"
        assert loaded["data_hash"] == "abc123" * 10 + "abcd"
        assert loaded["final_loss"] == 0.5432
        assert loaded["global_steps"] == 100
        assert loaded["config"]["lora_rank"] == 16
        assert loaded["config"]["lora_alpha"] == 32
        assert loaded["config"]["learning_rate"] == 2e-4
        assert loaded["config"]["lora_target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert loaded["config"]["lora_use_rslora"] is True
        assert loaded["adapter_hash"] == "def456" * 10 + "defg"
        assert "timestamp" in loaded

    def test_creates_output_dir(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        from types import SimpleNamespace

        output_dir = str(tmp_path / "nested" / "output")
        args = SimpleNamespace(
            lora_rank=16, lora_alpha=32, lr=2e-4, epochs=1,
            batch_size=4, max_seq_length=1024, max_steps=-1,
        )
        save_training_metadata(
            output_dir=output_dir,
            base_model="test",
            data_path="/data/train.jsonl",
            data_hash="hash",
            final_loss=1.0,
            global_steps=10,
            args=args,
        )
        assert os.path.isdir(output_dir)
        assert os.path.isfile(os.path.join(output_dir, "training_meta.json"))
