"""
Tests for cli.py -- exercises each subcommand via subprocess.run.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

CLI = [sys.executable, 'cli.py']
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run_cli(*args, cwd=None):
    """Run cli.py with given args, return CompletedProcess."""
    if cwd is None:
        cwd = PROJECT_DIR
    cmd = CLI + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
    )


class TestNoArgs:
    def test_no_args_returns_exit_1(self):
        p = run_cli()
        assert p.returncode == 1
        assert 'usage' in p.stdout.lower() or 'usage' in p.stderr.lower() or 'help' in p.stdout.lower()

    def test_invalid_command_returns_nonzero(self):
        p = run_cli('nonexistent-command')
        assert p.returncode != 0


class TestSmokeTest:
    def test_smoke_test_passes(self):
        p = run_cli('smoke-test')
        assert p.returncode == 0, f"stdout: {p.stdout}\nstderr: {p.stderr}"
        assert 'PASS' in p.stdout

    def test_smoke_test_output_contains_steps(self):
        p = run_cli('smoke-test')
        assert p.returncode == 0
        assert 'Generating 2 games' in p.stdout
        assert 'Verifying JSONL' in p.stdout
        assert 'Building manifest' in p.stdout
        assert 'Verifying manifest' in p.stdout


class TestGenerate:
    def test_generate_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_out = os.path.join(tmpdir, 'train.jsonl')
            eval_out = os.path.join(tmpdir, 'eval.jsonl')

            p = run_cli(
                'generate',
                '--num-games', '5',
                '--max-steps', '5',
                '--seed-start', '0',
                '--output', train_out,
                '--eval-output', eval_out,
                '--eval-fraction', '0.4',
            )
            assert p.returncode == 0, f"stdout: {p.stdout}\nstderr: {p.stderr}"
            assert os.path.isfile(train_out)
            assert os.path.isfile(eval_out)

            # Verify JSONL content
            with open(train_out) as f:
                lines = f.readlines()
            assert len(lines) > 0
            for line in lines:
                obj = json.loads(line.strip())
                assert 'conversations' in obj
                roles = [c['role'] for c in obj['conversations']]
                assert roles == ['system', 'user', 'assistant']

    def test_generate_no_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_out = os.path.join(tmpdir, 'train.jsonl')

            p = run_cli(
                'generate',
                '--num-games', '2',
                '--max-steps', '3',
                '--output', train_out,
            )
            assert p.returncode == 0
            assert os.path.isfile(train_out)
            assert 'Generation complete' in p.stdout

    def test_generate_stats_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_out = os.path.join(tmpdir, 'train.jsonl')

            p = run_cli(
                'generate',
                '--num-games', '2',
                '--max-steps', '3',
                '--output', train_out,
            )
            assert p.returncode == 0
            assert 'Total games:' in p.stdout
            assert 'Total examples:' in p.stdout


class TestReport:
    def test_report_creates_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = run_cli(
                'report',
                '--seed', '42',
                '--max-steps', '5',
                '--output-dir', tmpdir,
            )
            assert p.returncode == 0, f"stdout: {p.stdout}\nstderr: {p.stderr}"

            html_path = os.path.join(tmpdir, 'game_seed_42.html')
            assert os.path.isfile(html_path)

            with open(html_path) as f:
                content = f.read()
            assert '<html>' in content
            assert 'NLE Game Replay' in content

    def test_report_creates_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = run_cli(
                'report',
                '--seed', '42',
                '--max-steps', '5',
                '--output-dir', tmpdir,
            )
            assert p.returncode == 0

            text_path = os.path.join(tmpdir, 'game_seed_42.txt')
            assert os.path.isfile(text_path)

    def test_report_prints_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = run_cli(
                'report',
                '--seed', '42',
                '--max-steps', '5',
                '--output-dir', tmpdir,
            )
            assert p.returncode == 0
            assert 'Seed 42' in p.stdout


class TestEvaluate:
    def test_evaluate_no_server_exits_gracefully(self):
        p = run_cli(
            'evaluate',
            '--seeds', '100,101',
            '--max-steps', '3',
            '--server-url', 'http://127.0.0.1:59999',
        )
        # Should return 0 (graceful exit), not crash
        assert p.returncode == 0, f"stdout: {p.stdout}\nstderr: {p.stderr}"
        assert 'not available' in p.stdout.lower() or 'warning' in p.stdout.lower()


class TestManifest:
    def test_manifest_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy training data file
            train_path = os.path.join(tmpdir, 'train.jsonl')
            with open(train_path, 'w') as f:
                for i in range(5):
                    f.write(json.dumps({'conversations': []}) + '\n')

            adapter_dir = os.path.join(tmpdir, 'adapter')
            os.makedirs(adapter_dir)

            manifest_path = os.path.join(tmpdir, 'manifest.json')

            p = run_cli(
                'manifest',
                '--base-model', 'Qwen/Qwen2.5-3B-Instruct',
                '--training-data', train_path,
                '--adapter', adapter_dir,
                '--baseline-scores', '{"field_accuracy": 0.3}',
                '--post-scores', '{"field_accuracy": 0.7}',
                '--output', manifest_path,
            )
            assert p.returncode == 0, f"stdout: {p.stdout}\nstderr: {p.stderr}"
            assert os.path.isfile(manifest_path)

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Verify manifest structure
            assert manifest['version'] == '1.0'
            assert manifest['base_model']['name'] == 'Qwen/Qwen2.5-3B-Instruct'
            assert manifest['training_data']['path'] == train_path
            assert manifest['training_data']['num_lines'] == 5
            assert manifest['training_data']['sha256'] != ''
            assert 'manifest_hash' in manifest
            assert manifest['results']['baseline_scores'] == {'field_accuracy': 0.3}
            assert manifest['results']['post_training_scores'] == {'field_accuracy': 0.7}
            assert manifest['results']['improvement'] == pytest.approx({'field_accuracy': 0.4})

    def test_manifest_hash_verifiable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, 'train.jsonl')
            with open(train_path, 'w') as f:
                f.write('{"test": true}\n')

            adapter_dir = os.path.join(tmpdir, 'adapter')
            os.makedirs(adapter_dir)

            manifest_path = os.path.join(tmpdir, 'manifest.json')

            p = run_cli(
                'manifest',
                '--base-model', 'test-model',
                '--training-data', train_path,
                '--adapter', adapter_dir,
                '--baseline-scores', '{}',
                '--post-scores', '{}',
                '--output', manifest_path,
            )
            assert p.returncode == 0

            from src.manifest import load_manifest, verify_manifest
            manifest = load_manifest(manifest_path)
            result = verify_manifest(manifest)
            assert result['valid'] is True

    def test_manifest_missing_required_args(self):
        p = run_cli('manifest')
        assert p.returncode != 0
