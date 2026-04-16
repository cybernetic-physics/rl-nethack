import zipfile

from src.nld_dataset_prep import (
    discover_altorg_roots,
    extract_zip_archives,
    is_altorg_dataset_root,
    summarize_altorg_root,
)


def test_is_altorg_dataset_root_and_summary(tmp_path):
    root = tmp_path / "nld-aa"
    user_dir = root / "alice"
    user_dir.mkdir(parents=True)
    (root / "blacklist.txt").write_text("none\n")
    (root / "xlogfile.0").write_text("game metadata\n")
    (user_dir / "alice.ttyrec.bz2").write_text("fake ttyrec\n")

    assert is_altorg_dataset_root(root)
    summary = summarize_altorg_root(str(root))
    assert summary["valid_altorg_root"] is True
    assert summary["ttyrec_count"] == 1
    assert summary["user_count"] == 1


def test_extract_zip_archives_and_discover_root(tmp_path):
    archive = tmp_path / "nld.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset/blacklist.txt", "none\n")
        zf.writestr("dataset/xlogfile.0", "metadata\n")
        zf.writestr("dataset/bob/bob.ttyrec.bz2", "fake ttyrec\n")

    extract_dir = tmp_path / "extract"
    result = extract_zip_archives([str(archive)], str(extract_dir))
    assert result["extracted_files"] == 3

    roots = discover_altorg_roots(str(extract_dir))
    assert roots == [str(extract_dir / "dataset")]
