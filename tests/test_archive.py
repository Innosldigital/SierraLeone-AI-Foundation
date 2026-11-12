"""Tests for source archive generation and data deny-by-default allow-list (PA-006)."""

from pathlib import Path, PurePosixPath

from scripts.make_source_archive import _data_allowed, build_clean_archive, should_include_path


def test_data_allowed_allows_templates_and_manifests():
    assert _data_allowed(PurePosixPath("datasets/krio/cleaned/example_krio_text_train.jsonl"))
    assert _data_allowed(PurePosixPath("datasets/krio/audio/.gitkeep"))
    assert _data_allowed(PurePosixPath("datasets/source_register.csv"))
    assert _data_allowed(PurePosixPath("datasets/manifests/krio-train-v1.manifest.json"))
    assert _data_allowed(PurePosixPath("benchmarks/krio/.gitkeep"))
    assert _data_allowed(PurePosixPath("benchmarks/krio/held_out_eval_schema.jsonl"))
    assert _data_allowed(PurePosixPath("benchmarks/manifests/krio-v1.manifest.json"))


def test_data_allowed_denies_real_datasets_and_benchmarks():
    assert not _data_allowed(PurePosixPath("datasets/krio/cleaned/krio_text_train.jsonl"))
    assert not _data_allowed(PurePosixPath("datasets/krio/raw/uncleaned.txt"))
    assert not _data_allowed(PurePosixPath("datasets/krio/audio/spk01/rec.wav"))
    assert not _data_allowed(PurePosixPath("benchmarks/krio/real_benchmark.jsonl"))
    assert not _data_allowed(PurePosixPath("benchmarks/krio/eval_held_out.jsonl"))


def test_should_include_path_rules():
    assert should_include_path(PurePosixPath("datasets/krio/cleaned/example_krio_text_train.jsonl"))
    assert not should_include_path(PurePosixPath("datasets/krio/cleaned/krio_text_train.jsonl"))
    assert not should_include_path(PurePosixPath(".env"))
    assert not should_include_path(PurePosixPath(".venv/bin/python"))
    assert not should_include_path(PurePosixPath("weights.safetensors"))


def test_build_clean_archive_filters_disallowed_files(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()

    # Allowed files
    (root / "src").mkdir()
    (root / "src" / "code.py").write_text("print('hello')", encoding="utf-8")
    (root / "datasets" / "krio" / "cleaned").mkdir(parents=True)
    (root / "datasets" / "krio" / "cleaned" / ".gitkeep").touch()
    (root / "datasets" / "krio" / "cleaned" / "example_krio_text_train.jsonl").write_text(
        "{}", encoding="utf-8"
    )
    (root / "datasets" / "source_register.csv").write_text("source_name,decision", encoding="utf-8")
    (root / "datasets" / "manifests").mkdir(parents=True)
    (root / "datasets" / "manifests" / "krio-train-v1.manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / "benchmarks" / "krio").mkdir(parents=True)
    (root / "benchmarks" / "krio" / ".gitkeep").touch()
    (root / "benchmarks" / "krio" / "held_out_eval_schema.jsonl").write_text("{}", encoding="utf-8")

    # Disallowed files
    (root / "datasets" / "krio" / "cleaned" / "krio_text_train.jsonl").write_text(
        "secret real data", encoding="utf-8"
    )
    (root / "benchmarks" / "krio" / "real_eval.jsonl").write_text(
        "secret real benchmark", encoding="utf-8"
    )
    (root / ".env").write_text("SECRET=123", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "pyvenv.cfg").write_text("home = ...", encoding="utf-8")
    (root / "model.safetensors").write_text("fake weights", encoding="utf-8")

    out_zip = tmp_path / "out.zip"
    archived = build_clean_archive(root, out_zip)

    assert "src/code.py" in archived
    assert "datasets/krio/cleaned/.gitkeep" in archived
    assert "datasets/krio/cleaned/example_krio_text_train.jsonl" in archived
    assert "datasets/source_register.csv" in archived
    assert "datasets/manifests/krio-train-v1.manifest.json" in archived
    assert "benchmarks/krio/.gitkeep" in archived
    assert "benchmarks/krio/held_out_eval_schema.jsonl" in archived

    # Ensure disallowed are excluded
    assert "datasets/krio/cleaned/krio_text_train.jsonl" not in archived
    assert "benchmarks/krio/real_eval.jsonl" not in archived
    assert ".env" not in archived
    assert any(".venv" in p for p in archived) is False
    assert "model.safetensors" not in archived
