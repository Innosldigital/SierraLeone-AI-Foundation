"""
Sierra Leone AI Foundation – command-line interface (SLAI-020 / SLAI-024 / SLAI-029).
Entry point: `slai`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from sierraleone_ai import SUPPORTED_LANGUAGES, __version__
from sierraleone_ai.datasets.stats import print_dataset_summary
from sierraleone_ai.evaluation.tokenizer_audit import audit_tokenizer_on_corpus
from sierraleone_ai.models.registry import get_recommended, list_candidates
from sierraleone_ai.training.guard import check_dataset_readiness, require_training_approval

app = typer.Typer(
    name="slai",
    help="Sierra Leone AI Foundation CLI",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Show the current version."""
    console.print(f"[bold green]Sierra Leone AI Foundation[/bold green] v{__version__}")


@app.command()
def languages() -> None:
    """List supported and planned languages."""
    table = Table(title="Sierra Leonean Languages", show_lines=True)
    table.add_column("Code", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("ISO 639-3")
    table.add_column("Status", style="yellow")
    for code, info in SUPPORTED_LANGUAGES.items():
        table.add_row(
            code,
            info["name"],
            info.get("iso_639_3", ""),
            info["status"],
        )
    console.print(table)


@app.command()
def data_summary() -> None:
    """Display exploratory dataset summary table (PREVIEW – not for training)."""
    console.print("[bold yellow]PREVIEW – not for training[/bold yellow]")
    print_dataset_summary()


@app.command()
def build_manifest(
    language: str = typer.Argument("krio", help="Language code"),
    split: str = typer.Option("train", "--split", help="Data split to scan and manifest"),
) -> None:
    """Build a versioned, hash-verified dataset manifest for training readiness (PA-001)."""
    from sierraleone_ai.datasets.manifest import (
        build_manifest as _build_manifest,
    )
    from sierraleone_ai.datasets.manifest import (
        load_manifest,
    )

    try:
        manifest_path = _build_manifest(language=language, split=split)
        manifest = load_manifest(manifest_path)
        console.print(f"[bold green]Dataset manifest built:[/bold green] {manifest_path}")
        console.print(f"  [cyan]Manifest ID:[/cyan] {manifest.manifest_id}")
        console.print(f"  [cyan]Files indexed:[/cyan] {len(manifest.files)}")
        console.print("  [cyan]Eligible record counts:[/cyan]")
        for task, count in manifest.eligible_counts.items():
            console.print(f"    • {task}: {count:,}")
        if manifest.skipped:
            console.print(f"  [yellow]Skipped records:[/yellow] {sum(manifest.skipped.values()):,}")
    except Exception as exc:
        console.print(f"[red]Error building dataset manifest: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def build_benchmark_manifest(
    language: str = typer.Argument("krio", help="Language code"),
) -> None:
    """Build a versioned, hash-verified manifest for the held-out benchmark (PA-004)."""
    from sierraleone_ai.evaluation.benchmarks import (
        build_benchmark_manifest as _build_benchmark_manifest,
    )

    try:
        manifest_path = _build_benchmark_manifest(language=language)
        console.print(f"[bold green]Benchmark manifest built:[/bold green] {manifest_path}")
    except Exception as exc:
        console.print(f"[red]Error building benchmark manifest: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def check_readiness(
    language: str = typer.Argument("krio", help="Language code"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the training readiness dry-run plan (downloads nothing, starts nothing)",
    ),
) -> None:
    """Check dataset training readiness against minimum thresholds for a language."""
    items = check_dataset_readiness(language)
    table = Table(title=f"Dataset Training Readiness ({language})", show_lines=True)
    table.add_column("Requirement", style="cyan")
    table.add_column("Current Records", style="yellow")
    table.add_column("Required Threshold", style="green")
    table.add_column("Ready?", style="magenta")

    for item in items.values():
        status_text = "[green]YES[/green]" if item.ok else "[red]NO[/red]"
        table.add_row(item.key, f"{item.have:,}", f"{item.need:,}", status_text)

    console.print(table)

    if dry_run:
        plan = require_training_approval(
            task=f"{language} model training dry-run",
            language=language,
            dry_run=True,
        )
        console.print("\n[bold cyan]Dry-Run Plan (downloads=0, writes=0):[/bold cyan]")
        console.print(json.dumps(plan, indent=2))


@app.command()
def validate_file(
    path: str = typer.Argument(..., help="Path to JSONL dataset file to validate"),
    kind: str = typer.Option(
        "text", "--kind", help="Record kind: text, translation, conversation, audio, benchmark"
    ),
    task: str = typer.Option(None, "--task", help="Optional task for eligibility check"),
    split: str = typer.Option(
        "train", "--split", help="Split name (train, validation, test, held_out)"
    ),
) -> None:
    """Validate a JSONL dataset file against Pydantic schema and eligibility rules (PA-016)."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(code=1)

    from sierraleone_ai.datasets.eligibility import EligibilityPolicy, Task
    from sierraleone_ai.datasets.loader import LoadReport, load_jsonl_records, scan_jsonl_for_pii
    from sierraleone_ai.datasets.schema import (
        AudioRecord,
        BenchmarkItem,
        ConversationRecord,
        KrioTextRecord,
        TranslationRecord,
    )

    kind_map: dict[str, tuple[type[Any], Task]] = {
        "text": (KrioTextRecord, "text_training"),
        "translation": (TranslationRecord, "translation_training"),
        "translations": (TranslationRecord, "translation_training"),
        "conversation": (ConversationRecord, "conversation_training"),
        "conversations": (ConversationRecord, "conversation_training"),
        "audio": (AudioRecord, "asr_training"),
        "benchmark": (BenchmarkItem, "evaluation"),
    }
    if kind not in kind_map:
        console.print(
            f"[red]Error: Unsupported kind '{kind}'. Choices: {sorted(kind_map.keys())}[/red]"
        )
        raise typer.Exit(code=1)

    model_cls, default_task = kind_map[kind]
    actual_task = cast(Task, task or default_task)
    policy = EligibilityPolicy(task=actual_task, split=split)

    report: LoadReport[Any] = load_jsonl_records(
        file_path,
        model_cls=model_cls,
        split=split,
        policy=policy,
        strict=False,
    )
    pii_issues = scan_jsonl_for_pii(file_path)

    console.print(f"Validation results for [bold]{file_path.name}[/bold]:")
    console.print(f"  [cyan]Total lines read:[/cyan] {report.total_lines}")
    console.print(f"  [green]Valid and eligible records:[/green] {len(report.records)}")
    if report.skipped:
        console.print("  [yellow]Per-reason skipped records:[/yellow]")
        for reason, count in sorted(report.skipped.items()):
            console.print(f"    • {reason}: {count}")
    if report.issues:
        console.print(f"  [red]Schema/validation issues ({len(report.issues)} total):[/red]")
        for issue in report.issues[:10]:
            console.print(f"    • Line {issue.line}: {issue.error}")
    if pii_issues:
        console.print(f"  [red]PII detected ({len(pii_issues)} instances):[/red]")
        for pii in pii_issues[:10]:
            console.print(f"    • Line {pii.line}: {pii.error}")

    if report.issues or pii_issues or len(report.records) == 0:
        raise typer.Exit(code=1)


@app.command()
def audit_tokenizer(
    tokenizer: str = typer.Option(..., "--tokenizer", help="HuggingFace tokenizer ID"),
    corpus: str = typer.Option(..., "--corpus", help="Path to native-reviewed Krio JSONL corpus"),
    min_sentences: int = typer.Option(
        1000,
        "--min-sentences",
        help="Minimum required native-reviewed sentences (default: 1000)",
    ),
) -> None:
    """
    Run reproducible tokenizer fertility and orthography audit (SLAI-029).
    Downloads tokenizer files only — never model weights.
    """
    try:
        report = audit_tokenizer_on_corpus(
            corpus_path=Path(corpus),
            tokenizer_id=tokenizer,
            min_sentences=min_sentences,
        )
        console.print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    except Exception as exc:
        console.print(f"[red]Tokenizer audit failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def serve(
    host: str = typer.Option(None, help="Host to bind server (default from settings: 127.0.0.1)"),
    port: int = typer.Option(None, help="Port to bind server (default from settings: 8000)"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the FastAPI application server (PA-014)."""
    import uvicorn

    from sierraleone_ai.config import settings

    h = host or settings.api_host
    p = port or settings.api_port
    console.print(f"[bold green]Starting API server at http://{h}:{p}[/bold green]")
    uvicorn.run("sierraleone_ai.api.app:app", host=h, port=p, reload=reload)


@app.command()
def status() -> None:
    """
    Show current project state across the 5 canonical levels:
    `implemented`, `scaffolded`, `not configured`, `not ready`, `operational` (SLAI-020 / PA-016).
    """
    import subprocess

    from sierraleone_ai.datasets.manifest import latest_manifest_path, load_manifest
    from sierraleone_ai.evaluation.benchmarks import check_held_out_integrity
    from sierraleone_ai.serving.state import model_state

    console.print(f"\n[bold]Sierra Leone AI Foundation v{__version__} – Platform Status[/bold]\n")

    # Git commit
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    # Optional extras check
    extras_installed: list[str] = []
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401

        extras_installed.append("training")
    except ImportError:
        pass
    try:
        import soundfile  # noqa: F401

        extras_installed.append("audio")
    except ImportError:
        pass
    try:
        import sacrebleu  # noqa: F401

        extras_installed.append("eval")
    except ImportError:
        pass

    # GPU presence
    gpu_desc = "None"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_desc = f"{torch.cuda.get_device_name(0)} (CUDA)"
    except Exception:
        pass

    table = Table(title=f"Component Status (git: {git_hash})", show_lines=True)
    table.add_column("Subsystem", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("Details")

    # 1. Dataset Schemas & Loader
    table.add_row(
        "Dataset Schemas, Loader & Normaliser",
        "implemented",
        "Strict JSONL validation, consent/licence gates, and NFC normaliser active",
    )

    # 2. FastAPI Service
    if model_state.is_ready():
        api_state = "operational"
        api_details = f"Loaded models: {model_state.get_loaded_models()}"
    else:
        api_state = "scaffolded"
        api_details = "Endpoints active; returning 503 until models loaded into ModelState"
    table.add_row("FastAPI Service", api_state, api_details)

    # 3. Base Model Selection
    recommended = get_recommended()
    n_candidates = len(list_candidates())
    if recommended:
        table.add_row("Base Model Selection", "implemented", f"Selected: {recommended.name}")
    else:
        table.add_row(
            "Base Model Selection",
            "not configured",
            f"{n_candidates} candidates registered; awaiting tokenizer audit & benchmark",
        )

    # 4. Krio Dataset Manifest & Readiness
    latest_krio_manifest = latest_manifest_path("krio", "train")
    if latest_krio_manifest is not None:
        try:
            m = load_manifest(latest_krio_manifest)
            manifest_details = f"{m.manifest_id}: {sum(m.eligible_counts.values()):,} eligible records across {len(m.files)} files"
            manifest_state = "implemented"
        except Exception as exc:
            manifest_details = f"Invalid manifest: {exc}"
            manifest_state = "not ready"
    else:
        manifest_details = (
            "No dataset manifest built (run `slai build-manifest krio --split train`)"
        )
        manifest_state = "not configured"
    table.add_row("Dataset Manifest (Krio train)", manifest_state, manifest_details)

    # 5. Held-Out Benchmark Integrity
    bench_result = check_held_out_integrity("krio")
    if bench_result["ok"]:
        bench_state = "implemented"
        bench_details = f"{bench_result['count']} items verified against manifest"
    else:
        bench_state = "not ready"
        bench_details = f"Not ready ({', '.join(bench_result['reasons'])})"
    table.add_row("Held-Out Benchmark", bench_state, bench_details)

    # 6. Models & Serving
    if model_state.is_ready():
        model_state_str = "operational"
        model_details = f"Operational models: {model_state.get_loaded_models()}"
    else:
        model_state_str = "not ready"
        model_details = "No trained models operational yet"
    table.add_row("Model Operational State", model_state_str, model_details)

    # 7. Hardware & Extras
    table.add_row(
        "Hardware & Optional Extras",
        "implemented" if extras_installed else "scaffolded",
        f"Extras: {extras_installed or ['none']}; GPU: {gpu_desc}",
    )

    console.print(table)


if __name__ == "__main__":
    app()
