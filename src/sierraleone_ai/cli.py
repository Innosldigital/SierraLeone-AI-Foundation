"""
Sierra Leone AI Foundation – command-line interface (SLAI-020 / SLAI-024 / SLAI-029).
Entry point: `slai`
"""
from __future__ import annotations

import json
from pathlib import Path

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
def validate_file(path: str) -> None:
    """Validate a JSONL dataset file against Pydantic schema rules."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(code=1)

    valid_count = 0
    invalid_count = 0
    errors: list[str] = []

    with file_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                if "metadata" not in data:
                    invalid_count += 1
                    errors.append(f"Line {lineno}: missing 'metadata' field")
                    continue
                valid_count += 1
            except Exception as exc:
                invalid_count += 1
                errors.append(f"Line {lineno}: {exc}")

    console.print(f"Validation results for [bold]{file_path.name}[/bold]:")
    console.print(f"  [green]Valid lines:[/green] {valid_count}")
    console.print(f"  [red]Invalid lines:[/red] {invalid_count}")
    if errors:
        console.print("\nSample errors:")
        for err in errors[:5]:
            console.print(f"  [red]•[/red] {err}")


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
    `implemented`, `scaffolded`, `not configured`, `not ready`, `operational` (SLAI-020).
    """
    console.print("\n[bold]Sierra Leone AI Foundation – Project Status[/bold]\n")

    table = Table(title="Component Status (5-Level Audit State)", show_lines=True)
    table.add_column("Subsystem", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("Details")

    table.add_row(
        "Dataset Schemas, Loader & Normaliser",
        "implemented",
        "Strict JSONL validation, consent/licence gates, and NFC normaliser active",
    )
    table.add_row(
        "FastAPI Service (`/health/live`, `/health/ready`, 503 stubs)",
        "scaffolded",
        "Liveness up; readiness & model routes return 503 until a model is trained",
    )

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

    krio_readiness = check_dataset_readiness("krio")
    ready_count = sum(1 for item in krio_readiness.values() if item.ok)
    total_count = len(krio_readiness)
    data_state = "implemented" if ready_count == total_count else "not ready"
    table.add_row(
        "Krio Training & Benchmark Datasets",
        data_state,
        f"{ready_count}/{total_count} dataset readiness thresholds met",
    )

    table.add_row(
        "LLM / Translation / ASR / TTS Models",
        "not ready",
        "No trained model is operational yet (0 operational models)",
    )

    console.print(table)


if __name__ == "__main__":
    app()
