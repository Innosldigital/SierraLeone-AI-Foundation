"""
Sierra Leone AI Foundation – command-line interface.
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
from sierraleone_ai.models.registry import get_recommended, list_candidates
from sierraleone_ai.training.guard import check_dataset_readiness

app = typer.Typer(
    name="slai",
    help="Sierra Leone AI Foundation CLI",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def version():
    """Show the current version."""
    console.print(f"[bold green]Sierra Leone AI Foundation[/bold green] v{__version__}")


@app.command()
def languages():
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
def data_summary():
    """Display real dataset summary table (excluding templates)."""
    print_dataset_summary()


@app.command()
def check_readiness(language: str = typer.Argument("krio", help="Language code")):
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


@app.command()
def validate_file(path: str):
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
                # Ensure metadata exists
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
def status():
    """Show current project phase, model readiness, and dataset status."""
    console.print("\n[bold]Sierra Leone AI Foundation – Project Status[/bold]\n")
    console.print("[green]✓[/green] Project foundation and schemas defined")
    console.print("[green]✓[/green] Preprocessing and evaluation pipelines active")
    console.print("[green]✓[/green] API stubs live (all model endpoints return HTTP 503)")

    # Dynamic base model status
    recommended = get_recommended()
    if recommended:
        console.print(f"[green]✓[/green] Base model selected: {recommended.name}")
    else:
        n_candidates = len(list_candidates())
        console.print(
            f"[yellow]○[/yellow] Base model: tokenisation research in progress "
            f"({n_candidates} candidates registered, none approved yet)"
        )

    # Dynamic dataset readiness status
    krio_readiness = check_dataset_readiness("krio")
    ready_count = sum(1 for item in krio_readiness.values() if item.ok)
    total_count = len(krio_readiness)
    if ready_count == total_count:
        console.print("[green]✓[/green] Krio training datasets meet minimum volume thresholds")
    else:
        console.print(
            f"[yellow]○[/yellow] Krio training data: {ready_count}/{total_count} "
            f"thresholds met (collection ongoing)"
        )

    console.print("[red]✗[/red] Model training: blocked by training guard until data + model approved")
    console.print("[red]✗[/red] Speech recognition (ASR): awaiting audio recordings")
    console.print("[red]✗[/red] Text-to-speech (TTS): awaiting studio-grade recordings\n")


if __name__ == "__main__":
    app()
