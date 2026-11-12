"""
Dataset statistics and audit utilities.

Use these to understand the current state of collected data
before making decisions about model training.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from sierraleone_ai.datasets.loader import dataset_summary

console = Console()


def print_dataset_summary() -> None:
    """Print a formatted table showing what datasets are available."""
    summary = dataset_summary()

    table = Table(title="Sierra Leone AI Foundation – Dataset Summary", show_lines=True)
    table.add_column("Language", style="cyan")
    table.add_column("Subdirectory", style="green")
    table.add_column("Records", style="yellow")

    for lang, subdirs in summary.items():
        for subdir, count in subdirs.items():
            table.add_row(lang, subdir, str(count))

    console.print(table)

    if not summary:
        console.print(
            "[yellow]No dataset files found. "
            "See documentation/DATA_COLLECTION_GUIDE.md to begin data collection.[/yellow]"
        )
