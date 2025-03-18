"""Display utilities for console UI components.

This module provides functions for creating and managing console UI elements
using the Rich library.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from pyspark.sql import DataFrame


def create_progress(message: str) -> Progress:
    """Create a progress indicator with a message.
    
    Args:
        message: Message to display with the progress indicator
        
    Returns:
        Progress object configured with spinner and message
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[green]{message}[/green]")
    )


def display_dataframe_preview(
    df: DataFrame,
    console: Console,
    num_rows: int = 5,
    num_cols: int = 6
) -> None:
    """Display a preview of a DataFrame as a Rich table.
    
    Args:
        df: DataFrame to display
        console: Rich console instance
        num_rows: Number of rows to display
        num_cols: Maximum number of columns to display
    """
    schema = df.schema
    data = df.limit(num_rows).collect()
    
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )
    
    # Add columns (limit if more exist)
    fields = schema.fields
    total_cols = len(fields)
    display_fields = fields[:num_cols] if total_cols > num_cols else fields
    
    for field in display_fields:
        table.add_column(field.name, style="green")
        
    # Add ellipsis column if needed
    if total_cols > num_cols:
        table.add_column("...", style="dim")
    
    # Add rows
    for row in data:
        visible_values = [str(row[i]) for i in range(min(num_cols, total_cols))]
        if total_cols > num_cols:
            visible_values.append(f"({total_cols - num_cols} more)")
        table.add_row(*visible_values)
        
    console.print(table)
    
    # Show truncation information
    info_parts = []
    if num_rows < df.count():
        info_parts.append(f"Showing {min(num_rows, df.count())} of {df.count()} rows")
    if total_cols > num_cols:
        info_parts.append(f"Showing {num_cols} of {total_cols} columns")
        
    if info_parts:
        console.print(f"[italic]({' | '.join(info_parts)})[/italic]")


def create_menu_table(
    options: List[Dict[str, str]],
    title: str = "Menu Options"
) -> Table:
    """Create a table displaying menu options.
    
    Args:
        options: List of dictionaries with keys 'key' and 'description'
        title: Title for the menu
        
    Returns:
        Rich Table object with menu options
    """
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title=title,
        box=box.ROUNDED
    )
    
    table.add_column("Key", style="yellow", width=6)
    table.add_column("Description", style="green")
    
    for option in options:
        table.add_row(option["key"], option["description"])
        
    return table


def create_status_panel(
    df: Optional[DataFrame] = None,
    additional_info: Optional[List[str]] = None
) -> Panel:
    """Create a status panel showing current state.
    
    Args:
        df: Current DataFrame (if any)
        additional_info: Additional status information to display
        
    Returns:
        Rich Panel object with status information
    """
    status = []
    
    if df is not None:
        status.append(
            f"[green]Data loaded:[/green] "
            f"{df.count():,} rows, {len(df.columns)} columns"
        )
    else:
        status.append("[yellow]No data loaded[/yellow]")
        
    if additional_info:
        status.extend(additional_info)
        
    return Panel(
        "\n".join(status),
        title="Status",
        border_style="blue"
    ) 