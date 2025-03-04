# Created on Wed Feb 05 2025 by 240030614
# Copyright (c) 2025 University of St. Andrews
"""
Textual app for PySpark data manipulation, styled similarly to Harlequin.

Place this file (app.py) in the same directory as textual.css.
Some versions of Textual auto-load textual.css; if not, see "Manual CSS Loading" below.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import os
import sys
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box
from rich.markdown import Markdown
from rich.traceback import install as install_rich_traceback

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.spark_session import create_spark_session, stop_spark_session
from datetime import datetime

# Install rich traceback handler for better error display
install_rich_traceback()

# Default data directory
DEFAULT_DATA_DIR = Path("data/")

class SparkDataConsoleApp:
    """A console app for PySpark data manipulation."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, app_name: str = "SparkDataApp"):
        """
        Initialize the application.
        
        Args:
            data_dir: Directory containing data files
            app_name: Name for the Spark application
        """
        self.data_dir = data_dir
        self.app_name = app_name
        self.session = None
        self.df = None
        self.console = Console()
        self.command_history = []
        self.last_query_result = None
        
    def start(self):
        """Start the application."""
        self.console.print(Panel(
            "[bold blue]Welcome to SparkDataApp[/bold blue]\n"
            "[cyan]A console-based PySpark data analysis tool[/cyan]", 
            border_style="green"
        ))
        
        with Progress(SpinnerColumn(), TextColumn("[green]Creating Spark session...[/green]")) as progress:
            task = progress.add_task("", total=None)
            try:
                self.session = create_spark_session(self.app_name)
                progress.update(task, description="[green]Spark session created successfully![/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error creating Spark session:[/bold red] {str(e)}")
                sys.exit(1)
        
        self.run_main_loop()
    
    def run_main_loop(self):
        """Run the main application loop."""
        while True:
            # Show status bar with current data status
            self._show_status()
            
            menu = """
[bold cyan]Available commands:[/bold cyan]
[yellow]l[/yellow] - Load data
[yellow]/[/yellow] - Query data
[yellow]v[/yellow] - Visualize data
[yellow]f[/yellow] - Filter data
[yellow]s[/yellow] - Save data
[yellow]h[/yellow] - Help
[yellow]q[/yellow] - Quit
            """
            self.console.print(Panel(menu, title="Menu", border_style="blue"))
            
            choice = Prompt.ask("Enter command", choices=["l", "/", "v", "f", "s", "h", "q"], show_choices=False)
            self.command_history.append(choice)
            
            try:
                if choice == 'l':
                    self.load_data()
                elif choice == '/':
                    self.query_data()
                elif choice == 'v':
                    self.visualize_data()
                elif choice == 'f':
                    self.filter_data()
                elif choice == 's':
                    self.save_data()
                elif choice == 'h':
                    self.show_help()
                elif choice == 'q':
                    self.quit()
                    break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                self.console.print("[dim]Use 'h' for help or 'q' to quit[/dim]")
                
            # Add a visual separator between commands
            self.console.print("─" * self.console.width)
    
    def _show_status(self):
        """Show current status information."""
        status = []
        
        if self.df is not None:
            status.append(f"[green]Data loaded:[/green] {self.df.count()} rows, {len(self.df.columns)} columns")
        else:
            status.append("[yellow]No data loaded[/yellow]")
            
        status_text = " | ".join(status)
        self.console.print(f"[dim]{status_text}[/dim]")
    
    def load_data(self):
        """Load data from a CSV file."""
        self.console.print(Panel("[bold]Load Data[/bold]", border_style="yellow"))
        try:
            csv_files = list(self.data_dir.rglob('*.csv'))
            
            if not csv_files:
                self.console.print(f"[bold red]No CSV files found in {self.data_dir} directory[/bold red]")
                if Confirm.ask("Would you like to specify a different data directory?"):
                    dir_path = Prompt.ask("Enter path to data directory")
                    self.data_dir = Path(dir_path)
                    self.load_data()  # Retry with new directory
                return
                
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=6)
            table.add_column("Filename", style="green")
            table.add_column("Size", style="blue")
            table.add_column("Last Modified", style="cyan")
            
            for i, file in enumerate(csv_files, 1):
                size = f"{file.stat().st_size / 1024:.1f} KB"
                modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                table.add_row(str(i), file.name, size, modified)
                
            self.console.print(table)
                
            choice = Prompt.ask("Select file number (or 'c' to cancel)")
            if choice.lower() == 'c':
                return
                
            try:
                index = int(choice) - 1
                file_path = csv_files[index]
                
                with Progress(SpinnerColumn(), TextColumn("[green]Loading data...[/green]")) as progress:
                    task = progress.add_task("", total=None)
                    self.df = self.session.read.csv(str(file_path), header=True, inferSchema=True)
                    row_count = self.df.count()
                    progress.update(task, description=f"[green]Loaded {row_count} rows successfully![/green]")
                    
                self.console.print(f"[bold green]Loaded:[/bold green] {file_path.name}")
                self._display_dataframe(self.df, 5)
                
                # Show schema information
                self.console.print("\n[bold cyan]Schema Information:[/bold cyan]")
                schema_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                schema_table.add_column("Column Name", style="green")
                schema_table.add_column("Data Type", style="blue")
                
                for field in self.df.schema.fields:
                    schema_table.add_row(field.name, str(field.dataType))
                    
                self.console.print(schema_table)
                
            except (ValueError, IndexError):
                self.console.print("[bold red]Invalid selection[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def query_data(self):
        """Query the dataframe by column."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Query Data[/bold]", border_style="yellow"))
        
        # Display all available columns
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim", width=6)
        table.add_column("Column", style="green")
        
        for i, col in enumerate(self.df.columns, 1):
            table.add_row(str(i), col)
        
        self.console.print(table)
        
        choice = Prompt.ask("Select column number")
        try:
            col_index = int(choice) - 1
            if col_index < 0 or col_index >= len(self.df.columns):
                raise ValueError("Invalid column index")
                
            column = self.df.columns[col_index]
            
            with Progress(SpinnerColumn(), TextColumn("[green]Getting values...[/green]")) as progress:
                task = progress.add_task("", total=None)
                values = [row[0] for row in self.df.select(column).distinct().collect()]
            
            values_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            values_table.add_column("#", style="dim", width=6)
            values_table.add_column(f"{column} Values", style="green")
            values_table.add_column("Count", style="blue")
            
            # Count occurrences of each value
            value_counts = self.df.groupBy(column).count().collect()
            value_count_dict = {row[0]: row[1] for row in value_counts}
            
            for i, val in enumerate(values, 1):
                count = value_count_dict.get(val, 0)
                values_table.add_row(str(i), str(val), str(count))
                
            self.console.print(values_table)
                
            val_choice = Prompt.ask("Select value number to filter by (or 'a' for all)")
            
            if val_choice.lower() == 'a':
                self.console.print(f"[bold green]Showing all values for {column}:[/bold green]")
                result = self.df
            else:
                try:
                    index = int(val_choice) - 1
                    selected_val = values[index]
                    
                    with Progress(SpinnerColumn(), TextColumn("[green]Filtering data...[/green]")) as progress:
                        task = progress.add_task("", total=None)
                        result = self.df.filter(self.df[column] == selected_val)
                    
                    self.console.print(f"[bold green]Results for {column} = {selected_val}:[/bold green]")
                    
                except (ValueError, IndexError):
                    self.console.print("[bold red]Invalid selection[/bold red]")
                    return
            
            self._display_dataframe(result, 10)
            self.last_query_result = result
            
            # Ask if user wants to save this query result
            if Confirm.ask("Would you like to save this query result?"):
                self.df = result
                self.console.print("[bold green]Query result saved as current dataframe[/bold green]")
            
        except (ValueError, IndexError) as e:
            self.console.print(f"[bold red]Invalid selection: {str(e)}[/bold red]")
    
    def visualize_data(self):
        """Visualize data using matplotlib."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Visualize Data[/bold]", border_style="yellow"))
        
        # Choose visualization type
        viz_types = ["Bar Chart", "Histogram", "Scatter Plot", "Pie Chart"]
        viz_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        viz_table.add_column("#", style="dim", width=6)
        viz_table.add_column("Visualization Type", style="green")
        viz_table.add_column("Status", style="yellow")
        
        # Mark the visualization types that are unavailable without pandas
        statuses = ["Coming soon", "Coming soon", "Coming soon", "Coming soon"]
        
        for i, (viz, status) in enumerate(zip(viz_types, statuses), 1):
            viz_table.add_row(str(i), viz, status)
            
        self.console.print(viz_table)
        
        self.console.print("\n[bold yellow]Note:[/bold yellow] Advanced visualization features are coming soon. These will be implemented without using pandas library.")
        
        # Offer basic visualization options that don't require pandas
        self.console.print("\n[bold cyan]Basic Data Statistics[/bold cyan]")
        
        if Confirm.ask("Would you like to see summary statistics for the current data?"):
            # Display basic statistics using native PySpark functionality
            try:
                stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                stats_table.add_column("Column", style="green")
                stats_table.add_column("Count", style="blue")
                stats_table.add_column("Mean", style="blue")
                stats_table.add_column("Min", style="blue")
                stats_table.add_column("Max", style="blue")
                
                # Get numeric columns
                numeric_cols = []
                for field in self.df.schema.fields:
                    data_type = str(field.dataType).lower()
                    if "int" in data_type or "double" in data_type or "float" in data_type:
                        numeric_cols.append(field.name)
                
                if not numeric_cols:
                    self.console.print("[yellow]No numeric columns found for statistics[/yellow]")
                    return
                
                # Calculate statistics using PySpark built-in methods
                summary = self.df.select(numeric_cols).summary("count", "mean", "min", "max").collect()
                count_row = summary[0]
                mean_row = summary[1]
                min_row = summary[2]
                max_row = summary[3]
                
                for col in numeric_cols:
                    stats_table.add_row(
                        col,
                        count_row[col],
                        mean_row[col],
                        min_row[col],
                        max_row[col]
                    )
                
                self.console.print(stats_table)
                
            except Exception as e:
                self.console.print(f"[bold red]Error generating statistics:[/bold red] {str(e)}")
    
    def filter_data(self):
        """Filter data with custom conditions."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Filter Data[/bold]", border_style="yellow"))
        
        # Display column information
        col_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        col_table.add_column("#", style="dim", width=6)
        col_table.add_column("Column", style="green")
        col_table.add_column("Data Type", style="blue")
        
        for i, field in enumerate(self.df.schema.fields, 1):
            col_table.add_row(str(i), field.name, str(field.dataType))
            
        self.console.print(col_table)
        
        filter_options = [
            "Equal to (==)",
            "Not equal to (!=)",
            "Greater than (>)",
            "Less than (<)",
            "Contains",
            "Starts with"
        ]
        
        filter_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        filter_table.add_column("#", style="dim", width=6)
        filter_table.add_column("Filter Type", style="green")
        
        for i, f_type in enumerate(filter_options, 1):
            filter_table.add_row(str(i), f_type)
            
        self.console.print(filter_table)
        
        try:
            col_choice = Prompt.ask("Select column to filter")
            col_index = int(col_choice) - 1
            
            if col_index < 0 or col_index >= len(self.df.columns):
                raise ValueError("Invalid column selection")
                
            col_name = self.df.columns[col_index]
            
            filter_choice = Prompt.ask("Select filter type", choices=["1", "2", "3", "4", "5", "6"])
            filter_type = filter_options[int(filter_choice) - 1]
            
            filter_value = Prompt.ask(f"Enter value to filter {col_name} {filter_type}")
            
            with Progress(SpinnerColumn(), TextColumn("[green]Applying filter...[/green]")) as progress:
                task = progress.add_task("", total=None)
                
                if filter_choice == "1":  # Equal to
                    result = self.df.filter(self.df[col_name] == filter_value)
                elif filter_choice == "2":  # Not equal to
                    result = self.df.filter(self.df[col_name] != filter_value)
                elif filter_choice == "3":  # Greater than
                    result = self.df.filter(self.df[col_name] > float(filter_value))
                elif filter_choice == "4":  # Less than
                    result = self.df.filter(self.df[col_name] < float(filter_value))
                elif filter_choice == "5":  # Contains
                    result = self.df.filter(self.df[col_name].contains(filter_value))
                elif filter_choice == "6":  # Starts with
                    result = self.df.filter(self.df[col_name].startswith(filter_value))
                
            self.console.print(f"[bold green]Filter applied: {col_name} {filter_type} {filter_value}[/bold green]")
            self._display_dataframe(result, 10)
            
            if Confirm.ask("Save filtered data as current dataframe?"):
                self.df = result
                self.console.print("[bold green]Filtered data saved as current dataframe[/bold green]")
            
        except Exception as e:
            self.console.print(f"[bold red]Error applying filter:[/bold red] {str(e)}")
    
    def save_data(self):
        """Save the current dataframe."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Save Data[/bold]", border_style="yellow"))
        
        # Create processed directory if it doesn't exist
        processed_dir = self.data_dir / "processed"
        if not processed_dir.exists():
            processed_dir.mkdir(parents=True)
            
        path = Prompt.ask("Enter save path (or 'd' for default or 'c' to cancel)")
        if path.lower() == 'c':
            return
        elif path.lower() == 'd':
            path = str(processed_dir)

        file_format = Prompt.ask("Choose file format", choices=["csv", "parquet", "json"], default="csv")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        save_name = Prompt.ask("Enter filename (without extension)", default=f"spark_data_{timestamp}")

        try:
            save_path = Path(path) / save_name
            
            with Progress(SpinnerColumn(), TextColumn("[green]Saving data...[/green]")) as progress:
                task = progress.add_task("", total=None)
                
                if file_format == "csv":
                    self.df.repartition(1).write.csv(str(save_path), header=True, mode="errorifexists")
                elif file_format == "parquet":
                    self.df.write.parquet(str(save_path), mode="errorifexists")
                elif file_format == "json":
                    self.df.write.json(str(save_path), mode="errorifexists")
            
            self.console.print(f"[bold green]Data saved to:[/bold green] {save_path}")
            
            # Show filesystem details
            self.console.print("\n[bold cyan]Files created:[/bold cyan]")
            try:
                if save_path.exists():
                    for item in save_path.iterdir():
                        size = f"{item.stat().st_size / 1024:.1f} KB"
                        self.console.print(f"[green]{item.name}[/green] ({size})")
            except:
                self.console.print(f"[yellow]Unable to list directory contents (look for files in {save_path})[/yellow]")
                
        except Exception as e:
            self.console.print(f"[bold red]Error saving data:[/bold red] {str(e)}")
    
    def show_help(self):
        """Display help information."""
        help_text = """
# SparkDataApp Help

## Available Commands

* `l` - **Load data**: Load a CSV file into a Spark DataFrame
* `/` - **Query data**: Filter data by column values
* `v` - **Visualize data**: View basic statistics (advanced visualization coming soon)
* `f` - **Filter data**: Apply custom filters to your data
* `s` - **Save data**: Export data to various formats
* `h` - **Help**: Show this help message
* `q` - **Quit**: Exit the application

## Tips

* When loading data, the application will scan the data directory for CSV files
* Use query and filter to narrow down your dataset
* Statistics and basic data information are available in the visualize menu
* Save your results in various formats for later use
        """
        
        self.console.print(Markdown(help_text))
    
    def quit(self):
        """Stop the Spark session and exit."""
        with Progress(SpinnerColumn(), TextColumn("[green]Stopping Spark session...[/green]")) as progress:
            task = progress.add_task("", total=None)
            stop_spark_session(self.session)
        
        self.console.print(Panel("[bold blue]Goodbye![/bold blue]", border_style="green"))
    
    def _display_dataframe(self, df: DataFrame, num_rows: int = 5):
        """
        Display dataframe as a Rich table.
        
        Args:
            df: PySpark DataFrame to display
            num_rows: Number of rows to display
        """
        schema = df.schema
        data = df.limit(num_rows).collect()
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        
        # Add columns (limit to 10 if more exist)
        fields = schema.fields
        max_cols = 10
        total_cols = len(fields)
        display_fields = fields[:max_cols] if total_cols > max_cols else fields
        
        # Add columns to table
        for field in display_fields:
            table.add_column(field.name, style="green")
            
        # If we have more than max_cols columns, add a note column
        if total_cols > max_cols:
            table.add_column("...", style="dim")
        
        # Add rows
        for row in data:
            # Get values for visible columns
            visible_values = [str(row[i]) for i in range(min(max_cols, total_cols))]
            
            # If we have truncated columns, add an indicator
            if total_cols > max_cols:
                visible_values.append(f"({total_cols - max_cols} more)")
                
            table.add_row(*visible_values)
            
        self.console.print(table)
        
        # Show information about truncation
        info_parts = []
        if num_rows < df.count():
            info_parts.append(f"Showing {min(num_rows, df.count())} of {df.count()} rows")
        if total_cols > max_cols:
            info_parts.append(f"Showing {max_cols} of {total_cols} columns")
            
        if info_parts:
            self.console.print(f"[italic]({' | '.join(info_parts)})[/italic]")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SparkDataApp - A console app for PySpark data manipulation")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing data files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--app-name", 
        type=str, 
        default="SparkDataApp",
        help="Name for the Spark application (default: SparkDataApp)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = SparkDataConsoleApp(
        data_dir=Path(args.data_dir),
        app_name=args.app_name
    )
    app.start()
