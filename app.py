# Created on Wed Feb 05 2025 by 240030614
"""
PySpark data manipulation.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType

from src.utils.spark_utils import (
    create_spark_session,
    stop_spark_session,
    get_active_session
)
from src.utils.data_handling import (
    load_csv_data,
    handle_null_values,
    find_csv_files,
    create_file_table,
    create_null_value_table,
    save_dataframe,
    get_save_info
)
from src.utils.display_utils import (
    create_progress,
    display_dataframe_preview,
    create_menu_table,
    create_status_panel
)
from src.queries import (
    handle_local_authority_query,
    handle_school_type_query,
    handle_unauthorized_absences_query,
    analyze_absence_patterns
)
from src.visualisations import (
    display_numeric_statistics,
    analyze_regional_attendance,
    create_absence_pattern_plots
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box
from rich.markdown import Markdown
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler for better error display
install_rich_traceback()

# Configure logging
logging.basicConfig(level=logging.ERROR)

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
        
    def start(self) -> None:
        """Start the application."""
        self.console.print(Panel(
            "[bold blue]Welcome to SparkDataApp[/bold blue]\n"
            "[cyan]A console-based PySpark data analysis tool[/cyan]", 
            border_style="green"
        ))
        
        with Progress(SpinnerColumn(), TextColumn("[green]Creating Spark session...[/green]")) as progress:
            task = progress.add_task("", total=None)
            try:
                self.session = create_spark_session(self.app_name, log_level="ERROR")
                progress.update(task, description="[green]Spark session created successfully![/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error creating Spark session:[/bold red] {str(e)}")
                sys.exit(1)

        # Immediately prompt to load data
        self.load_data()
        
        self.run_main_loop()
    
    def run_main_loop(self) -> None:
        """Run the main application loop."""
        while True:
            # Show status bar with current data status
            self._show_status()
            
            menu = """
[bold cyan]Available commands:[/bold cyan]
[yellow]l[/yellow] - Load data
[yellow]q[/yellow] - Query data
[yellow]v[/yellow] - Visualize data
[yellow]f[/yellow] - Filter data
[yellow]s[/yellow] - Save data
[yellow]h[/yellow] - Help
[yellow]x[/yellow] - Quit
            """
            self.console.print(Panel(menu, title="Menu", border_style="blue"))
            
            choice = Prompt.ask("Enter command", choices=["l", "q", "v", "f", "s", "h", "x"], show_choices=False)
            self.command_history.append(choice)
            
            try:
                if choice == 'l':
                    self.load_data()
                elif choice == 'q':
                    self.query_data()
                elif choice == 'v':
                    self.visualize_data()
                elif choice == 'f':
                    self.filter_data()
                elif choice == 's':
                    self.save_data()
                elif choice == 'h':
                    self.show_help()
                elif choice == 'x':
                    self.quit()
                    break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                self.console.print("[dim]Use 'h' for help or 'x' to quit[/dim]")
                
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
            # Find CSV files and directories
            csv_files, processed_dirs = find_csv_files(self.data_dir)
            
            if not csv_files and not processed_dirs:
                self.console.print(
                    f"[bold red]No CSV files found in {self.data_dir} "
                    f"directory[/bold red]"
                )
                if Confirm.ask("Would you like to specify a different data directory?"):
                    dir_path = Prompt.ask("Enter path to data directory")
                    self.data_dir = Path(dir_path)
                    self.load_data()  # Retry with new directory
                return
            
            # Display file table
            table = create_file_table(csv_files, processed_dirs)
            self.console.print(table)
            
            # Get user selection
            choice = Prompt.ask("Select number (or 'c' to cancel)")
            if choice.lower() == 'c':
                return
            
            try:
                index = int(choice) - 1
                total_items = len(csv_files) + len(processed_dirs)
                
                if index < 0 or index >= total_items:
                    raise ValueError("Invalid selection number")
                
                # Get selected file info
                if index < len(csv_files):
                    file_info = csv_files[index]
                else:
                    file_info = processed_dirs[index - len(csv_files)]
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[green]Loading data...[/green]")
                ) as progress:
                    task = progress.add_task("", total=None)
                    
                    # Load data using data_loader module
                    self.df, null_counts = load_csv_data(
                        self.session,
                        file_info.path,
                        file_info.is_directory
                    )
                    
                    progress.update(
                        task,
                        description="[green]Data loaded, checking for null values...[/green]"
                    )
                    
                    # Create null value table
                    null_table, cols_with_nulls = create_null_value_table(
                        self.df,
                        null_counts
                    )
                    
                    row_count = self.df.count()
                    progress.update(
                        task,
                        description=f"[green]Loaded {row_count} rows successfully![/green]"
                    )
                
                if null_table:
                    self.console.print("\n[bold red]Warning: Null values detected in the dataset![/bold red]")
                    self.console.print(null_table)
                    
                    # Show handling information
                    self.console.print("\n[bold]Null values will be handled as follows:[/bold]")
                    self.console.print("• Text columns: Replace with 'MISSING'")
                    self.console.print("• Numeric columns: Replace with -1")
                    self.console.print("• Special values (':' characters): Convert to -1")
                    
                    if Confirm.ask(
                        "\nWould you like to handle the null values as described above?",
                        default=False
                    ):
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[cyan]Handling null values...[/cyan]")
                        ) as handle_progress:
                            handle_task = handle_progress.add_task("", total=None)
                            
                            # Handle null values
                            self.df, remaining_nulls = handle_null_values(
                                self.df,
                                null_counts
                            )
                            
                            handle_progress.update(
                                handle_task,
                                description="[green]Completed null value handling![/green]"
                            )
                        
                        self.console.print("\n[bold green]✓ Null values have been handled:[/bold green]")
                        self.console.print("• String columns filled with 'MISSING'")
                        self.console.print("• Numeric columns filled with -1")
                        self.console.print("• Special characters (':') replaced with -1")
                        
                        if any(count > 0 for count in remaining_nulls.values()):
                            self.console.print(
                                "\n[yellow]Note: Some null values could not be handled "
                                "automatically.[/yellow]"
                            )
                        else:
                            self.console.print(
                                "\n[bold green]✓ All null values have been successfully "
                                "handled![/bold green]"
                            )
                
                self.console.print(
                    f"\n[bold green]Loaded:[/bold green] {file_info.name} "
                    f"({row_count:,} rows)"
                )
                
            except (ValueError, IndexError):
                self.console.print("[bold red]Invalid selection[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")

    def query_data(self):
        """Query the dataframe by local authority or school type."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Query Data[/bold]", border_style="yellow"))
        
        # Display query options
        query_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        query_table.add_column("#", style="dim", width=6)
        query_table.add_column("Query Type", style="green")
        
        query_options = ["Local Authority", "School Type", "Unauthorized Absences"]
        for i, option in enumerate(query_options, 1):
            query_table.add_row(str(i), option)
            
        self.console.print(query_table)
        
        choice = Prompt.ask("Select query type", choices=["1", "2", "3"])
        
        try:
            if choice == "1":  # Local Authority
                result, save_result = handle_local_authority_query(self.df, self.console)
            elif choice == "2":  # School Type
                result, save_result = handle_school_type_query(self.df, self.console)
            else:  # Unauthorized Absences
                result, save_result = handle_unauthorized_absences_query(self.df, self.console)
            
            self.last_query_result = result
            
            # Ask if user wants to save this query result
            if save_result and Confirm.ask("Would you like to save this query result?"):
                self.df = result
                self.console.print("[bold green]Query result saved as current dataframe[/bold green]")
                
        except Exception as e:
            self.console.print(f"[bold red]Error during query:[/bold red] {str(e)}")
    
    def visualize_data(self):
        """Visualize data using matplotlib."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
        
        self.console.print(Panel("[bold]Visualize Data[/bold]", border_style="yellow"))
        
        # Create visualization options table
        viz_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        viz_table.add_column("#", style="dim", width=6)
        viz_table.add_column("Visualization Type", style="green")
        viz_table.add_column("Description", style="blue")
        
        # Available visualization options
        viz_options = [
            ("Regional Performance", "Analyze regional attendance trends over time"),
            ("Absences / School Type / Location", "Analyze relationships between school types, locations, and absence rates"),
            ("Basic Statistics", "View summary statistics for numeric columns")
        ]
        
        # Add options to table
        for i, (viz_type, description) in enumerate(viz_options, 1):
            viz_table.add_row(str(i), viz_type, description)
        
        self.console.print(viz_table)
        
        try:
            # Get user choice
            choice = Prompt.ask(
                "Select visualization type",
                choices=["1", "2", "3"]
            )
            
            if choice == "1":
                # Regional Performance Analysis
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[green]Analyzing regional performance...[/green]")
                ) as progress:
                    task = progress.add_task("", total=None)
                    success = analyze_regional_attendance(self.df, self.console)
                    
            elif choice == "2":
                # Handle School Type Location Analysis
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[green]Analyzing absence patterns...[/green]")
                ) as progress:
                    task = progress.add_task("", total=None)
                    
                    # Perform analysis and create visualizations
                    result_df, _ = analyze_absence_patterns(self.df, self.console)
                    success = create_absence_pattern_plots(result_df, self.console)
            else:
                # Basic Statistics
                display_numeric_statistics(self.df, self.console)
                success = True
            
            if not success:
                self.console.print(
                    "[yellow]Unable to create visualization. "
                    "Please check your data and try again.[/yellow]"
                )
                
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")

    def filter_data(self):
        """Filter data with custom conditions."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Filter Data[/bold]", border_style="yellow"))
        
        # Display column information for first 20 columns only
        col_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        col_table.add_column("#", style="dim", width=6)
        col_table.add_column("Column", style="green")
        col_table.add_column("Data Type", style="blue")
        
        # Get first 20 columns
        fields = self.df.schema.fields[:20]  # Only first 20 columns
        for i, field in enumerate(fields, 1):
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
            col_choice = Prompt.ask("Select column to filter (1-20)")
            col_index = int(col_choice) - 1
            
            if col_index < 0 or col_index >= 20:  # Only allow first 20 columns
                raise ValueError("Invalid column selection. Please choose a number between 1 and 20.")
                
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
            
            # Show info about partitioned output
            self.console.print("\n[bold yellow]Note about saving:[/bold yellow]")
            self.console.print(get_save_info(file_format))
            
            with Progress(SpinnerColumn(), TextColumn("[green]Saving data...[/green]")) as progress:
                task = progress.add_task("", total=None)
                
                result = save_dataframe(
                    self.df,
                    save_path,
                    file_format=file_format,
                    mode="errorifexists"
                )
                
                if result["success"]:
                    self.console.print(f"[bold green]Data saved to:[/bold green] {result['path']}")
                    
                    # Show filesystem details
                    if result["files"]:
                        self.console.print("\n[bold cyan]Files created:[/bold cyan]")
                        for file_info in result["files"]:
                            self.console.print(
                                f"[green]{file_info['name']}[/green] "
                                f"({file_info['size']:.1f} KB)"
                            )
                else:
                    self.console.print(f"[bold red]Error saving data:[/bold red] {result['error']}")
                
        except Exception as e:
            self.console.print(f"[bold red]Error saving data:[/bold red] {str(e)}")
    
    def show_help(self):
        """Display help information."""
        help_text = """
# SparkDataApp Help

## Available Commands

* `l` - **Load data**: Load a CSV file into a Spark DataFrame
* `q` - **Query data**: Filter data by column values
* `v` - **Visualize data**: View basic statistics (advanced visualization coming soon)
* `f` - **Filter data**: Apply custom filters to your data
* `s` - **Save data**: Export data to various formats
* `h` - **Help**: Show this help message
* `x` - **Quit**: Exit the application

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
    
    def _display_dataframe(self, df: DataFrame, num_rows: int = 5, num_cols: int = 6):
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
        max_cols = num_cols
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
