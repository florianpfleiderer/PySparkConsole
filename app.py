# Created on Wed Feb 05 2025 by 240030614
# Copyright (c) 2025 University of St. Andrews
"""
Textual app for PySpark data manipulation, styled similarly to Harlequin.

Place this file (app.py) in the same directory as textual.css.
Some versions of Textual auto-load textual.css; if not, see "Manual CSS Loading" below.
"""

from __future__ import annotations
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import box

from src.spark_session import create_spark_session, stop_spark_session
from datetime import datetime

DATA_DIR = Path("data/")

class SparkDataConsoleApp:
    """A console app for PySpark data manipulation."""

    def __init__(self):
        self.session = None
        self.df = None
        self.console = Console()
        
    def start(self):
        """Start the application."""
        self.console.print(Panel("[bold blue]Welcome to SparkDataApp[/bold blue]", 
                                border_style="green"))
        
        with Progress(SpinnerColumn(), TextColumn("[green]Creating Spark session...[/green]")) as progress:
            task = progress.add_task("", total=None)
            self.session = create_spark_session("SparkDataApp")
        
        self.run_main_loop()
    
    def run_main_loop(self):
        """Run the main application loop."""
        while True:
            menu = """
[bold cyan]Available commands:[/bold cyan]
[yellow]l[/yellow] - Load data
[yellow]/[/yellow] - Query data
[yellow]s[/yellow] - Save data
[yellow]q[/yellow] - Quit
            """
            self.console.print(Panel(menu, title="Menu", border_style="blue"))
            
            choice = Prompt.ask("Enter command", choices=["l", "/", "s", "q"], show_choices=False)
            
            if choice == 'l':
                self.load_data()
            elif choice == '/':
                self.query_data()
            elif choice == 's':
                self.save_data()
            elif choice == 'q':
                self.quit()
                break
    
    def load_data(self):
        """Load data from a CSV file."""
        self.console.print(Panel("[bold]Load Data[/bold]", border_style="yellow"))
        try:
            csv_files = list(DATA_DIR.rglob('*.csv'))
            
            if not csv_files:
                self.console.print("[bold red]No CSV files found in data/raw directory[/bold red]")
                return
                
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=6)
            table.add_column("Filename", style="green")
            
            for i, file in enumerate(csv_files, 1):
                table.add_row(str(i), file.name)
                
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
                    
                self.console.print(f"[bold green]Loaded:[/bold green] {file_path.name}")
                self._display_dataframe(self.df, 5)
                
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
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim", width=6)
        table.add_column("Column", style="green")
        table.add_column("Description", style="blue")
        
        table.add_row("1", "la_name", "Local Authority")
        table.add_row("2", "school_type", "School Type")
        
        self.console.print(table)
        
        choice = Prompt.ask("Select column", choices=["1", "2"])
        column = None
        
        if choice == "1":
            column = "la_name"
        elif choice == "2":
            column = "school_type"
        
        with Progress(SpinnerColumn(), TextColumn("[green]Getting values...[/green]")) as progress:
            task = progress.add_task("", total=None)
            values = [row[0] for row in self.df.select(column).distinct().collect()]
        
        values_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        values_table.add_column("#", style="dim", width=6)
        values_table.add_column(f"{column} Values", style="green")
        
        for i, val in enumerate(values, 1):
            values_table.add_row(str(i), str(val))
            
        self.console.print(values_table)
            
        val_choice = Prompt.ask("Select value number to filter by")
        try:
            index = int(val_choice) - 1
            selected_val = values[index]
            
            with Progress(SpinnerColumn(), TextColumn("[green]Filtering data...[/green]")) as progress:
                task = progress.add_task("", total=None)
                result = self.df.filter(self.df[column] == selected_val)
            
            self.console.print(f"[bold green]Results for {column} = {selected_val}:[/bold green]")
            self._display_dataframe(result, 10)
            
        except (ValueError, IndexError):
            self.console.print("[bold red]Invalid selection[/bold red]")
    
    def save_data(self):
        """Save the current dataframe."""
        if self.df is None:
            self.console.print("[bold red]No data loaded. Please load data first.[/bold red]")
            return
            
        self.console.print(Panel("[bold]Save Data[/bold]", border_style="yellow"))
        path = Prompt.ask("Enter save path (or 'd' for default or 'c' to cancel)")
        if path.lower() == 'c':
            return
        elif path.lower() == 'd':
            path = "data/processed"

        try:
            save_path = Path(path) / Path(datetime.now().strftime('%Y-%m-%d_%H-%M'))
            with Progress(SpinnerColumn(), TextColumn("[green]Saving data...[/green]")) as progress:
                task = progress.add_task("", total=None)
                self.df.repartition(1).write.csv(str(save_path), header=True, mode="errorifexists")
            
            self.console.print(f"[bold green]Data saved to:[/bold green] {save_path}")
        except Exception as e:
            self.console.print(f"[bold red]Error saving data:[/bold red] {str(e)}")
    
    def quit(self):
        """Stop the Spark session and exit."""
        with Progress(SpinnerColumn(), TextColumn("[green]Stopping Spark session...[/green]")) as progress:
            task = progress.add_task("", total=None)
            stop_spark_session(self.session)
        
        self.console.print(Panel("[bold blue]Goodbye![/bold blue]", border_style="green"))
    
    def _display_dataframe(self, df, num_rows=5):
        """Display dataframe as a Rich table."""
        schema = df.schema
        data = df.limit(num_rows).collect()
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        
        # Add columns
        for field in schema.fields:
            table.add_column(field.name, style="green")
        
        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row])
            
        self.console.print(table)
        self.console.print(f"[italic](Showing {min(num_rows, df.count())} of {df.count()} rows)[/italic]")

if __name__ == "__main__":
    app = SparkDataConsoleApp()
    app.start()
