# Created on Wed Feb 05 2025 by 240030614
"""
PySpark data manipulation.
"""

from __future__ import annotations
from pathlib import Path
import argparse
# import os
import sys
import logging
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
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType

from src.spark_session import create_spark_session, stop_spark_session
from datetime import datetime
import traceback

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
            # First, find all CSV files and directories that might contain CSV files
            csv_files = []
            processed_dirs = []
            
            # Handle data/processed directory specially
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                # Look for directories containing CSV files in processed/
                for item in processed_dir.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        # Check if directory contains CSV files (typical for Spark partitioned output)
                        has_csv = any(f.suffix == '.csv' and not f.name.startswith('.') for f in item.iterdir())
                        if has_csv:
                            processed_dirs.append(item)
            
            # Find CSV files in all directories except processed/
            for item in self.data_dir.rglob('*.csv'):
                # Skip files in processed/ directory and hidden files
                if not item.is_file() or item.name.startswith('.') or 'processed' in item.parts:
                    continue
                csv_files.append(item)
            
            if not csv_files and not processed_dirs:
                self.console.print(f"[bold red]No CSV files found in {self.data_dir} directory[/bold red]")
                if Confirm.ask("Would you like to specify a different data directory?"):
                    dir_path = Prompt.ask("Enter path to data directory")
                    self.data_dir = Path(dir_path)
                    self.load_data()  # Retry with new directory
                return
                
            # Create table for display
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="dim", width=6)
            table.add_column("Type", style="yellow")
            table.add_column("Location", style="blue")
            table.add_column("Name", style="green")
            table.add_column("Size", style="cyan")
            table.add_column("Last Modified", style="magenta")
            
            # Add regular CSV files
            for i, file in enumerate(csv_files, 1):
                size = f"{file.stat().st_size / 1024:.1f} KB"
                modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                # Get relative path from data directory
                rel_path = file.relative_to(self.data_dir)
                location = str(rel_path.parent)
                if location == '.':
                    location = 'data'
                table.add_row(str(i), "File", location, file.name, size, modified)
            
            # Add directories containing partitioned CSVs
            for i, dir_path in enumerate(processed_dirs, len(csv_files) + 1):
                # Calculate total size of CSV files in directory
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*.csv') if not f.name.startswith('.'))
                size = f"{total_size / 1024:.1f} KB"
                modified = datetime.fromtimestamp(dir_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                table.add_row(str(i), "Directory", "processed", dir_path.name, size, modified)
                
            self.console.print(table)
                
            choice = Prompt.ask("Select number (or 'c' to cancel)")
            if choice.lower() == 'c':
                return
                
            try:
                index = int(choice) - 1
                total_items = len(csv_files) + len(processed_dirs)
                
                if index < 0 or index >= total_items:
                    raise ValueError("Invalid selection number")
                
                # Determine if selection is a file or directory
                if index < len(csv_files):
                    file_path = csv_files[index]
                    is_directory = False
                else:
                    file_path = processed_dirs[index - len(csv_files)]
                    is_directory = True
                
                with Progress(SpinnerColumn(), TextColumn("[green]Loading data...[/green]")) as progress:
                    task = progress.add_task("", total=None)
                    
                    if is_directory:
                        # For directories, read all CSV files as one DataFrame
                        self.df = self.session.read.csv(str(file_path), header=True, inferSchema=True)
                    else:
                        # For single files, read normally
                        self.df = self.session.read.csv(str(file_path), header=True, inferSchema=True)
                        
                    progress.update(task, description="[green]Data loaded, checking for null values...[/green]")
                    
                    # Check for null values in each column
                    null_counts = {col: int(self.df.filter(F.col(col).isNull()).count()) 
                                 for col in self.df.columns}
                    
                    # Filter out columns with no null values
                    cols_with_nulls = {col: count for col, count in null_counts.items() if count > 0}
                    
                    row_count = self.df.count()
                    progress.update(task, description=f"[green]Loaded {row_count} rows successfully![/green]")
                
                if cols_with_nulls:
                    # Create a table to display null value information
                    null_table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
                    null_table.add_column("Column", style="yellow")
                    null_table.add_column("Data Type", style="magenta")
                    null_table.add_column("Null Count", style="cyan")
                    null_table.add_column("% of Total", style="green")
                    
                    # Create a dictionary of column types
                    col_types = {field.name: str(field.dataType) for field in self.df.schema.fields}
                    
                    for col, count in cols_with_nulls.items():
                        percentage = (count / row_count) * 100
                        null_table.add_row(
                            col,
                            col_types[col],
                            str(count),
                            f"{percentage:.2f}%"
                        )
                    
                    self.console.print("\n[bold red]Warning: Null values detected in the dataset![/bold red]")
                    self.console.print(null_table)
                    
                    # Show handling information
                    self.console.print("\n[bold]Null values will be handled as follows:[/bold]")
                    self.console.print("• Text columns: Replace with 'MISSING'")
                    self.console.print("• Numeric columns: Replace with -1")
                    self.console.print("• Special values (':' characters): Convert to -1")
                    
                    # Simple yes/no prompt
                    if Confirm.ask("\nWould you like to handle the null values as described above?", default=False):
                        # Identify string and numeric columns
                        string_cols = [col for col in self.df.schema.fields if isinstance(col.dataType, StringType)]
                        string_col_names = [col.name for col in string_cols]
                        numeric_col_names = [col.name for col in self.df.schema.fields if col.name not in string_col_names]
                        
                        # Fill null values in a new progress context
                        with Progress(SpinnerColumn(), TextColumn("[cyan]Handling null values...[/cyan]")) as handle_progress:
                            handle_task = handle_progress.add_task("", total=None)
                            
                            # Fill string columns with 'MISSING'
                            self.df = self.df.fillna('MISSING', subset=string_col_names)
                            handle_progress.update(handle_task, description="[cyan]Filled string columns...[/cyan]")
                            
                            # Fill numeric columns with -1
                            self.df = self.df.fillna(-1, subset=numeric_col_names)
                            handle_progress.update(handle_task, description="[cyan]Filled numeric columns...[/cyan]")
                            
                            # Special handling for any columns that need type conversion
                            special_cols = self.df.select([col for col in self.df.columns if ':' in str(self.df.select(col).first()[0])])
                            for special_col in special_cols.columns:
                                self.df = self.df.replace(":", "-1", subset=special_col)
                                if special_col in numeric_col_names:
                                    self.df = self.df.withColumn(special_col, F.col(special_col).cast(IntegerType()))
                            handle_progress.update(handle_task, description="[cyan]Handled special columns...[/cyan]")
                            
                            # Final verification
                            remaining_nulls = {col: int(self.df.filter(F.col(col).isNull()).count()) 
                                            for col in self.df.columns}
                            handle_progress.update(handle_task, description="[green]Completed null value handling![/green]")
                        
                        self.console.print("\n[bold green]✓ Null values have been handled:[/bold green]")
                        self.console.print("• String columns filled with 'MISSING'")
                        self.console.print("• Numeric columns filled with -1")
                        self.console.print("• Special characters (':') replaced with -1")
                        
                        if any(count > 0 for count in remaining_nulls.values()):
                            self.console.print("\n[yellow]Note: Some null values could not be handled automatically.[/yellow]")
                        else:
                            self.console.print("\n[bold green]✓ All null values have been successfully handled![/bold green]")
                
                self.console.print(f"\n[bold green]Loaded:[/bold green] {file_path.name} ({row_count:,} rows)")
                
            except (ValueError, IndexError):
                self.console.print("[bold red]Invalid selection[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def _display_enrollment_stats(self, df: DataFrame, authorities: List[str]):
        """
        Display enrollment statistics by year for selected local authorities.
        
        Args:
            df: PySpark DataFrame containing the data
            authorities: List of selected local authority names
        """
        # Create pivot table of enrollments by LA and year
        pivot_df = df.groupBy("la_name").pivot("time_period").agg(
            F.sum("enrolments").alias("enrollments")
        ).orderBy("la_name")
        
        # Get all years in order
        years = sorted([col for col in pivot_df.columns if col != "la_name"])
        
        # Create formatted table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title="Pupil Enrollments by Local Authority and Year"
        )
        
        # Add columns
        table.add_column("Local Authority", style="green")
        for year in years:
            table.add_column(str(year), justify="right", style="blue")
        
        # Add total column
        table.add_column("Total", justify="right", style="yellow")
        
        # Collect data and format rows
        rows = pivot_df.collect()
        for row in rows:
            if row["la_name"] in authorities:
                values = [row["la_name"]]
                total = 0
                for year in years:
                    value = row[year] if row[year] is not None else 0
                    total += value
                    values.append(f"{value:,}")
                values.append(f"{total:,}")
                table.add_row(*values)
        
        self.console.print(table)
        
        # Show summary statistics
        summary = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            title="Summary Statistics"
        )
        summary.add_column("Year", style="green")
        summary.add_column("Total Enrollments", justify="right", style="blue")
        summary.add_column("Average per LA", justify="right", style="yellow")
        
        for year in years:
            total = sum(row[year] if row[year] is not None else 0 
                       for row in rows if row["la_name"] in authorities)
            avg = total / len(authorities)
            summary.add_row(
                str(year),
                f"{total:,}",
                f"{avg:,.0f}"
            )
        
        self.console.print("\n")
        self.console.print(summary)

    def _display_absence_stats(self, df: DataFrame, school_type: str):
        """
        Display authorized absence statistics for the selected school type by year.
        
        Args:
            df: PySpark DataFrame containing the data
            school_type: Selected school type
        """
        # Get available years
        years = sorted([row[0] for row in df.select("time_period").distinct().collect()])
        
        # Create table for displaying statistics
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title=f"Authorized Absences Summary for {school_type}"
        )
        
        table.add_column("Year", style="green")
        table.add_column("Total Authorized Absences", justify="right", style="blue")
        table.add_column("Total Enrollments", justify="right", style="blue")
        table.add_column("Absences per Student", justify="right", style="yellow")
        
        # Calculate statistics for each year
        for year in years:
            year_data = df.filter(F.col("time_period") == year)
            
            # Calculate totals
            stats = year_data.agg(
                F.sum("sess_authorised").alias("total_absences"),
                F.sum("enrolments").alias("total_enrollments")
            ).collect()[0]
            
            total_absences = stats["total_absences"] or 0
            total_enrollments = stats["total_enrollments"] or 0
            
            # Calculate absences per student
            absences_per_student = (
                total_absences / total_enrollments if total_enrollments > 0 else 0
            )
            
            table.add_row(
                str(year),
                f"{total_absences:,}",
                f"{total_enrollments:,}",
                f"{absences_per_student:.2f}"
            )
        
        self.console.print(table)
        
        # Ask if user wants to see detailed breakdown
        if Confirm.ask("\nWould you like to see a detailed breakdown of absence types?"):
            # Let user select a year
            year_table = Table(show_header=True, header_style="bold magenta")
            year_table.add_column("#", style="dim", width=6)
            year_table.add_column("Year", style="green")
            
            for i, year in enumerate(years, 1):
                year_table.add_row(str(i), str(year))
                
            self.console.print("\nSelect a year for detailed breakdown:")
            self.console.print(year_table)
            
            year_choice = Prompt.ask("Enter year number")
            try:
                selected_year = years[int(year_choice) - 1]
                
                # Define absence types and their descriptions
                absence_types = {
                    "sess_auth_appointments": "Medical appointments",
                    "sess_auth_holiday": "Authorised holiday",
                    "sess_auth_illness": "Illness",
                    "sess_auth_other": "Other authorised",
                    "sess_auth_religious": "Religious observance",
                    "sess_auth_study": "Study leave",
                    "sess_auth_traveller": "Traveller"
                }
                
                # Filter data for selected year
                year_data = df.filter(F.col("time_period") == selected_year)
                
                # Create detailed breakdown table
                detail_table = Table(
                    show_header=True,
                    header_style="bold magenta",
                    box=box.ROUNDED,
                    title=f"Detailed Absence Breakdown for {school_type} in {selected_year}"
                )
                
                detail_table.add_column("Absence Type", style="green")
                detail_table.add_column("Total Sessions", justify="right", style="blue")
                detail_table.add_column("% of All Authorized", justify="right", style="yellow")
                detail_table.add_column("Sessions per Student", justify="right", style="cyan")
                
                # Calculate totals for percentage calculation
                totals = year_data.agg(
                    F.sum("sess_authorised").alias("total_auth"),
                    F.sum("enrolments").alias("total_enrol")
                ).collect()[0]
                
                total_authorized = totals["total_auth"] or 0
                total_enrollments = totals["total_enrol"] or 0
                
                # Calculate statistics for each absence type
                for col, description in absence_types.items():
                    stats = year_data.agg(F.sum(col).alias("total")).collect()[0]
                    total = stats["total"] or 0
                    
                    percentage = (
                        (total / total_authorized * 100)
                        if total_authorized > 0 else 0
                    )
                    
                    per_student = (
                        total / total_enrollments
                        if total_enrollments > 0 else 0
                    )
                    
                    detail_table.add_row(
                        description,
                        f"{total:,}",
                        f"{percentage:.1f}%",
                        f"{per_student:.2f}"
                    )
                
                self.console.print("\n")
                self.console.print(detail_table)
                
                # Add summary note
                self.console.print(
                    f"\n[dim]Total students: {total_enrollments:,} | "
                    f"Total authorized absences: {total_authorized:,}[/dim]"
                )
                
            except (ValueError, IndexError):
                self.console.print("[bold red]Invalid year selection[/bold red]")

    def _display_unauth_absence_stats(self, df: DataFrame, breakdown_by: str, year: str):
        """
        Display unauthorized absence statistics broken down by region or local authority.
        
        Args:
            df: PySpark DataFrame containing the data
            breakdown_by: Either 'region_name' or 'la_name'
            year: Selected year
        """
        # Filter for selected year
        year_data = df.filter(F.col("time_period") == year)
        
        # Group by region/LA and calculate statistics
        stats = (year_data.groupBy(breakdown_by)
                .agg(
                    F.sum("sess_unauthorised").alias("total_unauth"),
                    F.sum("enrolments").alias("total_students")
                )
                .orderBy(breakdown_by))
        
        # Calculate overall totals for percentages
        totals = stats.agg(
            F.sum("total_unauth").alias("total_unauth"),
            F.sum("total_students").alias("total_students")
        ).collect()[0]
        
        overall_unauth = totals["total_unauth"] or 0
        overall_students = totals["total_students"] or 0
        
        # Create table for displaying statistics
        title = "Regions" if breakdown_by == "region_name" else "Local Authorities"
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title=f"Unauthorized Absences by {title} in {year}"
        )
        
        table.add_column(title.rstrip('s'), style="green")
        table.add_column("Total Unauthorized", justify="right", style="blue")
        table.add_column("Total Students", justify="right", style="blue")
        table.add_column("% of All Unauthorized", justify="right", style="yellow")
        table.add_column("Sessions per Student", justify="right", style="cyan")
        
        # Add rows
        for row in stats.collect():
            area = row[breakdown_by]
            unauth = row["total_unauth"] or 0
            students = row["total_students"] or 0
            
            percentage = (
                (unauth / overall_unauth * 100)
                if overall_unauth > 0 else 0
            )
            
            per_student = (
                unauth / students
                if students > 0 else 0
            )
            
            table.add_row(
                str(area),
                f"{unauth:,}",
                f"{students:,}",
                f"{percentage:.1f}%",
                f"{per_student:.2f}"
            )
        
        self.console.print(table)
        
        # Show summary
        self.console.print(
            f"\n[dim]Total unauthorized absences: {overall_unauth:,} | "
            f"Total students: {overall_students:,} | "
            f"Overall sessions per student: {(overall_unauth/overall_students if overall_students > 0 else 0):.2f}[/dim]"
        )

    def _compare_local_authorities(self, df: DataFrame, auth1: str, auth2: str, year: str):
        """
        Compare two local authorities for a given year across multiple metrics.
        
        Args:
            df: PySpark DataFrame containing the data
            auth1: First local authority name
            auth2: Second local authority name
            year: Selected year for comparison
        """
        try:
            # Filter data for the selected year and authorities
            comparison_data = df.filter(
                (F.col("time_period") == year) & 
                (F.col("la_name").isin([auth1, auth2]))
            )
            
            # Check if we have data for both authorities
            auth_counts = comparison_data.groupBy("la_name").count().collect()
            auth_count_dict = {row["la_name"]: row["count"] for row in auth_counts}
            
            missing_auths = []
            if auth1 not in auth_count_dict:
                missing_auths.append(auth1)
            if auth2 not in auth_count_dict:
                missing_auths.append(auth2)
                
            if missing_auths:
                self.console.print(f"[bold red]No data found for the following authorities in {year}:[/bold red]")
                for auth in missing_auths:
                    self.console.print(f"• {auth}")
                return
            
            # Create comparison table
            table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                title=f"Comparison of {auth1} vs {auth2} in {year}"
            )
            
            table.add_column("Metric", style="green")
            table.add_column(auth1, justify="right", style="blue")
            table.add_column(auth2, justify="right", style="yellow")
            table.add_column("Difference", justify="right", style="cyan")
            
            # Calculate metrics for comparison
            metrics = comparison_data.groupBy("la_name").agg(
                F.sum("enrolments").alias("total_enrolments"),
                F.sum("sess_authorised").alias("total_authorised"),
                F.sum("sess_unauthorised").alias("total_unauthorised"),
                F.sum("sess_possible").alias("total_possible")
            ).collect()
            
            # Create a dictionary for easy access to metrics
            metrics_dict = {row["la_name"]: row.asDict() for row in metrics}
            
            if not metrics_dict:
                self.console.print(f"[bold red]No metrics data found for the selected authorities in {year}[/bold red]")
                return
                
            if auth1 not in metrics_dict or auth2 not in metrics_dict:
                self.console.print(f"[bold red]Missing metrics data for one or both authorities in {year}:[/bold red]")
                if auth1 not in metrics_dict:
                    self.console.print(f"• {auth1}")
                if auth2 not in metrics_dict:
                    self.console.print(f"• {auth2}")
                return
            
            # Helper function to calculate percentage
            def calc_percentage(part, whole):
                return (part / whole * 100) if whole > 0 else 0
            
            # Add rows for each metric
            metrics_to_compare = [
                ("Total Enrolments", "total_enrolments", "{:,}"),
                ("Total Authorised Absences", "total_authorised", "{:,}"),
                ("Total Unauthorised Absences", "total_unauthorised", "{:,}"),
                ("Total Possible Sessions", "total_possible", "{:,}")
            ]
            
            for metric_name, metric_key, format_str in metrics_to_compare:
                val1 = metrics_dict[auth1][metric_key]
                val2 = metrics_dict[auth2][metric_key]
                diff = val1 - val2
                
                table.add_row(
                    metric_name,
                    format_str.format(val1),
                    format_str.format(val2),
                    f"{format_str.format(abs(diff))} {'higher' if diff > 0 else 'lower'}"
                )
            
            # Calculate percentage metrics
            for auth in [auth1, auth2]:
                metrics_dict[auth]["auth_absence_rate"] = calc_percentage(
                    metrics_dict[auth]["total_authorised"], 
                    metrics_dict[auth]["total_possible"]
                )
                metrics_dict[auth]["unauth_absence_rate"] = calc_percentage(
                    metrics_dict[auth]["total_unauthorised"], 
                    metrics_dict[auth]["total_possible"]
                )
                metrics_dict[auth]["total_absence_rate"] = (
                    metrics_dict[auth]["auth_absence_rate"] + 
                    metrics_dict[auth]["unauth_absence_rate"]
                )
            
            percentage_metrics = [
                ("Authorised Absence Rate", "auth_absence_rate"),
                ("Unauthorised Absence Rate", "unauth_absence_rate"),
                ("Total Absence Rate", "total_absence_rate")
            ]
            
            for metric_name, metric_key in percentage_metrics:
                val1 = metrics_dict[auth1][metric_key]
                val2 = metrics_dict[auth2][metric_key]
                diff = val1 - val2
                
                table.add_row(
                    metric_name,
                    f"{val1:.1f}%",
                    f"{val2:.1f}%",
                    f"{abs(diff):.1f}% {'higher' if diff > 0 else 'lower'}"
                )
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[bold red]Error during comparison:[/bold red]")
            self.console.print(f"[red]• Selected year: {year}[/red]")
            self.console.print(f"[red]• First authority: {auth1}[/red]")
            self.console.print(f"[red]• Second authority: {auth2}[/red]")
            self.console.print(f"[red]• Error details: {str(e)}[/red]")

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
                column = "la_name"
                title = "Local Authority"
                
                with Progress(SpinnerColumn(), TextColumn("[green]Getting values...[/green]")) as progress:
                    task = progress.add_task("", total=None)
                    values = [row[0] for row in self.df.select(column).distinct().orderBy(column).collect()]
                
                # Ask user what type of analysis they want
                analysis_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                analysis_table.add_column("#", style="dim", width=6)
                analysis_table.add_column("Analysis Type", style="green")
                
                analysis_options = ["Enrollments by Year", "Compare Two Authorities"]
                for i, option in enumerate(analysis_options, 1):
                    analysis_table.add_row(str(i), option)
                    
                self.console.print("\nSelect analysis type:")
                self.console.print(analysis_table)
                
                analysis_choice = Prompt.ask("Enter choice", choices=["1", "2"])
                
                if analysis_choice == "1":  # Enrollments by Year
                    values_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                    values_table.add_column("#", style="dim", width=6)
                    values_table.add_column(f"{title}", style="green")
                    values_table.add_column("Count", style="blue")
                    
                    # Count occurrences of each value
                    value_counts = self.df.groupBy(column).count().orderBy(column).collect()
                    value_count_dict = {row[0]: row[1] for row in value_counts}
                    
                    for i, val in enumerate(values, 1):
                        count = value_count_dict.get(val, 0)
                        values_table.add_row(str(i), str(val), str(count))
                        
                    self.console.print(values_table)
                    
                    # Allow multiple selections for local authorities
                    selected_authorities = []
                    while True:
                        val_choice = Prompt.ask(
                            "Select value number to add (or 'a' for all, 'd' when done, 'c' to cancel)"
                        )
                        
                        if val_choice.lower() == 'c':
                            return
                        elif val_choice.lower() == 'a':
                            selected_authorities = values
                            break
                        elif val_choice.lower() == 'd':
                            if not selected_authorities:
                                self.console.print("[yellow]Please select at least one local authority[/yellow]")
                                continue
                            break
                        else:
                            try:
                                index = int(val_choice) - 1
                                if 0 <= index < len(values):
                                    authority = values[index]
                                    if authority not in selected_authorities:
                                        selected_authorities.append(authority)
                                        self.console.print(f"[green]Added: {authority}[/green]")
                                    else:
                                        self.console.print(f"[yellow]{authority} already selected[/yellow]")
                                else:
                                    self.console.print("[red]Invalid selection number[/red]")
                            except ValueError:
                                self.console.print("[red]Invalid input[/red]")
                    
                    # Filter data for selected authorities
                    result = self.df.filter(F.col(column).isin(selected_authorities))
                    
                    # Display enrollment statistics
                    self._display_enrollment_stats(result, selected_authorities)
                    
                    self.last_query_result = result
                    
                    # Ask if user wants to save this query result
                    if Confirm.ask("Would you like to save this query result?"):
                        self.df = result
                        self.console.print("[bold green]Query result saved as current dataframe[/bold green]")
                
                else:  # Compare Two Authorities
                    # Display available authorities
                    values_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                    values_table.add_column("#", style="dim", width=6)
                    values_table.add_column(f"{title}", style="green")
                    
                    for i, val in enumerate(values, 1):
                        values_table.add_row(str(i), str(val))
                        
                    self.console.print("\nSelect first local authority:")
                    self.console.print(values_table)
                    
                    # Get first authority
                    while True:
                        val_choice1 = Prompt.ask("Enter number for first authority")
                        try:
                            index1 = int(val_choice1) - 1
                            if 0 <= index1 < len(values):
                                auth1 = values[index1]
                                break
                            else:
                                self.console.print("[red]Invalid selection number[/red]")
                        except ValueError:
                            self.console.print("[red]Invalid input[/red]")
                    
                    self.console.print("\nSelect second local authority:")
                    self.console.print(values_table)
                    
                    # Get second authority
                    while True:
                        val_choice2 = Prompt.ask("Enter number for second authority")
                        try:
                            index2 = int(val_choice2) - 1
                            if 0 <= index2 < len(values):
                                auth2 = values[index2]
                                if auth2 != auth1:
                                    break
                                else:
                                    self.console.print("[yellow]Please select a different authority[/yellow]")
                            else:
                                self.console.print("[red]Invalid selection number[/red]")
                        except ValueError:
                            self.console.print("[red]Invalid input[/red]")
                    
                    # Get available years
                    years = sorted([str(row[0]) for row in self.df.select("time_period").distinct().collect()])
                    
                    year_table = Table(show_header=True, header_style="bold magenta")
                    year_table.add_column("#", style="dim", width=6)
                    year_table.add_column("Year", style="green")
                    
                    for i, year in enumerate(years, 1):
                        year_table.add_row(str(i), str(year))
                        
                    self.console.print("\nSelect year for comparison:")
                    self.console.print(year_table)
                    
                    # Get year selection
                    while True:
                        year_choice = Prompt.ask("Enter year number")
                        try:
                            index = int(year_choice) - 1
                            if 0 <= index < len(years):
                                selected_year = str(years[index])  # Ensure year is string
                                self.console.print(f"[green]Selected year: {selected_year}[/green]")
                                break
                            else:
                                self.console.print("[red]Invalid selection number[/red]")
                        except ValueError:
                            self.console.print("[red]Invalid input[/red]")
                    
                    # Display comparison
                    try:
                        self._compare_local_authorities(self.df, auth1, auth2, selected_year)
                    except Exception as e:
                        self.console.print(f"[bold red]Error during comparison:[/bold red]")
                        self.console.print(f"[red]• Year: {selected_year}[/red]")
                        self.console.print(f"[red]• First authority: {auth1}[/red]")
                        self.console.print(f"[red]• Second authority: {auth2}[/red]")
                        self.console.print(f"[red]• Error details: {str(e)}[/red]")
            
            elif choice == "2":  # School Type
                column = "school_type"
                title = "School Type"
                
                with Progress(SpinnerColumn(), TextColumn("[green]Getting values...[/green]")) as progress:
                    task = progress.add_task("", total=None)
                    values = [row[0] for row in self.df.select(column).distinct().orderBy(column).collect()]
                
                values_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                values_table.add_column("#", style="dim", width=6)
                values_table.add_column(f"{title}", style="green")
                values_table.add_column("Count", style="blue")
                
                # Count occurrences of each value
                value_counts = self.df.groupBy(column).count().orderBy(column).collect()
                value_count_dict = {row[0]: row[1] for row in value_counts}
                
                for i, val in enumerate(values, 1):
                    count = value_count_dict.get(val, 0)
                    values_table.add_row(str(i), str(val), str(count))
                    
                self.console.print(values_table)
                    
                val_choice = Prompt.ask("Select value number to filter by (or 'a' for all)")
                
                if val_choice.lower() == 'a':
                    self.console.print(f"[bold green]Showing all values for {title}:[/bold green]")
                    result = self.df
                    school_type = "All School Types"
                else:
                    try:
                        index = int(val_choice) - 1
                        selected_val = values[index]
                        
                        with Progress(SpinnerColumn(), TextColumn("[green]Filtering data...[/green]")) as progress:
                            task = progress.add_task("", total=None)
                            result = self.df.filter(self.df[column] == selected_val)
                        
                        self.console.print(f"[bold green]Results for {title} = {selected_val}:[/bold green]")
                        school_type = selected_val
                        
                    except (ValueError, IndexError):
                        self.console.print("[bold red]Invalid selection[/bold red]")
                        return
                
                # Display absence statistics
                self._display_absence_stats(result, school_type)
                
                self.last_query_result = result
                
                # Ask if user wants to save this query result
                if Confirm.ask("Would you like to save this query result?"):
                    self.df = result
                    self.console.print("[bold green]Query result saved as current dataframe[/bold green]")
            
            else:  # Unauthorized Absences
                # First, let user select a year
                years = sorted([row[0] for row in self.df.select("time_period").distinct().collect()])
                
                year_table = Table(show_header=True, header_style="bold magenta")
                year_table.add_column("#", style="dim", width=6)
                year_table.add_column("Year", style="green")
                
                for i, year in enumerate(years, 1):
                    year_table.add_row(str(i), str(year))
                    
                self.console.print("\nSelect a year for unauthorized absence analysis:")
                self.console.print(year_table)
                
                year_choice = Prompt.ask("Enter year number")
                try:
                    selected_year = years[int(year_choice) - 1]
                    
                    # Now let user choose breakdown type
                    breakdown_table = Table(show_header=True, header_style="bold magenta")
                    breakdown_table.add_column("#", style="dim", width=6)
                    breakdown_table.add_column("Breakdown By", style="green")
                    
                    breakdown_options = ["Region", "Local Authority"]
                    for i, option in enumerate(breakdown_options, 1):
                        breakdown_table.add_row(str(i), option)
                    
                    self.console.print("\nSelect how to break down the data:")
                    self.console.print(breakdown_table)
                    
                    breakdown_choice = Prompt.ask("Enter choice", choices=["1", "2"])
                    
                    # Set column to group by
                    breakdown_col = "region_name" if breakdown_choice == "1" else "la_name"
                    
                    # Display statistics
                    self._display_unauth_absence_stats(self.df, breakdown_col, selected_year)
                    
                except (ValueError, IndexError):
                    self.console.print("[bold red]Invalid selection[/bold red]")
                    return
            
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
        viz_table.add_column("Column to Analyze", style="green")
        
        # Available columns for visualization
        columns = ["school_type", "region_name"]
        target_col = "sess_overall_percent"
        
        # Add options to table
        for i, column in enumerate(columns, 1):
            viz_table.add_row(str(i), column.replace('_', ' ').title())
        
        self.console.print(viz_table)
        
        try:
            # Get user choice
            choice = Prompt.ask("Select column to analyze", choices=["1", "2"])
            selected_column = columns[int(choice) - 1]
            
            # Create visualization
            self._plot_stacked_bar(self.df, selected_column, target_col)
            
            # Show basic statistics option
            if Confirm.ask("\nWould you like to see summary statistics for the current data?"):
                self._display_basic_statistics()
                
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")

    def _plot_stacked_bar(self, df: DataFrame, column: str, target_col: str):
        """
        Plots stacked horizontal bar chart for the specified column.
        
        Args:
            df: PySpark DataFrame containing the data
            column: Column name to visualize
            target_col: Target column to analyze (e.g. 'sess_overall_percent')
        """
        try:
            # Check if target column has more than 6 unique values
            target_unique = df.select(target_col).distinct().count()
            
            # Create 6 bins if numerical or more than 6 unique values
            if target_unique > 6:
                # Get min and max values for binning
                stats = df.agg(F.min(target_col), F.max(target_col)).collect()[0]
                min_val, max_val = stats[0], stats[1]
                
                # Create bin edges with equal width
                bin_edges = np.linspace(min_val, max_val, 7)  # 7 edges for 6 bins
                bin_labels = [f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)]
                
                # Get unique values and sort them, filtering out None values
                categories = [row[0] for row in df.select(column).distinct().orderBy(column).collect() if row[0] is not None]
                if not categories:
                    self.console.print(f"[bold red]No valid categories found for column {column}[/bold red]")
                    return
                
                self.console.print(f"[dim]Found {len(categories)} categories for {column}[/dim]")
                
                # Initialize data structure for proportions
                data = []
                
                # Calculate proportions for each bin
                for i in range(len(bin_edges)-1):
                    filtered = df.filter(
                        (F.col(target_col) >= bin_edges[i]) & 
                        (F.col(target_col) < bin_edges[i+1]) &
                        (F.col(column).isNotNull())  # Filter out null values
                    )
                    
                    # Group by category and calculate proportions
                    counts = filtered.groupBy(column).count()
                    total_counts = counts.agg(F.sum('count')).collect()[0][0] or 0
                    
                    if total_counts > 0:
                        proportions = counts.withColumn(
                            'proportion',
                            F.col('count') / total_counts
                        ).orderBy(column)
                        
                        # Convert to dictionary for ordered access
                        prop_dict = {row[column]: float(row['proportion']) for row in proportions.collect() if row[column] in categories}
                        data.append([prop_dict.get(cat, 0.0) for cat in categories])
                    else:
                        data.append([0.0] * len(categories))
                
                # Create the plot
                plt.figure(figsize=(12, max(6, len(categories) * 0.4)))  # Adjust height based on number of categories
                
                # Create color scheme similar to the notebook
                colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(bin_labels)))
                
                # Plot stacked bars
                bottom = np.zeros(len(categories))
                for i, d in enumerate(data):
                    plt.barh(categories, d, left=bottom, color=colors[i], label=bin_labels[i])
                    bottom += d
                
                # Customize the plot
                plt.xlabel('Proportion')
                plt.ylabel(column.replace('_', ' ').title())
                plt.title(f"Distribution of {target_col.replace('_', ' ').title()} by {column.replace('_', ' ').title()}")
                plt.legend(title=target_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                # Save and show the plot
                plt.savefig(f'plots/{column}_{target_col}_distribution.png', bbox_inches='tight')
                plt.close()
                
                self.console.print(f"\nPlot saved as 'plots/{column}_{target_col}_distribution.png'")
        
        except Exception as e:
            self.console.print(f"[bold red]Error creating plot:[/bold red] {str(e)}")
            traceback.print_exc()

    def _display_basic_statistics(self):
        """Display basic statistics for numeric columns."""
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
            if file_format == "csv":
                self.console.print(
                    "The data will be saved in a directory named after your filename. "
                    "This directory will contain:\n"
                    "- A partitioned CSV file (part-00000-*.csv)\n"
                    "- A _SUCCESS file indicating successful save\n"
                    "You can load this data later by selecting the directory when loading."
                )
            
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
