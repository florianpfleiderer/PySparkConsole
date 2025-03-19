from __future__ import annotations
from typing import List, Dict, Tuple
import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Configure logging
logging.basicConfig(level=logging.ERROR)

def display_enrollment_stats(
    df: DataFrame,
    authorities: List[str],
    console: Console
) -> None:
    """
    Display enrollment statistics by year for selected local authorities.
    
    Args:
        df: PySpark DataFrame containing the data
        authorities: List of selected local authority names
        console: Rich console instance for output
    """
    # Create pivot table of enrollments by LA and year
    pivot_df = df.groupBy("la_name").pivot("time_period").agg(
        F.sum("enrolments")
    ).orderBy("la_name")

    # Get all years in order from the filtered data
    years = sorted([col for col in pivot_df.columns if col != "la_name"])
    
    if not years:
        console.print("[red]No data found for the selected time period[/red]")
        return

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

    console.print(table)

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

    console.print("\n")
    console.print(summary)

def display_absence_stats(
    df: DataFrame,
    school_type: str,
    console: Console
) -> None:
    """
    Display authorized absence statistics for the selected school type by year.
    
    Args:
        df: PySpark DataFrame containing the data
        school_type: Selected school type
        console: Rich console instance for output
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
            F.sum("enrolments").alias("total_enrolments")
        ).collect()[0]

        total_absences = stats["total_absences"] or 0
        total_enrolments = stats["total_enrolments"] or 0

        # Calculate absences per student
        absences_per_student = (
            total_absences / total_enrolments if total_enrolments > 0 else 0
        )

        table.add_row(
            str(year),
            f"{total_absences:,}",
            f"{total_enrolments:,}",
            f"{absences_per_student:.2f}"
        )

    console.print(table)

    # Ask if user wants to see detailed breakdown
    if Confirm.ask("\nWould you like to see a detailed breakdown of absence types?"):
        show_detailed_absence_breakdown(df, school_type, years, console)

def show_detailed_absence_breakdown(
    df: DataFrame,
    school_type: str,
    years: List[int],
    console: Console
) -> None:
    """
    Show detailed breakdown of absence types for a selected year.
    
    Args:
        df: PySpark DataFrame containing the data
        school_type: Selected school type
        years: List of available years
        console: Rich console instance for output
    """
    # Let user select a year
    year_table = Table(show_header=True, header_style="bold magenta")
    year_table.add_column("#", style="dim", width=6)
    year_table.add_column("Year", style="green")

    for i, year in enumerate(years, 1):
        year_table.add_row(str(i), str(year))

    console.print("\nSelect a year for detailed breakdown:")
    console.print(year_table)

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
        total_enrolments = totals["total_enrol"] or 0

        # Calculate statistics for each absence type
        for col, description in absence_types.items():
            stats = year_data.agg(F.sum(col).alias("total")).collect()[0]
            total = stats["total"] or 0

            percentage = (
                (total / total_authorized * 100)
                if total_authorized > 0 else 0
            )

            per_student = (
                total / total_enrolments
                if total_enrolments > 0 else 0
            )

            detail_table.add_row(
                description,
                f"{total:,}",
                f"{percentage:.1f}%",
                f"{per_student:.2f}"
            )

        console.print("\n")
        console.print(detail_table)

        # Add summary note
        console.print(
            f"\n[dim]Total students: {total_enrolments:,} | "
            f"Total authorized absences: {total_authorized:,}[/dim]"
        )

    except (ValueError, IndexError):
        console.print("[bold red]Invalid year selection[/bold red]")

def display_unauth_absence_stats(
    df: DataFrame,
    breakdown_by: str,
    year: str,
    console: Console
) -> None:
    """
    Display unauthorized absence statistics broken down by region or local authority.
    
    Args:
        df: PySpark DataFrame containing the data
        breakdown_by: Either 'region_name' or 'la_name'
        year: Selected year
        console: Rich console instance for output
    """
    # Filter for selected year and non-null values
    year_data = df.filter(
        (F.col("time_period") == year) &
        (F.col(breakdown_by).isNotNull()) &
        (F.col(breakdown_by) != "null")
    )

    # Group by region/LA and calculate statistics
    stats = (year_data.groupBy(breakdown_by)
            .agg(
                F.sum("sess_unauthorised").alias("total_unauth"),
                F.sum("enrolments").alias("total_students")
            )
            .orderBy(breakdown_by))

    # Calculate overall totals for percentages
    totals = stats.agg(F.sum("total_unauth"), F.sum("total_students")).collect()[0]

    overall_unauth = totals[0] or 0
    overall_students = totals[1] or 0

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

    console.print(table)

    # Show summary
    console.print(
        f"\n[dim]Total unauthorized absences: {overall_unauth:,} | "
        f"Total students: {overall_students:,} | "
        f"Overall sessions per student: {(overall_unauth/overall_students if overall_students > 0 else 0):.2f}[/dim]"
    )

def compare_local_authorities(
    df: DataFrame,
    auth1: str,
    auth2: str,
    year: str,
    console: Console
) -> None:
    """
    Compare two local authorities for a given year across multiple metrics.
    
    Args:
        df: PySpark DataFrame containing the data
        auth1: First local authority name
        auth2: Second local authority name
        year: Selected year for comparison
        console: Rich console instance for output
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
            console.print(f"[bold red]No data found for the following authorities in {year}:[/bold red]")
            for auth in missing_auths:
                console.print(f"• {auth}")
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
            console.print(f"[bold red]No metrics data found for the selected authorities in {year}[/bold red]")
            return

        if auth1 not in metrics_dict or auth2 not in metrics_dict:
            console.print(f"[bold red]Missing metrics data for one or both authorities in {year}:[/bold red]")
            if auth1 not in metrics_dict:
                console.print(f"• {auth1}")
            if auth2 not in metrics_dict:
                console.print(f"• {auth2}")
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

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error during comparison:[/bold red]")
        console.print(f"[red]• Selected year: {year}[/red]")
        console.print(f"[red]• First authority: {auth1}[/red]")
        console.print(f"[red]• Second authority: {auth2}[/red]")
        console.print(f"[red]• Error details: {str(e)}[/red]")

def get_distinct_values(
    df: DataFrame,
    column: str,
    with_counts: bool = False
) -> Tuple[List[str], Dict[str, int]]:
    """Get distinct values and their counts for a column.
    
    Args:
        df: Input DataFrame
        column: Column name to get values from
        with_counts: Whether to include value counts
        
    Returns:
        Tuple of values list and counts dictionary
    """
    # Filter out None values and get distinct values
    filtered_df = df.filter(F.col(column).isNotNull())
    values = [
        row[0] for row in filtered_df.select(column)
        .distinct()
        .orderBy(column)
        .collect()
    ]

    if with_counts:
        value_counts = filtered_df.groupBy(column).count().orderBy(column).collect()
        count_dict = {row[0]: row[1] for row in value_counts}
        return values, count_dict

    return values, {}

def display_value_table(
    values: List[str],
    title: str,
    console: Console,
    counts: Dict[str, int] = None
) -> None:
    """Display a table of values with optional counts.
    
    Args:
        values: List of values to display
        title: Column title
        console: Rich console instance
        counts: Optional dictionary of value counts
    """
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", style="dim", width=6)
    table.add_column(title, style="green")
    if counts:
        table.add_column("Count", style="blue")

    for i, val in enumerate(values, 1):
        if counts:
            count = counts.get(val, 0)
            table.add_row(str(i), str(val), str(count))
        else:
            table.add_row(str(i), str(val))

    console.print(table)

def select_multiple_values(
    values: List[str],
    console: Console,
    prompt_text: str = "Select value number to add",
    max_selections: int = 6
) -> List[str]:
    """Let user select multiple values from a list.
    
    Args:
        values: List of values to choose from
        console: Console instance
        prompt_text: Custom prompt text
        max_selections: Maximum number of items that can be selected
        
    Returns:
        List of selected values
    """
    selected = []
    while True:
        # Show remaining selections
        remaining = max_selections - len(selected)
        if remaining > 0:
            console.print(f"[blue]You can select {remaining} more item{'s' if remaining != 1 else ''}[/blue]")
        
        val_choice = Prompt.ask(
            f"{prompt_text} (or 'a' for all, 'd' when done, 'c' to cancel)"
        )

        if val_choice.lower() == 'c':
            return []
        elif val_choice.lower() == 'a':
            if len(values) > max_selections:
                console.print(f"[red]Cannot select all - maximum {max_selections} selections allowed[/red]")
                continue
            return values
        elif val_choice.lower() == 'd':
            if not selected:
                console.print("[yellow]Please select at least one value[/yellow]")
                continue
            return selected

        try:
            index = int(val_choice) - 1
            if 0 <= index < len(values):
                value = values[index]
                if value not in selected:
                    if len(selected) >= max_selections:
                        console.print(f"[red]Maximum {max_selections} selections allowed[/red]")
                        continue
                    selected.append(value)
                    console.print(f"[green]Added: {value}[/green]")
                else:
                    console.print(f"[yellow]{value} already selected[/yellow]")
            else:
                console.print("[red]Invalid selection number[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")

def select_single_value(
    values: List[str],
    console: Console,
    prompt_text: str = "Select value number",
    allow_all: bool = False
) -> Tuple[str, bool]:
    """Let user select a single value from a list.
    
    Args:
        values: List of values to choose from
        console: Rich console instance
        prompt_text: Custom prompt text
        allow_all: Whether to allow selecting all values
        
    Returns:
        Tuple of (selected value, whether all values were selected)
    """
    while True:
        val_choice = Prompt.ask(prompt_text + (" (or 'a' for all)" if allow_all else ""))
        
        if allow_all and val_choice.lower() == 'a':
            return "", True
            
        try:
            index = int(val_choice) - 1
            if 0 <= index < len(values):
                return values[index], False
            console.print("[red]Invalid selection number[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")

def get_available_years(df: DataFrame) -> List[str]:
    """Get sorted list of available years from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of years as strings
    """
    return sorted([str(row[0]) for row in df.select("time_period").distinct().collect()])

def display_year_selection(years: List[str], console: Console) -> None:
    """Display table of available years.
    
    Args:
        years: List of years to display
        console: Rich console instance
    """
    year_table = Table(show_header=True, header_style="bold magenta")
    year_table.add_column("#", style="dim", width=6)
    year_table.add_column("Year", style="green")
    
    for i, year in enumerate(years, 1):
        year_table.add_row(str(i), str(year))
        
    console.print("\nSelect year:")
    console.print(year_table)

def select_year(years: List[str], console: Console) -> str:
    """Let user select a year from the list.
    
    Args:
        years: List of years to choose from
        console: Rich console instance
        
    Returns:
        Selected year as string
    """
    while True:
        year_choice = Prompt.ask("Enter year number")
        try:
            index = int(year_choice) - 1
            if 0 <= index < len(years):
                selected_year = str(years[index])
                console.print(f"[green]Selected year: {selected_year}[/green]")
                return selected_year
            console.print("[red]Invalid selection number[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")

def handle_local_authority_query(
    df: DataFrame,
    console: Console
) -> Tuple[DataFrame, bool]:
    """Handle local authority query workflow.
    
    Args:
        df: Input DataFrame
        console: Rich console instance
        
    Returns:
        Tuple of (result DataFrame, whether to save result)
    """
    # First filter for Local authority level and Total school type
    base_df = df.filter(
        (F.col("geographic_level") == "Local authority") &
        (F.col("school_type") == "Total")
    )
    
    column = "la_name"
    title = "Local Authority"

    with Progress(
        SpinnerColumn(),
        TextColumn("[green]Getting values...[/green]")
    ) as progress:
        task = progress.add_task("", total=None)
        values, counts = get_distinct_values(base_df, column, True)

    # Show analysis options
    analysis_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )
    analysis_table.add_column("#", style="dim", width=6)
    analysis_table.add_column("Analysis Type", style="green")

    analysis_options = ["Enrollments by Year", "Compare Two Authorities"]
    for i, option in enumerate(analysis_options, 1):
        analysis_table.add_row(str(i), option)

    console.print("\nSelect analysis type:")
    console.print(analysis_table)

    analysis_choice = Prompt.ask("Enter choice", choices=["1", "2"])

    if analysis_choice == "1":
        return handle_enrollment_analysis(base_df, values, counts, column, title, console)
    else:
        return handle_authority_comparison(base_df, values, title, console)

def handle_enrollment_analysis(
    df: DataFrame,
    values: List[str],
    counts: Dict[str, int],
    column: str,
    title: str,
    console: Console
) -> Tuple[DataFrame, bool]:
    """Handle enrollment analysis workflow.
    
    Args:
        df: Input DataFrame
        values: List of authority values
        counts: Dictionary of value counts
        column: Column name
        title: Display title
        console: Rich console instance
        
    Returns:
        Tuple of (result DataFrame, whether to save result)
    """
    display_value_table(values, title, console, counts)

    selected_authorities = select_multiple_values(
        values,
        console,
        "Select authority number to add"
    )
    if not selected_authorities:
        return df, False

    # Get available years
    years = get_available_years(df)
    
    # Show time period options
    time_table = Table(show_header=True, header_style="bold magenta")
    time_table.add_column("#", style="dim", width=6)
    time_table.add_column("Option", style="green")
    
    time_options = ["Single Year", "Year Range", "All Years"]
    for i, option in enumerate(time_options, 1):
        time_table.add_row(str(i), option)
    
    console.print("\nSelect time period type:")
    console.print(time_table)
    
    time_choice = Prompt.ask("Enter choice", choices=["1", "2", "3"])
    
    result = df.filter(F.col(column).isin(selected_authorities))
    
    if time_choice == "1":  # Single year
        display_year_selection(years, console)
        selected_year = select_year(years, console)
        result = result.filter(F.col("time_period") == selected_year)
        time_desc = f"Year {selected_year}"
    elif time_choice == "2":  # Year range
        console.print("\nSelect start year:")
        display_year_selection(years, console)
        start_year = select_year(years, console)
        
        console.print("\nSelect end year:")
        while True:
            end_year = select_year(years, console)
            if end_year >= start_year:
                break
            console.print("[yellow]End year must be after start year[/yellow]")
        
        result = result.filter(
            (F.col("time_period") >= start_year) & 
            (F.col("time_period") <= end_year)
        )
        time_desc = f"Years {start_year}-{end_year}"
    else:  # All years
        time_desc = "All Years"

    console.print(f"\n[bold green]Results for selected authorities, {time_desc}:[/bold green]")
    display_enrollment_stats(result, selected_authorities, console)

    return result, True

def handle_authority_comparison(
    df: DataFrame,
    values: List[str],
    title: str,
    console: Console
) -> Tuple[DataFrame, bool]:
    """Handle authority comparison workflow.
    
    Args:
        df: Input DataFrame
        values: List of authority values
        title: Display title
        console: Rich console instance
        
    Returns:
        Tuple of (result DataFrame, whether to save result)
    """
    display_value_table(values, title, console)
    
    # Get first authority
    console.print("\nSelect first local authority:")
    auth1, _ = select_single_value(values, console)
    console.print(f"[bold green]Selected first authority:[/bold green] {auth1}")
    
    # Get second authority
    console.print("\nSelect second local authority:")
    while True:
        auth2, _ = select_single_value(values, console)
        if auth2 != auth1:
            console.print(f"[bold green]Selected second authority:[/bold green] {auth2}")
            break
        console.print(f"[yellow]Please select a different authority than {auth1}[/yellow]")
    
    # Get year selection
    years = get_available_years(df)
    display_year_selection(years, console)
    selected_year = select_year(years, console)
    
    try:
        compare_local_authorities(df, auth1, auth2, selected_year, console)
    except Exception as e:
        console.print(f"[bold red]Error during comparison:[/bold red]")
        console.print(f"[red]• Year: {selected_year}[/red]")
        console.print(f"[red]• First authority: {auth1}[/red]")
        console.print(f"[red]• Second authority: {auth2}[/red]")
        console.print(f"[red]• Error details: {str(e)}[/red]")
    
    return df, False

def handle_school_type_query(
    df: DataFrame,
    console: Console
) -> Tuple[DataFrame, bool]:
    """Handle school type query workflow.
    
    Args:
        df: Input DataFrame
        console: Rich console instance
        
    Returns:
        Tuple of (result DataFrame, whether to save result)
    """
    column = "school_type"
    title = "School Type"

    # Filter for school-level data to get accurate school type information
    base_df = df.filter(F.col("geographic_level") == "School")

    with Progress(
        SpinnerColumn(),
        TextColumn("[green]Getting values...[/green]")
    ) as progress:
        task = progress.add_task("", total=None)
        values, _ = get_distinct_values(base_df, column, False)

    display_value_table(values, title, console)

    selected_val, is_all = select_single_value(
        values,
        console,
        "Select value number to filter by",
        allow_all=True
    )

    # First filter by school type
    if is_all:
        console.print(f"[bold green]Showing all values for {title}:[/bold green]")
        result = base_df
        school_type = "All School Types"
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[green]Filtering data...[/green]")
        ) as progress:
            task = progress.add_task("", total=None)
            result = base_df.filter(base_df[column] == selected_val)
        school_type = selected_val

    # Now handle time period selection
    years = get_available_years(result)
    
    # Show time period options
    time_table = Table(show_header=True, header_style="bold magenta")
    time_table.add_column("#", style="dim", width=6)
    time_table.add_column("Option", style="green")
    
    time_options = ["Single Year", "Year Range", "All Years"]
    for i, option in enumerate(time_options, 1):
        time_table.add_row(str(i), option)
    
    console.print("\nSelect time period type:")
    console.print(time_table)
    
    time_choice = Prompt.ask("Enter choice", choices=["1", "2", "3"])
    
    if time_choice == "1":  # Single year
        display_year_selection(years, console)
        selected_year = select_year(years, console)
        result = result.filter(F.col("time_period") == selected_year)
        time_desc = f"Year {selected_year}"
    elif time_choice == "2":  # Year range
        console.print("\nSelect start year:")
        display_year_selection(years, console)
        start_year = select_year(years, console)
        
        console.print("\nSelect end year:")
        while True:
            end_year = select_year(years, console)
            if end_year >= start_year:
                break
            console.print("[yellow]End year must be after start year[/yellow]")
        
        result = result.filter(
            (F.col("time_period") >= start_year) & 
            (F.col("time_period") <= end_year)
        )
        time_desc = f"Years {start_year}-{end_year}"
    else:  # All years
        time_desc = "All Years"

    console.print(f"[bold green]Results for {school_type}, {time_desc}:[/bold green]")
    display_absence_stats(result, school_type, console)
    return result, True

def handle_unauthorized_absences_query(
    df: DataFrame,
    console: Console
) -> Tuple[DataFrame, bool]:
    """Handle unauthorized absences query workflow.
    
    Args:
        df: Input DataFrame
        console: Rich console instance
        
    Returns:
        Tuple of (result DataFrame, whether to save result)
    """
    years = get_available_years(df)
    display_year_selection(years, console)
    selected_year = select_year(years, console)

    # Show breakdown options
    breakdown_table = Table(show_header=True, header_style="bold magenta")
    breakdown_table.add_column("#", style="dim", width=6)
    breakdown_table.add_column("Breakdown By", style="green")

    breakdown_options = ["Region", "Local Authority"]
    for i, option in enumerate(breakdown_options, 1):
        breakdown_table.add_row(str(i), option)

    console.print("\nSelect how to break down the data:")
    console.print(breakdown_table)

    breakdown_choice = Prompt.ask("Enter choice", choices=["1", "2"])
    breakdown_col = "region_name" if breakdown_choice == "1" else "la_name"
    geographic_level = "Regional" if breakdown_choice == "1" else "Local authority"

    base_df = df.filter(
        (F.col("geographic_level") == geographic_level) &
        (F.col("school_type") == "Total")
    )

    display_unauth_absence_stats(base_df, breakdown_col, selected_year, console)
    return df, False

def analyse_absence_patterns(
    df: DataFrame,
    console: Console
) -> Tuple[DataFrame, bool]:
    """
    Analyze patterns between school types, locations, and absence rates.
    
    Args:
        df: Input DataFrame
        console: Rich console instance
        
    Returns:
        Tuple of (processed DataFrame, whether to save result)
    """
    try:
        base_df = df.filter(
            (F.col("geographic_level") == "Regional") &
            (F.col("school_type") != "Total") &
            (F.col("region_name").isNotNull()) &
            (F.col("region_name") != "null")
        )
        # Get unique school types and regions
        school_types = [row[0] for row in
                       base_df.select("school_type").distinct().orderBy("school_type").collect()]
        regions = [row[0] for row in
                  base_df.select("region_name").distinct().orderBy("region_name").collect()]
        years = [row[0] for row in
                base_df.select("time_period").distinct().orderBy("time_period").collect()]

        # Calculate weighted absence rates by school type, region, and year
        result = base_df.groupBy("school_type", "region_name", "time_period").agg(
            (100.0 * F.sum("sess_overall") / F.sum("sess_possible"))\
                .alias("avg_absence_rate"),
            F.sum("sess_overall").alias("total_sessions"),
            F.sum("sess_possible").alias("total_possible_sessions")
        ).orderBy("time_period", "school_type", "region_name")

        # Display summary statistics
        console.print("\n[bold]Analysis Summary:[/bold]")
        console.print(f"• Number of school types analyzed: {len(school_types)}")
        console.print(f"• Number of regions analyzed: {len(regions)}")
        console.print(f"• Time period: {min(years)} to {max(years)}")

        # Calculate overall patterns
        patterns_df = result.groupBy("school_type", "region_name").agg(
            F.sum("total_sessions").alias("total_sessions"),
            F.sum("total_possible_sessions").alias("total_possible_sessions"),
            (100.0 * F.sum("total_sessions") / F.sum("total_possible_sessions")).alias("avg_absence_rate")
        ).orderBy(F.desc("avg_absence_rate"))

        # Find highest absence rates by school type and region
        top_patterns = patterns_df.collect()

        # Display key findings
        console.print("\n[bold cyan]Key Findings:[/bold cyan]")
        
        # Top 3 highest absence rate combinations
        console.print("\n[bold]Highest Absence Rate Combinations:[/bold]")
        for i, row in enumerate(top_patterns[:3], 1):
            console.print(
                f"[yellow]{i}.[/yellow] {row['school_type']} schools in {row['region_name']}: "
                f"{row['avg_absence_rate']:.1f}% average absence rate"
            )

        # Calculate school type averages using raw session counts
        school_type_avg = patterns_df.groupBy("school_type").agg(
            F.sum("total_sessions").alias("total_sessions"),
            F.sum("total_possible_sessions").alias("total_possible_sessions"),
            (100.0 * F.sum("total_sessions") / F.sum("total_possible_sessions")).alias("avg_rate")
        ).orderBy(F.desc("avg_rate")).collect()

        console.print("\n[bold]School Type Analysis:[/bold]")
        for row in school_type_avg[:3]:
            console.print(
                f"• {row['school_type']} schools have an average absence rate of "
                f"{row['avg_rate']:.1f}% across all regions"
            )

        # Calculate regional averages using raw session counts
        region_avg = patterns_df.groupBy("region_name").agg(
            F.sum("total_sessions").alias("total_sessions"),
            F.sum("total_possible_sessions").alias("total_possible_sessions"),
            (100.0 * F.sum("total_sessions") / F.sum("total_possible_sessions")).alias("avg_rate")
        ).orderBy(F.desc("avg_rate")).collect()

        console.print("\n[bold]Regional Analysis:[/bold]")
        for row in region_avg[:3]:
            console.print(
                f"• {row['region_name']} has an average absence rate of "
                f"{row['avg_rate']:.1f}% across all school types"
            )

        # Overall conclusion
        console.print("\n[bold green]Summary Conclusion:[/bold green]")
        highest_combo = top_patterns[0]
        highest_type = school_type_avg[0]
        highest_region = region_avg[0]
        
        console.print(
            f"The analysis shows that {highest_type['school_type']} schools tend to have "
            f"the highest absence rates overall ({highest_type['avg_rate']:.1f}%), "
            f"with the highest rates specifically observed in {highest_combo['region_name']} "
            f"({highest_combo['avg_absence_rate']:.1f}%). "
            f"\nRegionally, {highest_region['region_name']} shows the highest average "
            f"absence rates ({highest_region['avg_rate']:.1f}%) across all school types."
        )

        return result, True

    except Exception as e:
        console.print(f"[bold red]Error during analysis:[/bold red] {str(e)}")
        return df, False
