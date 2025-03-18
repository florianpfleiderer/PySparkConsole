from __future__ import annotations
from typing import List, Dict, Any, Tuple
import logging
import traceback

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from rich.console import Console
from rich.table import Table
from rich import box

# Configure logging
logging.basicConfig(level=logging.ERROR)

def display_numeric_statistics(
    df: DataFrame,
    console: Console
) -> None:
    """Display basic statistics for numeric columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        console: Rich console instance for output
    """
    try:
        # Create statistics table
        stats_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        stats_table.add_column("Column", style="green")
        stats_table.add_column("Count", style="blue")
        stats_table.add_column("Mean", style="blue")
        stats_table.add_column("Min", style="blue")
        stats_table.add_column("Max", style="blue")
        
        # Get numeric columns
        numeric_cols = get_numeric_columns(df)
        
        if not numeric_cols:
            console.print("[yellow]No numeric columns found for statistics[/yellow]")
            return
        
        # Calculate and display statistics
        add_statistics_to_table(df, numeric_cols, stats_table)
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[bold red]Error generating statistics:[/bold red] {str(e)}")

def get_numeric_columns(df: DataFrame) -> List[str]:
    """Get list of numeric column names from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numeric column names
    """
    numeric_cols = []
    for field in df.schema.fields:
        data_type = str(field.dataType).lower()
        if "int" in data_type or "double" in data_type or "float" in data_type:
            numeric_cols.append(field.name)
    return numeric_cols

def add_statistics_to_table(
    df: DataFrame,
    numeric_cols: List[str],
    table: Table
) -> None:
    """Add statistics for numeric columns to the table.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        table: Rich table to add statistics to
    """
    summary = df.select(numeric_cols).summary("count", "mean", "min", "max").collect()
    count_row, mean_row, min_row, max_row = summary[:4]
    
    for col in numeric_cols:
        table.add_row(
            col,
            count_row[col],
            mean_row[col],
            min_row[col],
            max_row[col]
        ) 

def analyze_regional_attendance(
    df: DataFrame,
    console: Console
) -> bool:
    """Analyze regional attendance performance over time.
    
    Args:
        df: Input DataFrame containing the data
        console: Rich console instance for output
        
    Returns:
        bool: True if analysis was successful, False otherwise
    """
    try:
        # Filter for England regions only
        df_england = df.filter(
            (F.col("country_name") == "England") & 
            (F.col("geographic_level") == "Regional")
        )
        
        # Group by region and year, calculate average attendance
        region_year_stats = df_england.groupBy(
            "region_name", "time_period"
        ).agg(
            (100 - F.avg("sess_overall_percent")).alias("avg_attendance"),
            F.avg("sess_overall_percent").alias("avg_absence")
        ).orderBy("region_name", "time_period")
        
        if region_year_stats.count() == 0:
            console.print("[bold red]No regional data found for analysis[/bold red]")
            return False
            
        # Create plots
        create_regional_trend_plot(region_year_stats)
        create_regional_comparison_plot(region_year_stats)
        create_regional_improvement_plot(region_year_stats)
        
        # Display analysis results
        display_regional_analysis(region_year_stats, console)
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing regional data:[/bold red] {str(e)}")
        traceback.print_exc()
        return False

def create_regional_trend_plot(data: DataFrame) -> None:
    """Create line plot showing attendance trends by region.
    
    Args:
        data: PySpark DataFrame with regional statistics
    """
    plt.figure(figsize=(12, 6))
    
    # Collect unique regions
    regions = [row.region_name for row in 
              data.select("region_name").distinct().orderBy("region_name").collect()]
    
    for region in regions:
        # Filter and collect data for each region
        region_data = data.filter(F.col("region_name") == region).orderBy("time_period")
        region_rows = region_data.collect()
        
        # Convert year codes to academic year format (e.g., 200600 -> 2006/07)
        years = []
        for row in region_rows:
            year_start = str(row.time_period)[:4]
            year_end = str(int(year_start) + 1)[2:4]
            years.append(f"{year_start}/{year_end}")
        
        plt.plot(
            years,
            [row.avg_attendance for row in region_rows],
            marker='o',
            label=region
        )
    
    plt.xlabel("Academic Year")
    plt.ylabel("Average Attendance (%)")
    # plt.title("Regional Attendance Trends Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(
        'plots/regional_attendance_trends.png',
        bbox_inches='tight'
    )
    plt.close()

def create_regional_comparison_plot(data: DataFrame) -> None:
    """Create bar plot comparing absence rate distributions by region.
    
    Args:
        data: PySpark DataFrame with regional statistics
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate overall statistics for each region using absence rate
    region_stats = data.groupBy("region_name").agg(
        F.mean("avg_absence").alias("mean"),
        F.stddev("avg_absence").alias("std")
    ).orderBy("mean")
    
    # Collect results
    stats_rows = region_stats.collect()
    regions = [row.region_name for row in stats_rows]
    means = [row.mean for row in stats_rows]
    stds = [row.std if row.std is not None else 0 for row in stats_rows]
    
    # Create bar plot with error bars
    plt.bar(
        range(len(regions)),
        means,
        yerr=stds,
        capsize=5
    )
    
    plt.xticks(
        range(len(regions)),
        regions,
        rotation=45,
        ha='right'
    )
    
    plt.xlabel("Region")
    plt.ylabel("Average Absence Rate (%)")
    # plt.title("Overall Regional Absence Rate Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(
        'plots/regional_absence_comparison.png',
        bbox_inches='tight'
    )
    plt.close()

def create_regional_improvement_plot(data: DataFrame) -> None:
    """Create plot showing attendance improvement by region.
    
    Args:
        data: PySpark DataFrame with regional statistics
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate improvement for each region
    improvements = []
    regions = []
    
    # Get unique regions
    region_list = [row.region_name for row in 
                  data.select("region_name").distinct().orderBy("region_name").collect()]
    
    for region in region_list:
        region_data = data.filter(F.col("region_name") == region).orderBy("time_period")
        region_rows = region_data.collect()
        
        if len(region_rows) > 0:
            first_year = region_rows[0].avg_attendance
            last_year = region_rows[-1].avg_attendance
            improvement = last_year - first_year
            improvements.append(improvement)
            regions.append(region)
    
    # Sort by improvement
    sorted_indices = np.argsort(improvements)
    improvements = np.array(improvements)[sorted_indices]
    regions = np.array(regions)[sorted_indices]
    
    # Create bar plot
    colors = ['red' if x < 0 else 'green' for x in improvements]
    plt.bar(range(len(improvements)), improvements, color=colors)
    
    plt.xticks(
        range(len(improvements)),
        regions,
        rotation=45,
        ha='right'
    )
    
    plt.xlabel("Region")
    plt.ylabel("Change in Attendance (%)")
    # plt.title("Regional Attendance Improvement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(
        'plots/regional_attendance_improvement.png',
        bbox_inches='tight'
    )
    plt.close()

def display_regional_analysis(data: DataFrame, console: Console) -> None:
    """Display analysis of regional attendance performance.
    
    Args:
        data: PySpark DataFrame with regional statistics
        console: Rich console instance for output
    """
    # Calculate improvements and averages for each region
    improvements = {}
    overall_averages = {}
    
    # Get unique regions
    region_list = [row.region_name for row in 
                  data.select("region_name").distinct().orderBy("region_name").collect()]
    
    for region in region_list:
        region_data = data.filter(F.col("region_name") == region).orderBy("time_period")
        region_rows = region_data.collect()
        
        if len(region_rows) > 0:
            # Calculate improvement
            first_year = region_rows[0].avg_attendance
            last_year = region_rows[-1].avg_attendance
            improvement = last_year - first_year
            improvements[region] = improvement
            
            # Calculate overall average
            total_attendance = sum(row.avg_attendance for row in region_rows)
            overall_averages[region] = total_attendance / len(region_rows)
    
    # Find best and worst regions for both metrics
    best_improvement = max(improvements.items(), key=lambda x: x[1])
    worst_improvement = min(improvements.items(), key=lambda x: x[1])
    best_overall = max(overall_averages.items(), key=lambda x: x[1])
    worst_overall = min(overall_averages.items(), key=lambda x: x[1])
    
    # Create results table
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )
    table.add_column("Region", style="green")
    table.add_column("Change (%)", style="blue")
    table.add_column("Average (%)", style="cyan")
    table.add_column("Status", style="yellow")
    
    for region in sorted(region_list):
        change = improvements[region]
        average = overall_averages[region]
        status = (
            "Improved" if change > 0
            else "Worsened" if change < 0
            else "No Change"
        )
        table.add_row(
            region,
            f"{change:+.2f}",
            f"{average:.2f}",
            status
        )
    
    # Display results
    console.print("\n[bold]Regional Attendance Analysis[/bold]")
    console.print(
        f"\nBest improving region: [green]{best_improvement[0]}[/green] "
        f"(+{best_improvement[1]:.2f}%)"
    )
    console.print(
        f"Most declined region: [red]{worst_improvement[0]}[/red] "
        f"({worst_improvement[1]:.2f}%)"
    )
    console.print(
        f"\nBest overall region: [green]{best_overall[0]}[/green] "
        f"({best_overall[1]:.2f}% average attendance)"
    )
    console.print(
        f"Worst overall region: [red]{worst_overall[0]}[/red] "
        f"({worst_overall[1]:.2f}% average attendance)"
    )
    
    console.print("\n[bold]Detailed Regional Performance:[/bold]")
    console.print(table)
    
    console.print("\n[bold]Visualization files created:[/bold]")
    console.print("• plots/regional_attendance_trends.png")
    console.print("• plots/regional_absence_comparison.png")
    console.print("• plots/regional_attendance_improvement.png") 

def create_absence_pattern_plots(
    df: DataFrame,
    console: Console
) -> bool:
    """
    Create visualizations showing relationships between school types, 
    locations, and absence rates.
    
    Args:
        df: DataFrame containing the analyzed data
        console: Rich console instance
        
    Returns:
        bool: True if visualization was successful
    """
    try:
        # Ensure we have the correct column name
        if "avg_absence_rate" not in df.columns:
            console.print("[yellow]Warning: Column names may have changed during analysis[/yellow]")
            console.print("Available columns:", df.columns)
            return False

        # Get the latest year for the heatmap
        latest_year = df.select("time_period").distinct().orderBy(
            F.desc("time_period")
        ).first()[0]
        
        latest_data = df.filter(F.col("time_period") == latest_year)
        
        # Create pivot table with explicit column names
        pivot_data = latest_data.groupBy("school_type").pivot("region_name").agg(
            F.first("avg_absence_rate").alias("avg_absence_rate")
        ).orderBy("school_type")
        
        # Convert to numpy array for plotting
        school_types = [row["school_type"] for row in pivot_data.collect()]
        regions = [col for col in pivot_data.columns if col != "school_type"]
        data = np.array([[float(row[region] or 0) for region in regions] 
                        for row in pivot_data.collect()])
        
        # Create heatmap as separate plot
        plt.figure(figsize=(15, 8))
        heatmap = plt.imshow(data, aspect='auto', cmap='YlOrRd')
        plt.colorbar(heatmap, label='Average Absence Rate (%)')
        plt.xticks(range(len(regions)), regions, rotation=45, ha='right')
        plt.yticks(range(len(school_types)), school_types)
        plt.title(f'Regional Absence Rates by School Type ({latest_year})')
        
        # Add value annotations to the heatmap
        for i in range(len(school_types)):
            for j in range(len(regions)):
                plt.text(j, i, f'{data[i, j]:.1f}%',
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('plots/regional_heatmap.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Create trend lines as separate plot
        plt.figure(figsize=(15, 8))
        for school_type in school_types:
            school_data = df.filter(F.col("school_type") == school_type)
            years = [str(row["time_period"]) for row in 
                    school_data.select("time_period").distinct().orderBy("time_period").collect()]
            
            rates_data = school_data.groupBy("time_period").agg(
                F.avg("avg_absence_rate").alias("rate")
            ).orderBy("time_period").collect()
            
            rates = [float(row["rate"]) for row in rates_data]
            
            plt.plot(years, rates, marker='o', label=school_type, linewidth=2)
        
        plt.xlabel('Academic Year')
        plt.ylabel('Average Absence Rate (%)')
        plt.title('Absence Rate Trends Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/trends_by_type.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Display key findings and visualization summary
        console.print("\n[bold green]Analysis completed successfully![/bold green]")
        
        console.print("\n[bold]Key Findings:[/bold]")
        
        # Find school type with highest and lowest absence rates
        latest_avg = latest_data.groupBy("school_type").agg(
            F.avg("avg_absence_rate").alias("avg_rate")
        ).collect()
        
        highest = max(latest_avg, key=lambda x: float(x["avg_rate"] or 0))
        lowest = min(latest_avg, key=lambda x: float(x["avg_rate"] or 0))
        
        console.print(f"• Highest absence rate: [red]{highest['school_type']}[/red] ({float(highest['avg_rate']):.1f}%)")
        console.print(f"• Lowest absence rate: [green]{lowest['school_type']}[/green] ({float(lowest['avg_rate']):.1f}%)")
        
        # Calculate regional variations
        regional_var = latest_data.groupBy("region_name").agg(
            F.avg("avg_absence_rate").alias("avg_rate")
        ).collect()
        
        highest_region = max(regional_var, key=lambda x: float(x["avg_rate"] or 0))
        lowest_region = min(regional_var, key=lambda x: float(x["avg_rate"] or 0))
        
        console.print(f"• Region with highest absences: [red]{highest_region['region_name']}[/red] ({float(highest_region['avg_rate']):.1f}%)")
        console.print(f"• Region with lowest absences: [green]{lowest_region['region_name']}[/green] ({float(lowest_region['avg_rate']):.1f}%)")
        
        # Show visualization files created
        console.print("\n[bold]Visualization files created:[/bold]")
        console.print("• plots/regional_heatmap.png - Regional absence rates by school type")
        console.print("• plots/trends_by_type.png - Absence rate trends over time")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error creating visualization:[/bold red] {str(e)}")
        traceback.print_exc()
        return False 