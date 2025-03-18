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

def create_stacked_bar_plot(
    df: DataFrame,
    column: str,
    target_col: str,
    console: Console,
    num_bins: int = 6
) -> bool:
    """Create a stacked horizontal bar chart for the specified column.
    
    Args:
        df: Input DataFrame containing the data
        column: Column name to visualize
        target_col: Target column to analyze (e.g. 'sess_overall_percent')
        console: Rich console instance for output
        num_bins: Number of bins for numerical data (default: 6)
        
    Returns:
        bool: True if plot was created successfully, False otherwise
    """
    try:
        # Check if target column has more than num_bins unique values
        target_unique = df.select(target_col).distinct().count()
        
        # Create bins if numerical or more than num_bins unique values
        if target_unique > num_bins:
            # Get min and max values for binning
            stats = df.agg(F.min(target_col), F.max(target_col)).collect()[0]
            min_val, max_val = stats[0], stats[1]
            
            # Create bin edges with equal width
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
            bin_labels = [
                f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" 
                for i in range(len(bin_edges)-1)
            ]
            
            # Get categories and validate
            categories = get_valid_categories(df, column)
            if not categories:
                console.print(f"[bold red]No valid categories found for column {column}[/bold red]")
                return False
            
            console.print(f"[dim]Found {len(categories)} categories for {column}[/dim]")
            
            # Calculate proportions
            data = calculate_bin_proportions(
                df, column, target_col, categories, bin_edges
            )
            
            # Create and save plot
            create_plot(
                data, categories, bin_labels, column, target_col
            )
            
            console.print(f"\nPlot saved as 'plots/{column}_{target_col}_distribution.png'")
            return True
            
    except Exception as e:
        console.print(f"[bold red]Error creating plot:[/bold red] {str(e)}")
        traceback.print_exc()
        return False

def get_valid_categories(df: DataFrame, column: str) -> List[str]:
    """Get sorted list of valid categories from a column.
    
    Args:
        df: Input DataFrame
        column: Column name to get categories from
        
    Returns:
        List of valid category names
    """
    return [
        row[0] for row in df.select(column)
        .distinct()
        .orderBy(column)
        .collect() 
        if row[0] is not None
    ]

def calculate_bin_proportions(
    df: DataFrame,
    column: str,
    target_col: str,
    categories: List[str],
    bin_edges: np.ndarray
) -> List[List[float]]:
    """Calculate proportions for each bin and category.
    
    Args:
        df: Input DataFrame
        column: Column to group by
        target_col: Target column for binning
        categories: List of category names
        bin_edges: Array of bin edge values
        
    Returns:
        List of proportion lists for each bin
    """
    data = []
    
    for i in range(len(bin_edges)-1):
        filtered = df.filter(
            (F.col(target_col) >= bin_edges[i]) & 
            (F.col(target_col) < bin_edges[i+1]) &
            (F.col(column).isNotNull())
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
            prop_dict = {
                row[column]: float(row['proportion']) 
                for row in proportions.collect() 
                if row[column] in categories
            }
            data.append([prop_dict.get(cat, 0.0) for cat in categories])
        else:
            data.append([0.0] * len(categories))
            
    return data

def create_plot(
    data: List[List[float]],
    categories: List[str],
    bin_labels: List[str],
    column: str,
    target_col: str
) -> None:
    """Create and save the stacked bar plot.
    
    Args:
        data: List of proportion lists for each bin
        categories: List of category names
        bin_labels: List of bin labels
        column: Column name for y-axis
        target_col: Target column name for title
    """
    # Create figure with dynamic height
    plt.figure(figsize=(12, max(6, len(categories) * 0.4)))
    
    # Create color scheme
    colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(bin_labels)))
    
    # Plot stacked bars
    bottom = np.zeros(len(categories))
    for i, d in enumerate(data):
        plt.barh(categories, d, left=bottom, color=colors[i], label=bin_labels[i])
        bottom += d
    
    # Customize the plot
    plt.xlabel('Proportion')
    plt.ylabel(column.replace('_', ' ').title())
    plt.title(
        f"Distribution of {target_col.replace('_', ' ').title()} by "
        f"{column.replace('_', ' ').title()}"
    )
    plt.legend(
        title=target_col.replace('_', ' ').title(),
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.tight_layout()
    
    # Save and close
    plt.savefig(
        f'plots/{column}_{target_col}_distribution.png',
        bbox_inches='tight'
    )
    plt.close()

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