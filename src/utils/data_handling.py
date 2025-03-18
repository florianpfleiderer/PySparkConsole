"""Data handling utilities for loading, saving, and processing data.

This module provides functions and classes for handling data operations including:
- Finding and loading CSV files
- Handling null values
- Saving data in various formats
- Creating informative tables about data status
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, NamedTuple
from datetime import datetime
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from rich.table import Table
from rich.console import Console
from rich import box

# Configure logging
logging.basicConfig(level=logging.ERROR)


class FileInfo(NamedTuple):
    """Information about a CSV file or directory."""
    path: Path
    is_directory: bool
    size: float  # in KB
    modified: str
    location: str
    name: str


def find_csv_files(data_dir: Path) -> Tuple[List[FileInfo], List[FileInfo]]:
    """Find CSV files and directories containing CSV files.
    
    Args:
        data_dir: Base directory to search in
        
    Returns:
        Tuple containing:
        - List of FileInfo for individual CSV files
        - List of FileInfo for directories containing CSVs
    """
    csv_files = []
    processed_dirs = []
    
    try:
        # Check raw data directory
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            for file in raw_dir.glob("*.csv"):
                if file.is_file():
                    csv_files.append(FileInfo(
                        path=file,
                        is_directory=False,
                        size=file.stat().st_size / 1024,  # Convert to KB
                        modified=datetime.fromtimestamp(
                            file.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                        location="raw",
                        name=file.name
                    ))
        
        # Check processed data directory
        proc_dir = data_dir / "processed"
        if proc_dir.exists():
            for dir_path in proc_dir.iterdir():
                if dir_path.is_dir():
                    processed_dirs.append(FileInfo(
                        path=dir_path,
                        is_directory=True,
                        size=sum(
                            f.stat().st_size for f in dir_path.glob("*")
                        ) / 1024,
                        modified=datetime.fromtimestamp(
                            dir_path.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                        location="processed",
                        name=dir_path.name
                    ))
    
    except Exception as e:
        logging.error(f"Error finding CSV files: {str(e)}")
        
    return csv_files, processed_dirs


def create_file_table(
    csv_files: List[FileInfo],
    processed_dirs: List[FileInfo]
) -> Table:
    """Create a Rich table displaying file information.
    
    Args:
        csv_files: List of CSV file information
        processed_dirs: List of directory information
        
    Returns:
        Rich Table object with file information
    """
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )
    
    table.add_column("#", style="dim", width=6)
    table.add_column("Name", style="green")
    table.add_column("Location", style="blue")
    table.add_column("Size (KB)", justify="right")
    table.add_column("Modified", style="cyan")
    
    # Add CSV files
    for i, file_info in enumerate(csv_files, 1):
        table.add_row(
            str(i),
            file_info.name,
            file_info.location,
            f"{file_info.size:.1f}",
            file_info.modified
        )
        
    # Add processed directories
    for i, dir_info in enumerate(processed_dirs, len(csv_files) + 1):
        table.add_row(
            str(i),
            f"[bold]{dir_info.name}/[/bold]",
            dir_info.location,
            f"{dir_info.size:.1f}",
            dir_info.modified
        )
        
    return table


def create_null_value_table(
    df: DataFrame,
    null_counts: Dict[str, int]
) -> Tuple[Optional[Table], Dict[str, int]]:
    """Create a Rich table showing null value information.
    
    Args:
        df: DataFrame to analyze
        null_counts: Dictionary of column names to null counts
        
    Returns:
        Tuple containing:
        - Rich Table object with null value information (or None if no nulls)
        - Dictionary of columns with nulls
    """
    if not null_counts:
        return None, {}
        
    table = Table(
        show_header=True,
        header_style="bold red",
        box=box.ROUNDED
    )
    
    table.add_column("Column", style="yellow")
    table.add_column("Null Count", justify="right")
    table.add_column("Percentage", justify="right")
    
    row_count = df.count()
    cols_with_nulls = {}
    
    for col, count in null_counts.items():
        if count > 0:
            percentage = (count / row_count) * 100
            table.add_row(
                col,
                str(count),
                f"{percentage:.2f}%"
            )
            cols_with_nulls[col] = count
            
    return table, cols_with_nulls


def load_csv_data(
    spark: SparkSession,
    file_path: Path,
    is_directory: bool = False
) -> Tuple[DataFrame, Dict[str, int]]:
    """Load CSV data into a Spark DataFrame.
    
    Args:
        spark: Active Spark session
        file_path: Path to CSV file or directory
        is_directory: Whether the path is a directory
        
    Returns:
        Tuple containing:
        - Loaded DataFrame
        - Dictionary of null value counts by column
    """
    try:
        # Load the data
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(str(file_path))
            
        # Count null values
        null_counts = {
            col: df.filter(df[col].isNull()).count()
            for col in df.columns
        }
        
        return df, null_counts
        
    except Exception as e:
        logging.error(f"Error loading CSV data: {str(e)}")
        raise


def handle_null_values(
    df: DataFrame,
    null_counts: Dict[str, int]
) -> Tuple[DataFrame, Dict[str, int]]:
    """Handle null values in the DataFrame.
    
    Args:
        df: DataFrame to process
        null_counts: Dictionary of column names to null counts
        
    Returns:
        Tuple containing:
        - Processed DataFrame
        - Dictionary of remaining null counts
    """
    processed_df = df
    remaining_nulls = {}
    
    for col, count in null_counts.items():
        if count > 0:
            col_type = df.schema[col].dataType
            
            if isinstance(col_type, IntegerType):
                processed_df = processed_df.fillna(-1, subset=[col])
            else:
                processed_df = processed_df.fillna("MISSING", subset=[col])
                
            # Handle special characters
            processed_df = processed_df.withColumn(
                col,
                F.when(F.col(col) == ":", "-1")
                .otherwise(F.col(col))
            )
            
            # Check remaining nulls
            remaining_nulls[col] = processed_df.filter(
                processed_df[col].isNull()
            ).count()
            
    return processed_df, remaining_nulls


def save_dataframe(
    df: DataFrame,
    save_path: Path,
    file_format: str = "csv",
    mode: str = "errorifexists"
) -> Dict[str, Any]:
    """Save a DataFrame to the specified path in the given format.

    Args:
        df: PySpark DataFrame to save
        save_path: Path where to save the data
        file_format: Format to save as (csv, parquet, json)
        mode: Save mode (errorifexists, overwrite, append)

    Returns:
        Dictionary containing:
        - success: Whether save was successful
        - path: Path where data was saved
        - files: List of file information dictionaries
        - error: Error message if save failed
    """
    result = {
        "success": False,
        "path": str(save_path),
        "files": [],
        "error": None
    }
    
    try:
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the DataFrame
        df.write \
            .format(file_format) \
            .mode(mode) \
            .save(str(save_path))
            
        # Get information about saved files
        saved_files = list(save_path.glob("*"))
        result["files"] = [
            {
                "name": f.name,
                "size": f.stat().st_size / 1024  # Convert to KB
            }
            for f in saved_files
            if f.is_file()
        ]
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        logging.error(f"Error saving DataFrame: {str(e)}")
        
    return result


def get_save_info(file_format: str) -> str:
    """Get information about how data will be saved.

    Args:
        file_format: Format to save as (csv, parquet, json)

    Returns:
        String containing save format information
    """
    info = {
        "csv": (
            "Data will be saved as CSV files in a directory.\n"
            "• One file per partition\n"
            "• Headers included\n"
            "• UTF-8 encoding"
        ),
        "parquet": (
            "Data will be saved in Parquet format.\n"
            "• Compressed binary format\n"
            "• Optimized for analytics\n"
            "• Schema preserved"
        ),
        "json": (
            "Data will be saved as JSON files.\n"
            "• One record per line\n"
            "• UTF-8 encoding\n"
            "• One file per partition"
        )
    }
    
    return info.get(file_format, "Unknown format") 