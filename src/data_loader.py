from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, NamedTuple
from datetime import datetime
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from rich.table import Table
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
        - List of FileInfo for regular CSV files
        - List of FileInfo for directories with CSV files
    """
    csv_files = []
    processed_dirs = []
    
    # Handle processed directory specially
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        for item in processed_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                has_csv = any(
                    f.suffix == '.csv' and not f.name.startswith('.')
                    for f in item.iterdir()
                )
                if has_csv:
                    total_size = sum(
                        f.stat().st_size for f in item.rglob('*.csv')
                        if not f.name.startswith('.')
                    )
                    processed_dirs.append(FileInfo(
                        path=item,
                        is_directory=True,
                        size=total_size / 1024,
                        modified=datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).strftime('%Y-%m-%d %H:%M'),
                        location="processed",
                        name=item.name
                    ))
    
    # Find regular CSV files
    for item in data_dir.rglob('*.csv'):
        if (not item.is_file() or item.name.startswith('.') or
            'processed' in item.parts):
            continue
        rel_path = item.relative_to(data_dir)
        location = str(rel_path.parent)
        if location == '.':
            location = 'data'
        csv_files.append(FileInfo(
            path=item,
            is_directory=False,
            size=item.stat().st_size / 1024,
            modified=datetime.fromtimestamp(
                item.stat().st_mtime
            ).strftime('%Y-%m-%d %H:%M'),
            location=location,
            name=item.name
        ))
    
    return csv_files, processed_dirs

def create_file_table(
    csv_files: List[FileInfo],
    processed_dirs: List[FileInfo]
) -> Table:
    """Create a table displaying file information.
    
    Args:
        csv_files: List of regular CSV files
        processed_dirs: List of directories with CSV files
        
    Returns:
        Rich Table object with file information
    """
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )
    table.add_column("#", style="dim", width=6)
    table.add_column("Type", style="yellow")
    table.add_column("Location", style="blue")
    table.add_column("Name", style="green")
    table.add_column("Size", style="cyan")
    table.add_column("Last Modified", style="magenta")
    
    # Add regular CSV files
    for i, file_info in enumerate(csv_files, 1):
        table.add_row(
            str(i),
            "File",
            file_info.location,
            file_info.name,
            f"{file_info.size:.1f} KB",
            file_info.modified
        )
    
    # Add directories with CSV files
    for i, dir_info in enumerate(processed_dirs, len(csv_files) + 1):
        table.add_row(
            str(i),
            "Directory",
            dir_info.location,
            dir_info.name,
            f"{dir_info.size:.1f} KB",
            dir_info.modified
        )
    
    return table

def create_null_value_table(
    df: DataFrame,
    null_counts: Dict[str, int]
) -> Tuple[Table, Dict[str, int]]:
    """Create a table displaying null value information.
    
    Args:
        df: Input DataFrame
        null_counts: Dictionary of null counts per column
        
    Returns:
        Tuple containing:
        - Rich Table object with null value information
        - Dictionary of columns with null values
    """
    row_count = df.count()
    cols_with_nulls = {
        col: count for col, count in null_counts.items()
        if count > 0
    }
    
    if not cols_with_nulls:
        return None, {}
    
    null_table = Table(
        show_header=True,
        header_style="bold red",
        box=box.ROUNDED
    )
    null_table.add_column("Column", style="yellow")
    null_table.add_column("Data Type", style="magenta")
    null_table.add_column("Null Count", style="cyan")
    null_table.add_column("% of Total", style="green")
    
    col_types = {
        field.name: str(field.dataType)
        for field in df.schema.fields
    }
    
    for col, count in cols_with_nulls.items():
        percentage = (count / row_count) * 100
        null_table.add_row(
            col,
            col_types[col],
            str(count),
            f"{percentage:.2f}%"
        )
    
    return null_table, cols_with_nulls

def load_csv_data(
    spark: SparkSession,
    file_path: Path,
    is_directory: bool = False
) -> Tuple[DataFrame, Dict[str, int]]:
    """
    Load data from a CSV file or directory into a Spark DataFrame.

    Args:
        spark: Active Spark session
        file_path: Path to the CSV file or directory
        is_directory: Whether the path is a directory containing CSV files

    Returns:
        Tuple containing:
        - DataFrame: Loaded Spark DataFrame
        - Dict[str, int]: Dictionary of null counts per column
    """
    try:
        df = spark.read.csv(str(file_path), header=True, inferSchema=True)

        null_counts = {
            col: int(df.filter(F.col(col).isNull()).count())
            for col in df.columns
        }

        return df, null_counts

    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def handle_null_values(
    df: DataFrame,
    null_counts: Dict[str, int]
) -> Tuple[DataFrame, Dict[str, int]]:
    """
    Handle null values in the DataFrame based on column types.

    Args:
        df: Input DataFrame
        null_counts: Dictionary of null counts per column

    Returns:
        Tuple containing:
        - DataFrame: DataFrame with handled null values
        - Dict[str, int]: Dictionary of remaining null counts per column
    """
    try:
        # Get numeric column names
        numeric_col_names = [
            field.name for field in df.schema.fields 
            if isinstance(field.dataType, IntegerType)
        ]
        
        # Handle special case for sess_auth_ext_holiday
        special_cols = df.select([
            col for col in df.columns 
            if ":" in str(df.select(col).first()[0])
        ])
        
        # Fill null values based on column type
        for col in df.columns:
            if col in numeric_col_names:
                df = df.fillna(-1, subset=[col])
            else:
                df = df.fillna("MISSING", subset=[col])
        
        # Handle special characters in columns
        for special_col in special_cols.columns:
            df = df.replace(":", "-1", subset=special_col)
            if special_col in numeric_col_names:
                df = df.withColumn(
                    special_col, 
                    F.col(special_col).cast(IntegerType())
                )
        
        # Verify remaining null values
        remaining_nulls = {
            col: int(df.filter(F.col(col).isNull()).count()) 
            for col in df.columns
        }
        
        return df, remaining_nulls
        
    except Exception as e:
        logging.error(f"Error handling null values: {str(e)}")
        raise 