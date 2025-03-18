# Created on Tue Mar 18 2025 by 240030614
"""
This module provides a function to save a PySpark DataFrame to disk in
various formats (e.g., CSV, Parquet, JSON), handling directory creation,
logging errors, and providing detailed output on the save operation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import logging

from pyspark.sql import DataFrame
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

# Configure logging
logging.basicConfig(level=logging.ERROR)

def save_dataframe(
    df: DataFrame,
    save_path: Path,
    file_format: str = "csv",
    mode: str = "errorifexists"
) -> Dict[str, Any]:
    """
    Save a DataFrame to the specified path in the given format.

    Args:
        df: PySpark DataFrame to save
        save_path: Path where to save the data
        file_format: Format to save the data in ('csv', 'parquet', or 'json')
        mode: Save mode ('errorifexists', 'overwrite', 'append', 'ignore')

    Returns:
        Dict containing:
        - success: bool indicating if save was successful
        - path: Path where data was saved
        - files: List of files created (if available)
        - error: Error message if save failed
    """
    try:
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if file_format == "csv":
            df.repartition(1).write.csv(
                str(save_path), 
                header=True, 
                mode=mode
            )
        elif file_format == "parquet":
            df.write.parquet(str(save_path), mode=mode)
        elif file_format == "json":
            df.write.json(str(save_path), mode=mode)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Get file information if directory exists
        files = []
        if save_path.exists():
            files = [
                {
                    "name": item.name,
                    "size": item.stat().st_size / 1024  # Size in KB
                }
                for item in save_path.iterdir()
            ]

        return {
            "success": True,
            "path": save_path,
            "files": files,
            "error": None
        }

    except Exception as e:
        logging.error(f"Error saving data to {save_path}: {str(e)}")
        return {
            "success": False,
            "path": save_path,
            "files": [],
            "error": str(e)
        }

def get_save_info(file_format: str) -> str:
    """
    Get information about how data will be saved in the specified format.

    Args:
        file_format: Format to save the data in ('csv', 'parquet', or 'json')

    Returns:
        String containing format-specific save information
    """
    if file_format == "csv":
        return (
            "The data will be saved in a directory named after your filename. "
            "This directory will contain:\n"
            "- A partitioned CSV file (part-00000-*.csv)\n"
            "- A _SUCCESS file indicating successful save\n"
            "You can load this data later by selecting the directory when loading."
        )
    elif file_format == "parquet":
        return (
            "The data will be saved in Parquet format, which is:\n"
            "- A columnar storage format\n"
            "- Optimized for handling complex data in bulk\n"
            "- Efficient for querying specific columns"
        )
    elif file_format == "json":
        return (
            "The data will be saved in JSON format:\n"
            "- Each record will be a JSON object\n"
            "- Good for interoperability with other systems\n"
            "- Maintains data types and structure"
        )
    else:
        return f"Unknown format: {file_format}"
