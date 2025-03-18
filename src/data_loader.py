from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# Configure logging
logging.basicConfig(level=logging.ERROR)


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
        # Load the data
        df = spark.read.csv(str(file_path), header=True, inferSchema=True)
        
        # Check for null values in each column
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