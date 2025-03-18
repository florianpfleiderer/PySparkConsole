"""Spark session management and configuration utilities.

This module provides functions for creating, managing, and configuring Spark sessions.
"""

from __future__ import annotations
from pyspark.sql import SparkSession
import logging
from typing import Optional

# Configure logging
logging.getLogger("py4j").setLevel(logging.ERROR)


def set_spark_log_level(spark: SparkSession, level: str = "ERROR") -> None:
    """Set Spark's log level.
    
    Args:
        spark: Active Spark session
        level: Log level (ERROR, WARN, INFO, DEBUG)
    """
    spark.sparkContext.setLogLevel(level)


def create_spark_session(app_name: str = "MySpark", log_level: str = "ERROR") -> SparkSession:
    """Create and return a new Spark session.

    Args:
        app_name: Name of the Spark application
        log_level: Log level for Spark (ERROR, WARN, INFO, DEBUG)

    Returns:
        Configured Spark session
    """
    spark = (SparkSession.builder
            .master("local[*]")
            .appName(app_name)
            .config("spark.driver.extraJavaOptions", 
                   "-Djava.security.manager=allow")
            .getOrCreate())
    
    set_spark_log_level(spark, log_level)
    return spark


def stop_spark_session(spark: Optional[SparkSession]) -> None:
    """Safely stop a Spark session.

    Args:
        spark: Active Spark session to stop
    """
    if spark:
        spark.stop()


def get_active_session() -> Optional[SparkSession]:
    """Get the current active Spark session if it exists.

    Returns:
        Active session or None
    """
    return SparkSession.getActiveSession() 