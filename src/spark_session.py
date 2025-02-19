# Created on Wed Feb 05 2025 by 2400614
# Copyright (c) 2025 University of St. Andrews
"""
Initialize Spark session and configurations.
"""

from pyspark.sql import SparkSession


def create_spark_session(app_name="MySpark"):
    """
    Create and return a new Spark session.

    Args:
        app_name (str): Name of the Spark application

    Returns:
        SparkSession: Configured Spark session
    """
    return SparkSession.builder.master("local[*]").appName(app_name).getOrCreate()


def stop_spark_session(spark):
    """
    Safely stop a Spark session.

    Args:
        spark (SparkSession): Active Spark session to stop
    """
    if spark:
        spark.stop()


def get_active_session():
    """
    Get the current active Spark session if it exists.

    Returns:
        SparkSession: Active session or None
    """
    return SparkSession.getActiveSession()


if __name__ == "__main__":
    DATA_PATH = "data/raw/Absence_3term201819_nat_reg_la_sch.csv"

    spark = create_spark_session()
    df = spark.read.format("csv").option("header", "true").load(DATA_PATH)
    df.show()
    print(spark)
    print("Default Spark session created.")
    stop_spark_session(spark)
    print("Default Spark session stopped.")
