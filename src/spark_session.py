# Created on Wed Feb 05 2025 by 2400614
"""
Initialize Spark session and configurations.
"""

from pyspark.sql import SparkSession
import logging

# Configure logging
logging.getLogger("py4j").setLevel(logging.ERROR)


def set_spark_log_level(spark, level="ERROR"):
    """
    Set Spark's log level.
    
    Args:
        spark: Active Spark session
        level (str): Log level (ERROR, WARN, INFO, DEBUG)
    """
    spark.sparkContext.setLogLevel(level)


def create_spark_session(app_name="MySpark", log_level="ERROR"):
    """
    Create and return a new Spark session.

    Args:
        app_name (str): Name of the Spark application
        log_level (str): Log level for Spark (ERROR, WARN, INFO, DEBUG)

    Returns:
        SparkSession: Configured Spark session
    """
    # Create the session
    spark = (SparkSession.builder
            .master("local[*]")
            .appName(app_name)
            # .config("spark.driver.host", "localhost")
            # .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
            # .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
            # .config("spark.authenticate", "false")
            # .config("spark.ui.enabled", "false")
            # .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())
    
    # Set log level
    set_spark_log_level(spark, log_level)
    
    return spark


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
