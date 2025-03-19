"""
School Attendance Data Analysis Package.

This package provides tools and utilities for analyzing school attendance data
using PySpark.
"""

from src.utils.spark_utils import (
    create_spark_session,
    stop_spark_session,
    get_active_session
)

from src.utils.data_handling import (
    load_csv_data,
    handle_null_values,
    find_csv_files,
    create_file_table,
    create_null_value_table,
    save_dataframe,
    get_save_info
)

from src.utils.display_utils import (
    create_progress,
    display_dataframe_preview,
    create_menu_table,
    create_status_panel
)

from src.queries import (
    handle_local_authority_query,
    handle_school_type_query,
    handle_unauthorized_absences_query,
    analyse_absence_patterns
)

__version__ = '1.0.0'
__author__ = '240030614'

__all__ = [
    # Spark utilities
    'create_spark_session',
    'stop_spark_session',
    'get_active_session',
    
    # Data handling
    'load_csv_data',
    'handle_null_values',
    'find_csv_files',
    'create_file_table',
    'create_null_value_table',
    'save_dataframe',
    'get_save_info',
    
    # Display utilities
    'create_progress',
    'display_dataframe_preview',
    'create_menu_table',
    'create_status_panel',
    
    # Query helpers
    'handle_local_authority_query',
    'handle_school_type_query',
    'handle_unauthorized_absences_query',
    'analyse_absence_patterns'
]
