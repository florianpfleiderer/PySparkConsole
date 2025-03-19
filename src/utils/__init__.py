"""
Utility modules for the School Attendance Data Analysis Package.

This package provides utility functions for:
- Spark session management
- Data handling and processing
- Display and UI components
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

__all__ = [
    # Spark session management
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
    'create_status_panel'
]
