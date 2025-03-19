# Practical 1 - Console App For Data Analysis

Author: 240030614

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Java 8 or higher (required for PySpark)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare data directories:
   ```bash
   mkdir -p data/raw data/processed
   ```
   Place your input CSV files in the `data/raw` directory. The application will save processed data to `data/processed` by default.

## Project Structure

```
.
├── app.py              # Main application file
├── src/               # Source code directory
├── data/              # Data directory
│   ├── raw/          # Raw CSV files directory
│   └── processed/    # Default save location for processed data
├── plots/             # Generated plots and visualizations
└── requirements.txt   # Project dependencies
```

## Running the Application

1. Ensure your virtual environment is activated:
   ```bash
   # On macOS/Linux
   source venv/bin/activate

   # On Windows
   .\venv\Scripts\activate
   ```

2. Run the main application:
   ```bash
   python3 app.py
   ```

## Available Commands

The application provides an interactive console interface with the following options:

1. Load Data - Load and process CSV files
2. Query by Local Authority - Analyze data for specific local authorities
3. Query by School Type - Filter and analyze data by school types
4. Analyze Unauthorized Absences - Study patterns in unauthorized absences
5. Generate Visualizations - Create plots and charts for data analysis
6. Exit - Quit the application

## Troubleshooting

1. If you encounter Java-related errors:
   - Ensure Java 8 or higher is installed
   - Set JAVA_HOME environment variable correctly

2. If you see PySpark initialization warnings:
   - These are normal and won't affect functionality
   - You can set SPARK_LOCAL_IP if needed

3. For data loading issues:
   - Ensure CSV files are present in the data/ directory
   - Check file permissions

## Notes

- The application uses PySpark for data processing, which may take a few seconds to initialize on first run
- Generated plots are saved in the plots/ directory
- Use Ctrl+C to force quit the application if needed


