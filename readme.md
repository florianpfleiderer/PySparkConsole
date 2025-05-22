# Data Analysis Console App

A Python-based console application for analyzing school attendance data across different local authorities and school types. Built with PySpark for efficient data processing and analysis.

## Prerequisites

- Python 3.8+
- pip
- Java 8+ (for PySpark)

## Setup

1. Clone and enter the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   .\venv\Scripts\activate   # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create data directories:
   ```bash
   mkdir -p data/raw data/processed
   ```

## Project Structure

```
.
├── app.py              # Main application
├── src/               # Source code
├── data/              # Data storage
│   ├── raw/          # Input CSV files
│   └── processed/    # Processed data
├── plots/             # Visualizations
└── requirements.txt   # Dependencies
```

## Usage

1. Activate virtual environment:
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   .\venv\Scripts\activate   # On Windows
   ```

2. Run the application:
   ```bash
   python app.py
   ```

## Features

- Load and process CSV files
- Query by local authority
- Filter by school type
- Analyze unauthorized absences
- Generate visualizations

## Troubleshooting

- Java errors: Ensure Java 8+ is installed and JAVA_HOME is set
- PySpark warnings: Normal on first run
- Data loading: Check CSV files exist in data/ directory


