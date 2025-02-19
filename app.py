# Created on Wed Feb 05 2025 by 240030614
# Copyright (c) 2025 University of St. Andrews
"""
Textual app for PySpark data manipulation, styled similarly to Harlequin.

Place this file (app.py) in the same directory as textual.css.
Some versions of Textual auto-load textual.css; if not, see "Manual CSS Loading" below.
"""

import os

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, DataTable, Button
from textual.binding import Binding

from src.spark_session import create_spark_session, stop_spark_session

DATA_DIR = "data/raw"


class CSVFileList(Static):
    """A widget that displays all CSV files in the data/raw/ directory as clickable buttons."""

    def on_mount(self) -> None:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            self.update("No CSV files found.")
        else:
            # Create a button for each CSV file, ensuring valid ID.
            for file in csv_files:
                valid_id = f"csv_{file.replace('.', '_')}"
                self.mount(Button(file, id=valid_id, classes="file-button"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # When a button is pressed, trigger the app to load that file.
        self.app.load_csv_file(str(event.button.label))


class SparkDataApp(App):
    """A Textual app for PySpark data manipulation with a Harlequin-like UI."""

    default_session = None
    data_table = None

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "load", "Load"),
    ]

    def compose(self) -> ComposeResult:
        yield Header("SparkDataApp")
        with Container(id="main_container"):
            # Left panel (1/4 width) remains static.
            with Container(id="left_panel"):
                yield Static("Data Catalog", id="data_catalog_title")
                yield Static("Placeholder widget here", id="placeholder")

            # Right panel (3/4 width), vertically split.
            with Container(id="right_panel"):
                # Upper panel initially shows a prompt.
                with Container(id="upper_right_panel"):
                    yield Static("Press 'l' to load file list", id="upper_placeholder")
                # Bottom panel for loaded data.
                with Container(id="bottom_right_panel"):
                    yield DataTable(id="main_table")
        yield Footer()

    async def on_load(self) -> None:
        """Called before terminal UI starts; load the stylesheet here if auto-load fails."""
        self.stylesheet.read("textual.css")

    def on_mount(self) -> None:
        """Initialize the Spark session when the app starts."""
        if self.default_session is None:
            self.session = create_spark_session("SparkDataApp")
        else:
            self.session = self.default_session

    def action_load(self) -> None:
        """Load the file list into the upper_right_panel when 'l' is pressed."""
        container = self.query_one("#upper_right_panel", Container)
        for child in container.children:
            child.remove()  # Remove any existing widgets.
        container.mount(Static("CSV Files in data/raw/", id="file_list_title"))
        container.mount(CSVFileList(id="file_list"))

    def load_csv_file(self, filename: str) -> None:
        """Load CSV data from the selected file and display a sample in the DataTable."""
        file_path = os.path.join(DATA_DIR, filename)
        self.data_table = self.query_one(DataTable)
        self.data_table.clear(columns=True)

        df = self.session.read.csv(file_path, header=True, inferSchema=True)
        for col in df.columns:
            self.data_table.add_column(col)

        # Retrieve the first 10 rows (adjust as needed) and populate the table.
        sample = df.head(10)
        for row in sample:
            self.data_table.add_row(*[str(item) for item in row])

    def action_quit(self) -> None:
        """Stop the Spark session and exit the application."""
        stop_spark_session(self.session)
        self.exit()


if __name__ == "__main__":
    app = SparkDataApp()
    app.run()
