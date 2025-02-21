# Created on Wed Feb 05 2025 by 240030614
# Copyright (c) 2025 University of St. Andrews
"""
Textual app for PySpark data manipulation, styled similarly to Harlequin.

Place this file (app.py) in the same directory as textual.css.
Some versions of Textual auto-load textual.css; if not, see "Manual CSS Loading" below.
"""

from __future__ import annotations
import os

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, Button, Select
from textual.binding import Binding

from src.spark_session import create_spark_session, stop_spark_session

DATA_DIR = "data/raw"


class FileSelector(Vertical):
    """A widget that displays a CSV file selector using the Select widget."""

    changed: bool = False

    def on_mount(self) -> None:
        self.changed = False
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            self.update("No CSV files found.")
        else:
            options = [(file, file) for file in csv_files]
            self.mount(Select(options=options, prompt="Select a CSV file", id="csv_file_selector"))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle selection change and load the chosen CSV file."""
        self.app.load_csv_file(event.value)
        self.change = True
        self.remove()

class QueryPanel(Vertical):
    """Should be displayed after startQuery button is pressed. 
    
    Should use a Select-Widget to display the columns that are available for the query.
    The available columns are: Local Autority ('la_name'), School Type ('school_type').
    """
    
    def on_mount(self) -> None:
        self.mount(Button("Query", id="query_button"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Display selector after query button is pressed.
        """
        options = [("Local Authority", "la_name"), ("School Type", "school_type")]
        self.mount(Select(options=options, prompt="Select a column", id="column_selector"))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle query methods and display results."""
        pass


class SparkDataApp(App):
    """A Textual app for PySpark data manipulation with a Harlequin-like UI."""

    default_session = None
    data_table = None

    CSS_PATH = "textual.css"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "load", "Load"),
        Binding("s", "save", "Save"),
    ]

    def compose(self) -> ComposeResult:
        yield Header("SparkDataApp")
        with Container(id="main_container"):
            with Container(id="left_panel"):
                with Horizontal(id="upper_left_panel"):
                    yield QueryPanel(id="query_panel")
            with Container(id="right_panel"):
                with Container(id="upper_right_panel"):
                    yield Static("Press 'l' to load file list", id="upper_placeholder")
                with Container(id="bottom_right_panel"):
                    yield DataTable(id="main_table")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the Spark session when the app starts."""
        if self.default_session is None:
            self.session = create_spark_session("SparkDataApp")
        else:
            self.session = self.default_session

    def action_load(self) -> None:
        """Load the file list into the upper_right_panel when 'l' is pressed.
        Once a CSV file is selected and loaded, remove the container.
        """
        container = self.query_one("#upper_right_panel", Container)
        for child in container.children:
            child.remove()
        file_selector = FileSelector(id="file_list")
        container.mount(file_selector)


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
        
    def action_save(self) -> None:
        """Save the selected dataframe using the spark methods."""
        pass

    def action_quit(self) -> None:
        """Stop the Spark session and exit the application."""
        stop_spark_session(self.session)
        self.exit()


if __name__ == "__main__":
    app = SparkDataApp()
    app.run()
