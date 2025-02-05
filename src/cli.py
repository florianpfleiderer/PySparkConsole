# Created on Wed Feb 05 2025 by 240030614
# Copyright (c) 2025 University of St. Andrews
"""
Typer CLI for user interaction with the application.
"""

import typer

app = typer.Typer()


@app.command()
def greet(name: str):
    """Greet the user with a friendly message."""
    typer.echo(f"Hello, {name}! Welcome to the PySpark Console App.")


@app.command()
def add_numbers(a: int, b: int):
    """Add two numbers and return the result."""
    result = a + b
    typer.echo(f"The sum of {a} and {b} is {result}.")


@app.command()
def show_menu():
    """Display available commands."""
    typer.echo("Available commands:")
    typer.echo("  greet NAME  -> Greets the user.")
    typer.echo("  add_numbers A B  -> Adds two numbers.")
    typer.echo("  show_menu  -> Displays this menu.")


if __name__ == "__main__":
    app()
