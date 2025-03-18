# Developer Context for Python Console Application

## Coding Standards
- Follow the latest PEP 8 standards for code style and formatting.
- Ensure all code is properly linted and formatted according to these standards.
- line length should not exceed 80 characters

## Type Hints
- Use type hints to improve code clarity and maintainability.
- Always import `from __future__ import annotations` to ensure forward compatibility of type hints.

## Dependencies
- Only use libraries specified in `requirements.txt`.
- Avoid introducing additional dependencies unless necessary and approved.

## Project Structure
- The application code (except the app.py) should reside in the `src/` directory.
- Organize functionalities into multiple modules within `src/`.
- Follow a modular approach, separating concerns into appropriate files.

## Code Reusability
- Maximize code reuse by abstracting common logic into shared modules.
- Avoid redundant code by implementing utility functions or classes where appropriate.
- Utilize imports effectively to maintain clean and efficient code.

## Directory Structure Example
```
project_root/
│-- requirements.txt
│-- developer_context.md
│-- app.py
│-- src/
│   │-- utils/
│   │   │-- helpers.py
│   │-- modules/
│   │   │-- feature_x.py
│   │   │-- feature_y.py
│-- data/ (...)
│-- plots/ (...)
```

## Best Practices
- Write clean, readable, and maintainable code.
- Use meaningful variable and function names.
- Keep functions and classes concise and focused on a single responsibility.
- Write short(!) docstrings for all functions, classes, and modules.
- Follow proper exception handling to improve application stability.
- Use proper input santitisation where user input is expected.
