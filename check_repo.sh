#!/bin/bash -e

# sync the project
uv sync

# format all files
uv run ruff format

# lint all files
uv run ruff check --fix
