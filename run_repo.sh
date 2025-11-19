#!/bin/bash -e

# make sure the output is unbuffered
export PYTHONUNBUFFERED=1

# run the project
uv run -m rl_for_llms.main
