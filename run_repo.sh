#!/bin/bash -e

# make sure the output is unbuffered
export PYTHONUNBUFFERED=1

# make sure experimental warnings from trl are silenced
export TRL_EXPERIMENTAL_SILENCE=1

# run the project (output is visible in terminal and appended to a log file)
uv run -m rl_for_llms.main 2>&1 | tee -a logs/rl_for_llms.main.log
