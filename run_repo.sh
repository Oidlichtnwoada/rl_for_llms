#!/bin/bash -e

# make sure the output is unbuffered
export PYTHONUNBUFFERED=1

# make sure experimental warnings from trl are silenced
export TRL_EXPERIMENTAL_SILENCE=1

# run the project (output is visible in terminal and appended to a log file)
uv run accelerate launch --num_processes=0 --num_machines=1 --mixed_precision=no --dynamo_backend=no -m rl_for_llms.main 2>&1 | tee -a logs/rl_for_llms.main.log
