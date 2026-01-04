#!/bin/bash -e

# make sure the output is unbuffered
export PYTHONUNBUFFERED=1

# make sure experimental warnings from trl are silenced
export TRL_EXPERIMENTAL_SILENCE=1

# set the correct parameters based on GPU availability
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DYNAMO_BACKEND="no"
    MIXED_PRECISION="no"
else
    DYNAMO_BACKEND="no"
    MIXED_PRECISION="no"
fi

# run the project (output is visible in terminal and appended to a log file)
uv run accelerate launch --num_processes=0 --num_machines=1 --mixed_precision="$MIXED_PRECISION" --dynamo_backend="$DYNAMO_BACKEND" -m rl_for_llms.main 2>&1 | tee -a logs/rl_for_llms.main.log
