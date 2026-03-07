# rl_for_llms

Research code and writing for reinforcement learning experiments on reasoning-capable language models, with a focus on self-monitoring, verifiable rewards, and confidence-aware training.

This repository combines the implementation, experiment scripts, generated artifacts, and accompanying academic write-ups for the project. The codebase centers on training and evaluating language models with reinforcement learning workflows built around PyTorch, Hugging Face tooling, TRL, PEFT/LoRA, datasets utilities, and experiment tracking utilities.

## Overview

At a high level, the repository contains:

- training code for reinforcement-learning-based reasoning experiments,
- evaluation and chart-generation utilities,
- thesis and report sources describing the motivation, methodology, and experimental variants.

If you want the conceptual background first, start with the thesis material:

- [thesis/main-report.tex](thesis/main-report.tex) — Practical Work in AI report
- [thesis/main-seminarreport.tex](thesis/main-seminarreport.tex) — Seminar in AI (MSc) report
- [thesis/main-thesis.tex](thesis/main-thesis.tex) — Master’s Thesis document

The reports provide the research context and describe the experimental variants referenced by the code.

## Setup

### 1. Install the required Python interpreter

This project expects the exact Python version declared in [pyproject.toml](pyproject.toml).

Recommended approach: install and manage that interpreter with [pyenv](https://github.com/pyenv/pyenv).

Use `pyenv` to:

1. install the Python version required by the project manifest,
2. activate it for this repository,
3. verify that your shell is using that interpreter before running any project scripts.

The repository scripts assume that [uv](https://github.com/astral-sh/uv) is available in your shell environment.

### 2. Sync the project environment and verify the repository

Once the correct Python interpreter is active, run:

```bash
./check_repo.sh
```

This script is the main entry point for local repository setup. It synchronizes dependencies, prepares the project environment, formats the codebase, runs linting, and performs type checking.

### 3. Optional: install the pre-commit hook

To automatically run repository checks before each commit, execute:

```bash
./setup_pre_commit_hook.sh
```

This installs a Git pre-commit hook that calls [check_repo.sh](check_repo.sh).

## Running the project

Use the repository runner:

```bash
./run_repo.sh
```

Before launching a run, set the environment variables for the variant you want to execute. The primary variant switches are controlled through the runtime environment, and the experiment definitions are documented in [thesis/main-report.tex](thesis/main-report.tex).

In practice, the workflow is:

1. choose the experiment variant described in [thesis/main-report.tex](thesis/main-report.tex),
2. set the corresponding environment variables in your shell,
3. execute [run_repo.sh](run_repo.sh).

The variant selection is driven by environment configuration, including:

- `USE_CONFIDENCE_LOSS`
- `USE_CONFIDENCE_REWARD`

Depending on your setup, you may also want to configure optional runtime settings such as experiment tracking and other run-specific parameters before calling the runner.

## Repository map

- [rl_for_llms/main.py](rl_for_llms/main.py) — training entrypoint
- [rl_for_llms/evaluate.py](rl_for_llms/evaluate.py) — evaluation and chart generation entrypoint
- [pyproject.toml](pyproject.toml) — project metadata and Python requirement
- [check_repo.sh](check_repo.sh) — repository setup and validation script
- [run_repo.sh](run_repo.sh) — main execution script
- [thesis/main-report.tex](thesis/main-report.tex) — primary project report and experiment reference

## Notes

- Prefer the thesis documents for methodology, experiment design, and interpretation.
- Prefer the shell scripts at the repository root for setup and execution.
- Prefer referencing existing reports instead of duplicating their details when extending documentation.
