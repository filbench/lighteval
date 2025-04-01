#!/bin/bash

# Usage: ./run_all_evals_accelerate.sh <MODEL_NAME>
# This script runs all evaluation tasks listed in all_filbench_tasks.txt using the specified model.
# The model name should be provided as an argument to the script.

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  echo "Error: No model name provided."
  echo "Usage: ./run_all_evals_accelerate.sh <MODEL_NAME>"
  exit 1
fi

cat examples/tasks/all_filbench_tasks.txt | xargs -I {} python -m lighteval accelerate ${MODEL_NAME} {} --push-to-hub --results-org UD-Filipino --custom-tasks community_tasks/filbench_evals.py
