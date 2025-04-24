#!/bin/bash

MODEL_NAME=$1
TASK_NAME=$2

if [ -z "$MODEL_NAME" ]; then
  echo "Error: No model name provided."
  echo "Usage: ./single_target.sh <MODEL_NAME> <TASK_NAME>"
  exit 1
fi

if [ -z "$TASK_NAME" ]; then
  echo "Error: No task name provided."
  echo "Usage: ./single_target.sh <MODEL_NAME> <TASK_NAME>"
  exit 1
fi

python -m lighteval vllm ${MODEL_NAME} ${TASK_NAME} \
    --push-to-hub \
    --results-org UD-Filipino \
    --custom-tasks community_tasks/filbench_evals.py
