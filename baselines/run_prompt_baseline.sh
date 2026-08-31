#!/usr/bin/env bash
# Run explicit-prompt baseline for all three models from the repository root.
EVAL_DATA=arabic_steering_vector/eval_data

python baselines/generate_explicit_prompt_eval.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --eval-data "$EVAL_DATA" \
  --output-dir baselines/explicit_prompt_outputs/ALLaM-7B-Instruct-preview \
  --max-new-tokens 100 \
  --batch-size 64

python baselines/generate_explicit_prompt_eval.py \
  --model QCRI/Fanar-1-9B-Instruct \
  --eval-data "$EVAL_DATA" \
  --output-dir baselines/explicit_prompt_outputs/Fanar-1-9B-Instruct \
  --max-new-tokens 100 \
  --batch-size 64

python baselines/generate_explicit_prompt_eval.py \
  --model inceptionai/Jais-2-8B-Chat \
  --eval-data "$EVAL_DATA" \
  --output-dir baselines/explicit_prompt_outputs/Jais-2-8B-Chat \
  --max-new-tokens 100 \
  --batch-size 64
