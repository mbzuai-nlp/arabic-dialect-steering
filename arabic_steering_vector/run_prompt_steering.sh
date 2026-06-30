#!/bin/bash

SCRIPT="generate_steered_responses.py"
COEF=2.0

# ALLaM | egy | all layers
python $SCRIPT \
    --model humain-ai/ALLaM-7B-Instruct-preview \
    --eval-file eval_data/egy.jsonl \
    --vector-path dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_prompt_avg_diff.pt \
    --all-layers \
    --coef $COEF \
    --steering-type prompt \
    --output-dir results_prompt_steering/allam

# Fanar | egy | all layers
python $SCRIPT \
    --model QCRI/Fanar-1-9B-Instruct \
    --eval-file eval_data/egy.jsonl \
    --vector-path dialect_vectors/Fanar-1-9B-Instruct/Cairo_prompt_avg_diff.pt \
    --all-layers \
    --coef $COEF \
    --steering-type prompt \
    --output-dir results_prompt_steering/fanar

