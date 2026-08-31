#!/bin/bash

SCRIPT="generate_steered_responses.py"
COEF=2.0


# ALLaM | mar | all layers
python $SCRIPT \
    --model humain-ai/ALLaM-7B-Instruct-preview \
    --eval-file eval_data/mar.jsonl \
    --vector-path dialect_vectors/ALLaM-7B-Instruct-preview/Rabat_prompt_avg_diff.pt \
    --all-layers \
    --coef $COEF \
    --steering-type prompt \
    --output-dir results_prompt_steering/allam

# Fanar | mar | all layers
python $SCRIPT \
    --model QCRI/Fanar-1-9B-Instruct \
    --eval-file eval_data/mar.jsonl \
    --vector-path dialect_vectors/Fanar-1-9B-Instruct/Rabat_prompt_avg_diff.pt \
    --all-layers \
    --coef $COEF \
    --steering-type prompt \
    --output-dir results_prompt_steering/fanar
