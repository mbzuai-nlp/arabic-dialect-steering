#!/bin/bash

SCRIPT="generate_steered_responses_token_ablation.py"
N_STEER_TOKENS="10 20 30 40 50 60 70 80 90 0"

# ALLaM | egy | layer 21 | coef 2
python $SCRIPT \
    --model humain-ai/ALLaM-7B-Instruct-preview \
    --eval-file eval_data/egy.jsonl \
    --vector-path dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
    --layers 21 \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef 2.0 \
    --output-dir results_token_ablation/allam

# Fanar | egy | layer 24 | coef 3
python $SCRIPT \
    --model QCRI/Fanar-1-9B-Instruct \
    --eval-file eval_data/egy.jsonl \
    --vector-path dialect_vectors/Fanar-1-9B-Instruct/Cairo_response_avg_diff.pt \
    --layers 24 \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef 3.0 \
    --output-dir results_token_ablation/fanar

# ALLaM | mar | layer 20 | coef 1
python $SCRIPT \
    --model humain-ai/ALLaM-7B-Instruct-preview \
    --eval-file eval_data/mar.jsonl \
    --vector-path dialect_vectors/ALLaM-7B-Instruct-preview/Rabat_response_avg_diff.pt \
    --layers 20 \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef 1.0 \
    --output-dir results_token_ablation/allam

# Fanar | mar | layer 21 | coef 2
python $SCRIPT \
    --model QCRI/Fanar-1-9B-Instruct \
    --eval-file eval_data/mar.jsonl \
    --vector-path dialect_vectors/Fanar-1-9B-Instruct/Rabat_response_avg_diff.pt \
    --layers 21 \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef 2.0 \
    --output-dir results_token_ablation/fanar
