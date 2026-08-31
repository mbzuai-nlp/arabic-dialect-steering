#!/bin/bash

MODEL="humain-ai/ALLaM-7B-Instruct-preview"
VECTORS_DIR="dialect_vectors/ALLaM-7B-Instruct-preview"
SCRIPT="generate_steered_responses_coeff_sweep-Copy1.py"

# Cairo → egy
python $SCRIPT \
    --model $MODEL \
    --eval-file eval_data/egy.jsonl \
    --vector-path $VECTORS_DIR/Cairo_response_avg_diff.pt

# # Rabat → mar
# python $SCRIPT \
#     --model $MODEL \
#     --eval-file eval_data/mar.jsonl \
#     --vector-path $VECTORS_DIR/Rabat_response_avg_diff.pt

