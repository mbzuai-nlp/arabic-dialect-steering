#!/bin/bash

MODEL="QCRI/Fanar-1-9B-Instruct"
VECTORS_DIR="dialect_vectors/Fanar-1-9B-Instruct"
SCRIPT="generate_steered_responses_coeff_sweep.py"

# # Cairo -> egy
# python $SCRIPT \
#     --model $MODEL \
#     --eval-file eval_data/egy.jsonl \
#     --vector-path $VECTORS_DIR/Cairo_response_avg_diff.pt

# Rabat -> mar
python $SCRIPT \
    --model $MODEL \
    --eval-file eval_data/mar.jsonl \
    --vector-path $VECTORS_DIR/Rabat_response_avg_diff.pt

