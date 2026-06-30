#!/bin/bash

MODEL="QCRI/Fanar-1-9B-Instruct"
VECTORS_DIR="dialect_vectors/Fanar-1-9B-Instruct"
COEF=2.0

python generate_steered_responses2.py \
    --model $MODEL \
    --eval-file eval_data/egy.jsonl \
    --vector-path $VECTORS_DIR/Cairo_response_avg_diff.pt \
    --all-layers --coef $COEF

# python generate_steered_responses.py \
#     --model $MODEL \
#     --eval-file eval_data/sau.jsonl \
#     --vector-path $VECTORS_DIR/Riyadh_response_avg_diff.pt \
#     --all-layers --coef $COEF

python generate_steered_responses2.py \
    --model $MODEL \
    --eval-file eval_data/mar.jsonl \
    --vector-path $VECTORS_DIR/Rabat_response_avg_diff.pt \
    --all-layers --coef $COEF

# python generate_steered_responses.py \
#     --model $MODEL \
#     --eval-file eval_data/syr.jsonl \
#     --vector-path $VECTORS_DIR/Aleppo_response_avg_diff.pt \
#     --all-layers --coef $COEF

# python generate_steered_responses.py \
#     --model $MODEL \
#     --eval-file eval_data/sdn.jsonl \
#     --vector-path $VECTORS_DIR/Khartoum_response_avg_diff.pt \
#     --all-layers --coef $COEF

# python generate_steered_responses.py \
#     --model $MODEL \
#     --eval-file eval_data/dza.jsonl \
#     --vector-path $VECTORS_DIR/Algiers_response_avg_diff.pt \
#     --all-layers --coef $COEF

# python generate_steered_responses.py \
#     --model $MODEL \
#     --eval-file eval_data/pse.jsonl \
#     --vector-path $VECTORS_DIR/Jerusalem_response_avg_diff.pt \
#     --all-layers --coef $COEF

# python generate_steered_responses.py \
#     --model $MODEL \
#     --eval-file eval_data/kwt.jsonl \
#     --vector-path $VECTORS_DIR/Doha_response_avg_diff.pt \
#     --all-layers --coef $COEF
