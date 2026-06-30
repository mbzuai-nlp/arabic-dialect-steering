#!/bin/bash

SCRIPT="generate_steered_responses_token_ablation.py"
COEF=2.0
LAYER=21
N_STEER_TOKENS="30 50 70"

# ── ALLaM ──────────────────────────────────────────────────────────────────────
MODEL="humain-ai/ALLaM-7B-Instruct-preview"
VECTORS_DIR="dialect_vectors/ALLaM-7B-Instruct-preview"
OUT_DIR="results_token_ablation/allam"



# Doha → sau (closest Gulf eval)
python $SCRIPT \
    --model $MODEL \
    --eval-file eval_data/sau.jsonl \
    --vector-path $VECTORS_DIR/Doha_response_avg_diff.pt \
    --layers $LAYER \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF \
    --output-dir $OUT_DIR

# Riyadh → sau
python $SCRIPT \
    --model $MODEL \
    --eval-file eval_data/sau.jsonl \
    --vector-path $VECTORS_DIR/Riyadh_response_avg_diff.pt \
    --layers $LAYER \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF \
    --output-dir $OUT_DIR


# ── Fanar ──────────────────────────────────────────────────────────────────────
MODEL="QCRI/Fanar-1-9B-Instruct"
VECTORS_DIR="dialect_vectors/Fanar-1-9B-Instruct"
OUT_DIR="results_token_ablation/fanar"



# Doha → kwt
python $SCRIPT \
    --model $MODEL \
    --eval-file eval_data/sau.jsonl \
    --vector-path $VECTORS_DIR/Doha_response_avg_diff.pt \
    --layers $LAYER \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF \
    --output-dir $OUT_DIR

# Riyadh → sau
python $SCRIPT \
    --model $MODEL \
    --eval-file eval_data/sau.jsonl \
    --vector-path $VECTORS_DIR/Riyadh_response_avg_diff.pt \
    --layers $LAYER \
    --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF \
    --output-dir $OUT_DIR
