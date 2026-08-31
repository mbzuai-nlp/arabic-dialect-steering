#!/bin/bash

SCRIPT="extract_dialect_vectors_prompt.py"
DATA_DIR="data"

# ALLaM
python $SCRIPT \
    --model humain-ai/ALLaM-7B-Instruct-preview \
    --all-dialects \
    --data-dir $DATA_DIR

# Fanar
python $SCRIPT \
    --model QCRI/Fanar-1-9B-Instruct \
    --all-dialects \
    --data-dir $DATA_DIR

# # Jais
# python $SCRIPT \
#     --model inceptionai/jais-2-8b-chat \
#     --all-dialects \
#     --data-dir $DATA_DIR
