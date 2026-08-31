#!/bin/bash

python extract_dialect_vectors_fast.py \
    --model QCRI/Fanar-1-9B-Instruct \
    --all-dialects\
    --data-dir data
