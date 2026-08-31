#!/bin/bash

SCRIPT="generate_steered_responses_token_ablation.py"
MODEL="QCRI/Fanar-1-9B-Instruct"
VECTORS_DIR="dialect_vectors/Fanar-1-9B-Instruct"
OUT_DIR="results_token_ablation_msa/fanar"
LAYER=21
COEF=2.0
N_STEER_TOKENS="10 20 30 40 50 60 70 80 90 0"
EVAL_FILE="eval_data/msa_samples_300.jsonl"

# Convert JSON to JSONL if not already done
python -c "
import json, os
if os.path.exists('$EVAL_FILE'):
    print('$EVAL_FILE already exists, skipping conversion.')
else:
    with open('eval_data/msa_samples_300.json', encoding='utf-8') as f:
        data = json.load(f)
    with open('$EVAL_FILE', 'w', encoding='utf-8') as out:
        for s in data['samples']:
            out.write(json.dumps({'prompt': s['prompt'], 'dialect': 'msa'}, ensure_ascii=False) + '\n')
    print(f'Converted {len(data[\"samples\"])} samples to $EVAL_FILE')
"

# Cairo
python $SCRIPT --model $MODEL --eval-file $EVAL_FILE \
    --vector-path $VECTORS_DIR/Cairo_response_avg_diff.pt \
    --layers $LAYER --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF --output-dir $OUT_DIR --name Cairo

# Rabat
python $SCRIPT --model $MODEL --eval-file $EVAL_FILE \
    --vector-path $VECTORS_DIR/Rabat_response_avg_diff.pt \
    --layers $LAYER --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF --output-dir $OUT_DIR --name Rabat

# Aleppo
python $SCRIPT --model $MODEL --eval-file $EVAL_FILE \
    --vector-path $VECTORS_DIR/Aleppo_response_avg_diff.pt \
    --layers $LAYER --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF --output-dir $OUT_DIR --name Aleppo

# Beirut
python $SCRIPT --model $MODEL --eval-file $EVAL_FILE \
    --vector-path $VECTORS_DIR/Beirut_response_avg_diff.pt \
    --layers $LAYER --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF --output-dir $OUT_DIR --name Beirut


# Doha
python $SCRIPT --model $MODEL --eval-file $EVAL_FILE \
    --vector-path $VECTORS_DIR/Doha_response_avg_diff.pt \
    --layers $LAYER --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF --output-dir $OUT_DIR --name Doha


# Riyadh
python $SCRIPT --model $MODEL --eval-file $EVAL_FILE \
    --vector-path $VECTORS_DIR/Riyadh_response_avg_diff.pt \
    --layers $LAYER --n-steer-tokens $N_STEER_TOKENS \
    --coef $COEF --output-dir $OUT_DIR --name Riyadh

