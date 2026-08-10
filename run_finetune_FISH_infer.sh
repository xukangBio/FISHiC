#!/usr/bin/env bash

DATA_PATH="./Data/FISH_impute_data"
CHECKPOINT="./output_finetune_FISHImpute/best_finetuned.pt"
OUTPUT_DIR="./Result_FISH_impute"

echo "=================================================="
echo "FISHi-C FISH imputation inference"
echo "Data path: $DATA_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

python finetune_FISH_impute_infer.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --num_vis 10

echo "Inference completed. Results saved to: $OUTPUT_DIR"
