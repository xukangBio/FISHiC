#!/usr/bin/env bash

DATA_PATH="./Data/Bulk_enhance_data"
CHECKPOINT="./output_finetune_enhanced_bulk_hic/best_finetuned_enhanced.pt"
OUTPUT_DIR="./Result_finetune_enhanced_bulk_hic"

INPUT_FILENAME="lr_all.npy"
LABEL_FILENAME="hr_all.npy"
VIZ_CLIP_PERCENTILE=90

echo "=================================================="
echo "FISHi-C bulk Hi-C enhancement inference"
echo "Data path: $DATA_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

python finetune_enhanced_infer.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --input_filename "$INPUT_FILENAME" \
    --label_filename "$LABEL_FILENAME" \
    --split test \
    --batch_size 32 \
    --num_vis 30 \
    --viz_format pdf \
    --viz_clip_percentile "$VIZ_CLIP_PERCENTILE"

echo "Inference completed. Results saved to: $OUTPUT_DIR"
