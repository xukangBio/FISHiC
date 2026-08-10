#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="./Data/SC_enhance_data"
PRETRAINED_CKPT="./pretrain_model/best_model.pt"
OUTPUT_DIR="./output_finetune_enhanced_sc_hic"

BATCH_SIZE=32
NUM_EPOCHS=20
LR=1e-4

INPUT_FILENAME="lr_extra.npy"
LABEL_FILENAME="hr_extra.npy"

echo "=================================================="
echo "FISHi-C scHi-C enhancement fine-tuning"
echo "Data path: $DATA_PATH"
echo "Input file: $INPUT_FILENAME"
echo "Label file: $LABEL_FILENAME"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

python finetune_enhanced_train.py \
    --data_path "$DATA_PATH" \
    --pretrained_checkpoint "$PRETRAINED_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --input_filename "$INPUT_FILENAME" \
    --label_filename "$LABEL_FILENAME" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --save_interval 5 \
    --lambda_l1 1.0 \
    --lambda_l2 1.0 \
    --lambda_ssim 0.5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0

echo "Fine-tuning completed. Results saved to: $OUTPUT_DIR"
