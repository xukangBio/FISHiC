#!/usr/bin/env bash

DATA_PATH="./Data/Pretrain_data"
OUTPUT_DIR="./pretrain_model"

BATCH_SIZE=32
NUM_EPOCHS=80
LR=1e-4

MODEL_TYPE="unet"
BASE_CHANNELS=64
TIMESTEPS=1000

echo "=================================================="
echo "FISHi-C pretraining"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

python pretrain.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --model_type "$MODEL_TYPE" \
    --base_channels "$BASE_CHANNELS" \
    --timesteps "$TIMESTEPS" \
    --save_interval 5 \
    --loss_type l1+ssim

echo "Pretraining completed. Results saved to: $OUTPUT_DIR"
