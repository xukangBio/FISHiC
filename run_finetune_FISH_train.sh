#!/usr/bin/env bash

DATA_PATH="./Data/FISH_impute_data"
PRETRAINED_CKPT="./pretrain_model/best_model.pt"
OUTPUT_DIR="./output_finetune_FISHImpute"

BATCH_SIZE=16
NUM_EPOCHS=20
LR=1e-4

echo "=================================================="
echo "FISHi-C FISH imputation fine-tuning"
echo "Data path: $DATA_PATH"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

python finetune_FISH_impute_train.py \
    --data_path $DATA_PATH \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --save_interval 5 \
    --min_mask_ratio 0.2 \
    --max_mask_ratio 0.8 \
    --observed_loss_weight 0.1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    # --multi_gpu


echo "Fine-tuning completed. Results saved to: $OUTPUT_DIR"
