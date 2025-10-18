#!/usr/bin/env bash
set -euo pipefail

# ==== Environment ====
export PATH="${CONDA_PREFIX:-$HOME/miniconda3}/bin:$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1

# GPUs / NCCL
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Accelerate/DeepSpeed config (여기에 ZeRO-2 설정이 있어야 DeepSpeed 적용)
ACC_CONFIG_FILE=acc_zero2_config.yaml

# ==== Data & Model ====
MODEL_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
TOKENIZER_NAME="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
TRAIN_DATA="/mnt/hdd/taehpark/LLM_finetuning/k12data/Q1_data/train.json"

# ==== Output ====
OUTPUT_DIR="./outputs/solar_설명글1_with_prompting_full_t3_lora"

# ==== Hyperparameters (당신이 요청한 값 그대로) ====
TRAIN_BATCH_SIZE=1
DEV_BATCH_SIZE=1
LR=2e-4
MAX_SOURCE_LEN=1024
MAX_TARGET_LEN=5
SEED=42
EPOCHS=10
EVAL_STRATEGY=steps
EVAL_STEPS=1000
SAVE_STRATEGY=steps
SAVE_STEPS=1000
EVAL_ACC_STEP=1
TRAIN_ACC_STEP=16
LOGGING_STEP=50
NUM_WORKER=4

# (옵션) 4bit 사용: 켜려면 "--use-4bit" 로 바꾸고, 끄려면 빈 문자열로 둬도 됨
USE_4BIT=""

# (옵션) intent level
INTENT_LEVEL="metaIntent"   # metaIntent | intent

mkdir -p "${OUTPUT_DIR}" log

echo "[trainer.sh] GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[trainer.sh] Output: ${OUTPUT_DIR}"
echo "[trainer.sh] Accelerate Config: ${ACC_CONFIG_FILE}"

accelerate launch --config_file="${ACC_CONFIG_FILE}" trainer.py \
  --model "${MODEL_NAME}" \
  --tokenizer "${TOKENIZER_NAME}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --dev-batch-size "${DEV_BATCH_SIZE}" \
  --lr "${LR}" \
  --max-source-len "${MAX_SOURCE_LEN}" \
  --max-target-len "${MAX_TARGET_LEN}" \
  --seed "${SEED}" \
  --data-path "${TRAIN_DATA}" \
  --output-path "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --eval-strategy "${EVAL_STRATEGY}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-strategy "${SAVE_STRATEGY}" \
  --save-steps "${SAVE_STEPS}" \
  --eval-acc-step "${EVAL_ACC_STEP}" \
  --train-acc-step "${TRAIN_ACC_STEP}" \
  --logging-step "${LOGGING_STEP}" \
  --num-worker "${NUM_WORKER}" \
  --intent-level "${INTENT_LEVEL}" \
  ${USE_4BIT}
