#!/usr/bin/env bash
set -euo pipefail

############################################
# ==== Paths (직접 입력해서 사용) ====
############################################
BASE_MODEL="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"   # 학습 때 쓴 베이스 모델
ADAPTER_DIR="/mnt/hdd/taehpark/LLM_finetuning_naver/outputs/solar_설명글1_with_prompting_full_t3_lora/5"                  # 학습 결과 LoRA 어댑터 디렉토리 (예: saved_model/<epoch>)
TOKENIZER="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"                                    # 비워두면 BASE_MODEL 토크나이저 사용
INPUT_JSON="/mnt/hdd/taehpark/LLM_finetuning_naver/k12data/Q1_data/test.json"                       # 네가 준 JSON 배열 파일 (response, point)
OUTPUT_CSV="/mnt/hdd/taehpark/LLM_finetuning_naver/outputs/solar_설명글1_with_prompting_full_t3_lora/5/out.csv"                      # 저장할 CSV 경로

############################################
# ==== Inference Hyperparameters ====
############################################
MAX_NEW_TOKENS=5
TEMPERATURE=0.0
TOP_P=1.0
USE_4BIT=1            # 1이면 --use-4bit 사용, 0이면 미사용

############################################
# ==== Runtime / Env ====
############################################
CUDA_ID=0,1            # 사용할 GPU ID (쉼표로 구분)
PYFILE="inference.py"   # 배치 추론 파이썬 파일명

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "${OUTPUT_CSV}")"

python3 "${PYFILE}" \
  --base-model "${BASE_MODEL}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --input-json "${INPUT_JSON}" \
  --output-csv "${OUTPUT_CSV}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  $( [[ -n "${TOKENIZER}" ]] && printf -- ' --tokenizer %q' "${TOKENIZER}" ) \
  $( [[ "${USE_4BIT}" == "1" ]] && printf -- ' --use-4bit' )
