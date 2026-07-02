#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_results/evaluation}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/metrics_summary.csv}"
DATASETS_CSV="${DATASETS:-golden,metamdb}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MODE="${EVAL_MODE:-pretrained_zero_shot}"
BACKEND="${BACKEND:-chython_reset_mapping}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RUN_TRAINING="${RUN_TRAINING:-0}"
TRAIN_MODES_CSV="${TRAIN_MODES:-scratch,fine_tuned}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-42}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/lukas/tmp/uv-cache}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/lukas/tmp/chython-rxnmap-venv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/scratch/lukas/tmp/uv-python}"

if ! mkdir -p "${UV_CACHE_DIR}" 2>/dev/null; then
  UV_CACHE_DIR="/scratch/lukas/tmp/uv-cache"
fi
if ! mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")" 2>/dev/null; then
  UV_PROJECT_ENVIRONMENT="/scratch/lukas/tmp/chython-rxnmap-venv"
fi
if ! mkdir -p "${UV_PYTHON_INSTALL_DIR}" 2>/dev/null; then
  UV_PYTHON_INSTALL_DIR="/scratch/lukas/tmp/uv-python"
fi

export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR
export LOCALMAPPER_CGRTOOLS_IGNORE="${LOCALMAPPER_CGRTOOLS_IGNORE:-1}"
export LOCALMAPPER_CGR_MP_CONTEXT="${LOCALMAPPER_CGR_MP_CONTEXT:-fork}"
export LOCALMAPPER_CGR_TIMEOUT_SECONDS="${LOCALMAPPER_CGR_TIMEOUT_SECONDS:-2}"

IFS=',' read -r -a DATASET_NAMES <<< "${DATASETS_CSV}"
IFS=',' read -r -a TRAIN_MODES <<< "${TRAIN_MODES_CSV}"

mkdir -p "${OUTPUT_DIR}"
rm -f "${SUMMARY_CSV}"

for dataset in "${DATASET_NAMES[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  if [[ -z "${dataset}" ]]; then
    continue
  fi

  if [[ "${RUN_TRAINING}" == "1" ]]; then
    for mode in "${TRAIN_MODES[@]}"; do
      mode="$(echo "${mode}" | xargs)"
      if [[ -z "${mode}" ]]; then
        continue
      fi
      uv run python -m nora.train \
        --dataset "${dataset}" \
        --mode "${mode}" \
        --data_root "${DATA_ROOT}" \
        --batch_size "${BATCH_SIZE}" \
        --max_epochs "${MAX_EPOCHS}" \
        --seed "${SEED}" \
        --output_json "experiment_results/training/${dataset}_${mode}_seed${SEED}.json" \
        --checkpoint_dir "experiment_results/checkpoints"
    done
  fi

  eval_args=(
    --dataset "${dataset}"
    --split "${EVAL_SPLIT}"
    --data_root "${DATA_ROOT}"
    --backend "${BACKEND}"
    --mode "${EVAL_MODE}"
    --timeout_seconds "${TIMEOUT_SECONDS}"
    --output_dir "${OUTPUT_DIR}"
    --summary_csv "${SUMMARY_CSV}"
  )
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    eval_args+=(--limit "${EVAL_LIMIT}")
  fi

  uv run python -m nora.evaluate "${eval_args[@]}"
done

echo "Wrote summary: ${SUMMARY_CSV}"
