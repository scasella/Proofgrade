#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${ROOT_DIR}/outputs}"
SUMMARY_RUN_ROOT="${SUMMARY_RUN_ROOT:-${ROOT_DIR}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/analysis/outputs/meta_transfer_full}"
TARGET_DOMAIN="${TARGET_DOMAIN:-imo_grading}"
SELECTOR="${SELECTOR:-descendant_growth}"
META_MODEL="${META_MODEL:-}"
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-}"
DOCKERFILE_NAME="${DOCKERFILE_NAME:-}"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python interpreter found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

META_MODEL_ARGS=()
if [[ -n "${META_MODEL}" ]]; then
  META_MODEL_ARGS=(--meta_model "${META_MODEL}")
fi

DOCKER_ARGS=()
if [[ -n "${DOCKER_IMAGE_NAME}" ]]; then
  DOCKER_ARGS+=(--docker_image_name "${DOCKER_IMAGE_NAME}")
fi
if [[ -n "${DOCKERFILE_NAME}" ]]; then
  DOCKER_ARGS+=(--dockerfile_name "${DOCKERFILE_NAME}")
fi

"${PYTHON_BIN}" analysis/build_lineage_dataset.py \
  --auto_discover "${SOURCE_RUN_ROOT}" \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" analysis/select_transfer_agents.py \
  --auto_discover "${SOURCE_RUN_ROOT}" \
  --selector "${SELECTOR}" \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" analysis/run_transfer_ablation.py \
  --auto_discover_sources "${SOURCE_RUN_ROOT}" \
  --auto_discover_summary "${SUMMARY_RUN_ROOT}" \
  --selector "${SELECTOR}" \
  --target_domain "${TARGET_DOMAIN}" \
  --use_meta_memory \
  "${META_MODEL_ARGS[@]}" \
  "${DOCKER_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}"

# If Genesis source runs need to be created or validated and local GPUs are unavailable,
# use the Modal runner rather than local CUDA execution.
# Example:
# "${PYTHON_BIN}" scripts/modal_gpu.py --gpu=A10G --timeout=60 -- \
#   "${PYTHON_BIN}" generate_loop.py --domains genesis_go2walking --max_generation 10
