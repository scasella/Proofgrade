#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_ROOT="${ROOT_DIR}/analysis/outputs/meta_transfer_smoke/fixture"
OUTPUT_DIR="${ROOT_DIR}/analysis/outputs/meta_transfer_smoke"

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

"${PYTHON_BIN}" - <<'PY'
from analysis.smoke_data import create_meta_transfer_fixture
create_meta_transfer_fixture("analysis/outputs/meta_transfer_smoke/fixture")
PY

"${PYTHON_BIN}" analysis/build_lineage_dataset.py \
  --auto_discover "${FIXTURE_ROOT}/source_runs" "${FIXTURE_ROOT}/target_runs" "${FIXTURE_ROOT}/selector_runs" \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" analysis/select_transfer_agents.py \
  --auto_discover "${FIXTURE_ROOT}/source_runs" \
  --selector descendant_growth \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" analysis/reproduce_transfer_figures.py \
  --auto_discover "${FIXTURE_ROOT}/target_runs" "${FIXTURE_ROOT}/selector_runs" \
  --output_dir "${OUTPUT_DIR}" \
  --report_path "${OUTPUT_DIR}/reproduce_transfer_results_smoke.md"

"${PYTHON_BIN}" analysis/run_transfer_ablation.py \
  --auto_discover_sources "${FIXTURE_ROOT}/source_runs" \
  --summary_run_dirs "${FIXTURE_ROOT}/target_runs"/generate_smoke_* "${FIXTURE_ROOT}/selector_runs"/generate_smoke_* \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_meta_transfer.py'
