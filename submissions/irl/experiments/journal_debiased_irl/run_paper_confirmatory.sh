#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/Users/larsvanderlaan/repos/RLtools/.venv/bin/python"
FROZEN_FORE="${SCRIPT_DIR}/artifacts/fore_selection_pilot/frozen_fore_config.json"
FROZEN_G="${SCRIPT_DIR}/artifacts/data_fusion_summary_gamma80_v4_shared/frozen_outcome_regression.pkl"
FUSION_PILOT_MANIFEST="${SCRIPT_DIR}/artifacts/data_fusion_paper_pilot_v1/data_fusion_manifest.json"
FROZEN_FORE_SHA256="ba5810fa12774b558eecb7fba2c7c8b9ce575e0b1a796179fa3124a7fb87d862"
FROZEN_G_SHA256="fcb34bff7c0909100e8a6120bd1e2349bdffdd538067933a2efc721ed9275b0e"
FUSION_PILOT_MANIFEST_SHA256="97030d7d631894f113e0b2dc35032583c7818ed7379095fcfe079ef30e09b760"
PAPER_PROTOCOL_ID="jasa-neural-fore-v2-corrected"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rltools-jasa-matplotlib"
mkdir -p "${MPLCONFIGDIR}"

COMMON_MAIN_ARGS=(
  --mode monte-carlo
  --repetitions 300
  --ratio-mode neural-fore
  --crossfit-folds 5
  --crossfit-se-method iid
  --crossfit-ci-method normal
  --behavior-policy-design quadratic-logit
  --main-sieve-mode fixed-quadratic
  --main-sieve-degree 2
  --main-sieve-c 10
  --main-grid-points 41
  --fore-iteration-budgets 30 100 300
  --fore-max-training-rows 20000
  --fore-frozen-config "${FROZEN_FORE}"
  --expected-fore-sha256 "${FROZEN_FORE_SHA256}"
  --paper-protocol-id "${PAPER_PROTOCOL_ID}"
  --jobs 4
  --resume
)

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_jrssb_simulation.py" \
  "${COMMON_MAIN_ARGS[@]}" \
  --examples 1a \
  --sample-sizes 2500 5000 10000 \
  --example1a-policy-estimator sieve-logit \
  --output-dir "${SCRIPT_DIR}/artifacts/paper_confirmatory_v2_example1a"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_jrssb_simulation.py" \
  "${COMMON_MAIN_ARGS[@]}" \
  --examples 1b \
  --sample-sizes 25000 50000 100000 \
  --example1b-policy-estimator sieve-logit \
  --output-dir "${SCRIPT_DIR}/artifacts/paper_confirmatory_v2_example1b"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_jrssb_simulation.py" \
  --mode data-fusion-confirmatory \
  --repetitions 300 \
  --sample-sizes 2500 5000 10000 \
  --crossfit-folds 5 \
  --data-fusion-policy-mode known-logging \
  --data-fusion-transition-mode sieve \
  --data-fusion-g-mode frozen \
  --data-fusion-ratio-mode neural-fore \
  --data-fusion-target-gamma 0.80 \
  --fore-iteration-budgets 30 100 300 \
  --fore-max-training-rows 20000 \
  --fore-frozen-config "${FROZEN_FORE}" \
  --expected-fore-sha256 "${FROZEN_FORE_SHA256}" \
  --fusion-g-cache "${FROZEN_G}" \
  --expected-fusion-g-sha256 "${FROZEN_G_SHA256}" \
  --fusion-pilot-manifest "${FUSION_PILOT_MANIFEST}" \
  --expected-fusion-pilot-manifest-sha256 "${FUSION_PILOT_MANIFEST_SHA256}" \
  --paper-protocol-id "${PAPER_PROTOCOL_ID}" \
  --jobs 4 \
  --resume \
  --output-dir "${SCRIPT_DIR}/artifacts/paper_confirmatory_v2_data_fusion"

"${PYTHON_BIN}" "${SCRIPT_DIR}/assemble_paper_results.py"
