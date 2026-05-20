#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run a long training job unattended and emit a single, machine-readable
# terminal-state line when it finishes.
#
# Why this exists
# ---------------
# A 3-seed comparison on Modal takes hours. Nobody (and no agent) should sit
# and watch it. Launch this as a background task and walk away:
#
#     bash scripts/run_and_notify.sh modal              # full 3-seed run
#     bash scripts/run_and_notify.sh modal --steps 500 --seeds 0   # smoke test
#     bash scripts/run_and_notify.sh local --device cpu            # local
#
# It streams logs to runs/<job>.log, then prints exactly one of:
#     EXPERIMENT_OK     <job>   — finished cleanly
#     EXPERIMENT_FAILED <job>   — crashed (last 40 log lines echoed)
#
# The exit code mirrors the job: 0 on success, non-zero on failure. When this
# script is run as an agent background task, the agent is re-invoked ONLY on
# that exit — dormant (zero tokens) for the whole multi-hour run.
# ---------------------------------------------------------------------------
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 99
JOB="${1:-modal}"; shift || true
mkdir -p runs
LOG="runs/${JOB}_run.log"
STAMP() { date -Is; }

echo "[$(STAMP)] START job=$JOB args=$*" | tee "$LOG"

case "$JOB" in
  modal)
    .venv/bin/modal run modal_train.py "$@" 2>&1 | tee -a "$LOG"
    ;;
  local)
    .venv/bin/python -m scripts.run_comparison --out runs/local "$@" 2>&1 | tee -a "$LOG"
    ;;
  *)
    echo "[$(STAMP)] unknown job '$JOB' (use: modal | local)" | tee -a "$LOG"
    exit 98
    ;;
esac
CODE=${PIPESTATUS[0]}

if [ "$CODE" -eq 0 ]; then
  echo "[$(STAMP)] EXPERIMENT_OK $JOB" | tee -a "$LOG"
else
  echo "[$(STAMP)] EXPERIMENT_FAILED $JOB exit=$CODE" | tee -a "$LOG"
  echo "----- last 40 log lines -----"
  tail -n 40 "$LOG"
fi
exit "$CODE"
