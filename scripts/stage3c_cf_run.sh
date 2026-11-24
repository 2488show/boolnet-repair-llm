#!/bin/zsh
# Counterfactual tasks with GPT-6-Luna (Codex CLI); failed jobs retried after a pause.
cd "${0:A:h}/.."
for i in {1..20}; do
  .venv/bin/python scripts/stage3b_run.py --spec codex:gpt-6-luna --sets cf --threads ${THREADS:-4} >> logs/stage3c_cf.log 2>&1 \
    && { echo "ALL_DONE $(date '+%F %T')" >> logs/stage3c_cf.log; exit 0; }
  sleep 1800
done
