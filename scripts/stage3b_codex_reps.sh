#!/bin/zsh
# Repeat samples 2 and 3 of every Stage 3b job for GPT-6-Luna (Codex CLI), to measure answer-to-answer
# variation (Codex exposes no temperature). Failed jobs are retried in the next pass after a pause.
cd "${0:A:h}/.."
for i in {1..20}; do
  ok=1
  for rep in 2 3; do
    .venv/bin/python scripts/stage3b_run.py --spec codex:gpt-6-luna --sets reg --rep $rep --threads ${THREADS:-6} >> logs/stage3b_codex_reps.log 2>&1 || ok=0
    .venv/bin/python scripts/stage3b_run.py --spec codex:gpt-6-luna --sets orig --sample 150 --rep $rep --threads ${THREADS:-6} >> logs/stage3b_codex_reps.log 2>&1 || ok=0
  done
  [ $ok -eq 1 ] && { echo "ALL_DONE $(date '+%F %T')" >> logs/stage3b_codex_reps.log; exit 0; }
  sleep 1800
done
