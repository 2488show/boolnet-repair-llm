#!/bin/zsh
# GPT-6-Luna through the Codex CLI (ChatGPT plan): run every Stage 3b task (all Stage 3a tasks + 150 sampled
# original tasks). Finished jobs are skipped; jobs that fail (for example when the plan's 5-hour usage window
# is exhausted) are retried in the next pass after a pause.
cd "${0:A:h}/.."
for i in {1..40}; do
  echo "=== pass $i $(date '+%F %T')" >> logs/stage3b_codex_full.log
  .venv/bin/python scripts/stage3b_run.py --spec codex:gpt-6-luna --sets reg --threads ${THREADS:-2} >> logs/stage3b_codex_full.log 2>&1; a=$?
  .venv/bin/python scripts/stage3b_run.py --spec codex:gpt-6-luna --sets orig --sample 150 --threads ${THREADS:-2} >> logs/stage3b_codex_full.log 2>&1; b=$?
  if [ $a -eq 0 ] && [ $b -eq 0 ]; then echo "ALL_DONE $(date '+%F %T')" >> logs/stage3b_codex_full.log; exit 0; fi
  sleep 1800
done
