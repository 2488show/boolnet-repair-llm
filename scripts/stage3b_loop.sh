#!/bin/zsh
# Free-tier Gemini: rerun Stage 3b until every job is done. Finished jobs are skipped, failed ones
# (503 "high demand", 429 quota) are retried in the next pass after a pause.
cd "${0:A:h}/.."
for i in {1..300}; do
  echo "=== pass $i $(date '+%F %T')" >> logs/stage3b_loop.log
  .venv/bin/python scripts/stage3b_run.py --sets reg --threads 2 >> logs/stage3b_loop.log 2>&1; a=$?
  .venv/bin/python scripts/stage3b_run.py --sets orig --sample 150 --threads 2 >> logs/stage3b_loop.log 2>&1; b=$?
  if [ $a -eq 0 ] && [ $b -eq 0 ]; then echo "ALL_DONE $(date '+%F %T')" >> logs/stage3b_loop.log; exit 0; fi
  sleep 900
done
