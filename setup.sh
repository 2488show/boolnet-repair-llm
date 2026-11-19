#!/bin/zsh
# Rebuild the environment from scratch on an Apple-silicon Mac.
set -e
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.uv-cache" HF_HOME="$PWD/.hf-cache"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# Benchmark networks: Biodivine Boolean Models, pinned to the commit used in this project.
# Model copyright stays with the original authors; the repository only downloads them.
if [ ! -d vendor/bbm ]; then
  git clone https://github.com/sybila/biodivine-boolean-models vendor/bbm
fi
git -C vendor/bbm checkout -q 314b2fed94689c38d3aaaf8101ed622b3501c7f7
# Local model for the small-model condition (about 4.5 GB, downloaded on first use).
.venv/bin/python -m mlx_lm generate --model mlx-community/Qwen3-8B-4bit --prompt "hi /no_think" --max-tokens 5 >/dev/null
echo "Environment ready. Put GEMINI_API_KEY=... in .env (never commit it)."
