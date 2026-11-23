"""One interface for every language model used in the project.

Backends:
  gemini:<model>  Google Gemini REST API; key read from GEMINI_API_KEY (or the .env file)
  deepseek:<model> DeepSeek API (OpenAI-compatible); key read from DEEPSEEK_API_KEY (or the .env file)
  codex:<model>   OpenAI model through the Codex CLI signed in with the user's ChatGPT plan (no API key):
                  one non-interactive turn in an empty folder, read-only sandbox, no web search, apps or
                  browser, user config and rules ignored. Codex adds its own agent instructions and does not
                  expose temperature, so this condition is not identical to a plain API call.
  local:<model>   a local MLX server with an OpenAI-compatible endpoint (mlx_lm.server)

Every call is appended to runs/llm_log.jsonl (prompt, reply, token counts, latency) so that
results can be audited and re-analysed without calling the model again. The key is never logged.
"""
import json
import os
import pathlib
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "llm_log.jsonl"
LOCAL_URL = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")


def _key(name: str = "GEMINI_API_KEY") -> str:
    if name not in os.environ:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith(name + "="):
                    os.environ[name] = line.split("=", 1)[1].strip()
    return os.environ[name]


def _gemini(model, prompt, temperature, max_tokens, json_mode):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    for attempt in range(6):
        r = requests.post(url, json=body, headers={"x-goog-api-key": _key()}, timeout=180)
        if r.status_code in (429, 500, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        d = r.json()
        parts = d["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = d.get("usageMetadata", {})
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    r.raise_for_status()


def _deepseek(model, prompt, temperature, max_tokens, json_mode):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        r = requests.post("https://api.deepseek.com/chat/completions", json=body, timeout=600,
                          headers={"Authorization": f"Bearer {_key('DEEPSEEK_API_KEY')}"})
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    r.raise_for_status()


CODEX = os.environ.get("CODEX_BIN", "/Applications/Codex.app/Contents/Resources/codex-cli/bin/codex")
CODEX_EFFORT = os.environ.get("CODEX_EFFORT", "medium")


def _codex(model, prompt, temperature, max_tokens, json_mode):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "last.txt"
        work = pathlib.Path(tmp) / "work"
        work.mkdir()
        cmd = [CODEX, "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "-s", "read-only", "-C", str(work), "-m", model, "-c", f'model_reasoning_effort="{CODEX_EFFORT}"',
               "-c", 'web_search="disabled"', "--disable", "apps", "--disable", "browser_use",
               "--json", "-o", str(out), "-"]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
        n_in = n_out = 0
        for line in r.stdout.splitlines():
            if '"usage"' in line:
                try:
                    u = json.loads(line).get("usage", {})
                    n_in, n_out = u.get("input_tokens", 0), u.get("output_tokens", 0)
                except json.JSONDecodeError:
                    pass
            if '"web_search"' in line:
                raise RuntimeError("codex used web search")
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex exit {r.returncode}: {r.stderr[-300:]}")
        return out.read_text(), n_in, n_out


def _local(model, prompt, temperature, max_tokens, json_mode):
    if "qwen3" in model.lower():
        prompt += " /no_think"  # Qwen3 thinking would make an 8B model on a laptop far too slow
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(LOCAL_URL, json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    text = d["choices"][0]["message"]["content"]
    text = text.split("</think>")[-1]  # drop an (empty) thinking block if the model emits one
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def ask(spec: str, prompt: str, temperature: float = 0.0, max_tokens: int = 2048,
        json_mode: bool = False, tag: str = "") -> str:
    backend, model = spec.split(":", 1)
    t0 = time.time()
    fn = {"gemini": _gemini, "deepseek": _deepseek, "codex": _codex, "local": _local}[backend]
    text, n_in, n_out = fn(model, prompt, temperature, max_tokens, json_mode)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(dict(time=t0, spec=spec, tag=tag, temperature=temperature, prompt=prompt,
                                reply=text, tokens_in=n_in, tokens_out=n_out,
                                seconds=round(time.time() - t0, 2)), ensure_ascii=False) + "\n")
    return text
