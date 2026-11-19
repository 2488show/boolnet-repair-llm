"""One interface for every language model used in the project.

Backends:
  gemini:<model>  Google Gemini REST API; key read from GEMINI_API_KEY (or the .env file)
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


def _key() -> str:
    if "GEMINI_API_KEY" not in os.environ:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip()
    return os.environ["GEMINI_API_KEY"]


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


def _local(model, prompt, temperature, max_tokens, json_mode):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(LOCAL_URL, json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def ask(spec: str, prompt: str, temperature: float = 0.0, max_tokens: int = 2048,
        json_mode: bool = False, tag: str = "") -> str:
    backend, model = spec.split(":", 1)
    t0 = time.time()
    fn = {"gemini": _gemini, "local": _local}[backend]
    text, n_in, n_out = fn(model, prompt, temperature, max_tokens, json_mode)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(dict(time=t0, spec=spec, tag=tag, temperature=temperature, prompt=prompt,
                                reply=text, tokens_in=n_in, tokens_out=n_out,
                                seconds=round(time.time() - t0, 2)), ensure_ascii=False) + "\n")
    return text
