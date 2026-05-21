"""
Probe Tensorix and OpenRouter for available models matching keywords of interest.

Usage:
    uv run python probe_models.py                 # all VLM-relevant models
    uv run python probe_models.py qwen3 qwen3.5   # narrow by keyword(s)

Reads TENSORIX_API_KEY/BASE_URL and OPENROUTER_API_KEY/BASE_URL from .env.
Calls each backend's /models endpoint and prints the matching model ids
plus context length and pricing if exposed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Reuse the .env loader from run_experiment.py
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import load_env_file

load_env_file(Path(__file__).parent / ".env")

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: uv sync", file=sys.stderr)
    sys.exit(2)


# Keywords for the SME-AI benchmark model search. Pass cmdline args to override.
DEFAULT_KEYWORDS = [
    "qwen3",        # qwen3-vl-235b, qwen3.5-vl, etc.
    "qwen3.5",
    "qwen-3.5",
    "internvl",
    "intern-vl",
    "glm-4.5v",
    "glm-4.5-v",
    "glm-5",
    "deepseek-vl",
    "deepseek-v4",
    "llama-3.2-vision",
    "llama-4",
    "pixtral",
    "molmo",
    "nvlm",
    "chandra",
    "olmocr",
    "got-ocr",
    "minicpm-v",
    "aria",
    "vl-",
    "-vl",
    "-vision",
    "vision-",
]


def probe(name: str, base_url: str | None, api_key: str | None, keywords: list[str]) -> None:
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    if not api_key or not base_url:
        print(f"  (skipped — {name} env vars not set)")
        return
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=30.0)
        resp = client.models.list()
    except Exception as e:
        print(f"  ERROR listing models: {e}")
        return

    matched = []
    for m in resp.data:
        mid = (m.id or "").lower()
        if any(k.lower() in mid for k in keywords):
            matched.append(m)

    if not matched:
        print(f"  No models matching {keywords} found in {len(resp.data)} total models.")
        # Show a few totally non-matching ones to confirm the endpoint actually returned something
        sample_ids = [m.id for m in resp.data[:8]]
        print(f"  (Sample of available: {sample_ids})")
        return

    print(f"  {len(matched)} matching models out of {len(resp.data)} total:\n")
    for m in sorted(matched, key=lambda x: x.id):
        # Try to surface useful metadata if exposed
        ctx = getattr(m, "context_length", None) or (getattr(m, "top_provider", None) or {}).get("context_length", None) if hasattr(m, "top_provider") else None
        pricing = getattr(m, "pricing", None)
        pricing_str = ""
        if pricing:
            try:
                p = pricing if isinstance(pricing, dict) else pricing.__dict__
                pin = p.get("prompt") or p.get("input")
                pout = p.get("completion") or p.get("output")
                if pin or pout:
                    pricing_str = f"  in=${pin} out=${pout}"
            except Exception:
                pass
        ctx_str = f"  ctx={ctx}" if ctx else ""
        print(f"  {m.id}{ctx_str}{pricing_str}")


def main() -> None:
    keywords = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_KEYWORDS
    print(f"Filtering by keywords: {keywords}")

    probe(
        "Tensorix",
        os.environ.get("TENSORIX_BASE_URL"),
        os.environ.get("TENSORIX_API_KEY"),
        keywords,
    )
    probe(
        "OpenRouter",
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        os.environ.get("OPENROUTER_API_KEY"),
        keywords,
    )


if __name__ == "__main__":
    main()
