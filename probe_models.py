"""
Probe Tensorix, OpenRouter, and OpenAI for available models matching keywords of interest.

Usage:
    uv run python probe_models.py                 # all VLM-relevant models
    uv run python probe_models.py qwen3 qwen3.5   # narrow by keyword(s)
    uv run python probe_models.py --check-credits  # check OpenRouter credit balance

Reads API keys from .env. Calls each backend's /models endpoint and prints the
matching model ids plus context length and pricing if exposed.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
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
    "gpt-5",
    "claude",
    "opus",
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


def check_openrouter_credits() -> None:
    """Check OpenRouter credit balance via /api/v1/auth/key."""
    print(f"\n{'='*78}\nOpenRouter Credit Check\n{'='*78}")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("  (skipped — OPENROUTER_API_KEY not set)")
        return
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read().decode())
            data = info.get("data", {})
            label = data.get("label", "?")
            limit_remaining = data.get("limit_remaining")
            usage = data.get("usage")
            print(f"  Key label:  {label}")
            if limit_remaining is not None:
                print(f"  Remaining:  ${limit_remaining:.2f}")
                if limit_remaining < 1.0:
                    print(f"  WARNING: Balance is low. Top up before a full run.")
            elif usage is not None:
                print(f"  Usage:      ${usage:.4f}")
                print(f"  (No hard limit — pay-as-you-go)")
            else:
                print(f"  (Could not determine balance)")
    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--check-credits" in flags:
        check_openrouter_credits()
        if not args and len(flags) == 1:
            return  # only credit check requested

    keywords = args if args else DEFAULT_KEYWORDS
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
    probe(
        "OpenAI (direct)",
        "https://api.openai.com/v1",
        os.environ.get("OPENAI_API_KEY"),
        keywords,
    )

    if "--check-credits" in flags:
        check_openrouter_credits()


if __name__ == "__main__":
    main()
