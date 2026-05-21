# Local AI Models for Businesses Experiment

A validation harness that benchmarks open-source LLMs on six real-world business workflows commonly found in SME operations and engineering teams. The goal: measure whether locally-hosted open models on commodity GPU hardware can reliably handle the assistive-AI tasks that small and mid-sized businesses actually need — and identify the cheapest hardware tier that still works.

## Why this exists

Public LLM benchmarks (MMLU, HumanEval, GPQA) measure general capability. They don't answer the question a 50-person manufacturer or engineering firm actually asks: *"Can a 70B model running on a workstation I can buy replace the repetitive knowledge work my team does every week?"*

This experiment bridges that gap with six task types drawn from real ICP (Ideal Customer Profile) workflows, grounded in actual consulting engagements with manufacturing and engineering SMEs.

## What's tested

| Workflow | Real-world analog | What it tests |
|---|---|---|
| `compliance` | Quality engineer comparing an incoming material certificate against a customer specification | Multi-attribute compliance reasoning, tolerance logic, structured flag-with-justification output |
| `dims` | Production engineer cross-checking drawn dimensions against CAM-extracted dims | Tabular comparison, 10%-tolerance-band rule, false-positive cost awareness |
| `dims_ocr` | Same engineer extracting a dim+tol table from a drawing without retyping | OCR + VLM grounding, table reconstruction, units/tolerance-frame parsing |
| `docs` | Operations manager pulling structured fields out of a mixed pile of PDFs (English native + German scanned) | OCR fallback, multilingual NER, schema-conformant JSON output |
| `schedule_read` | PM scanning a project plan for issues humans miss | Spreadsheet ingestion, multi-row dependency reasoning |
| `schedule_write` | PM updating a plan after a supplier delay, with cascade propagation | Multi-cell coordinated write, protected-region awareness |

Coverage: text/data, image/vision, PDF (native + scanned OCR), spreadsheet read, spreadsheet write.

## Model ladder

**Text workflows** (compliance, dims, docs, schedule_read, schedule_write):

| Model | Size | Hardware tier (Q4 inference) |
|---|---|---|
| `openai/gpt-oss-120b` | 120B MoE | Tier 2 (~60 GB VRAM) |
| `qwen/qwen-2.5-72b-instruct` | 72B | Tier 1+ (~40 GB) |
| `meta-llama/llama-3.3-70b-instruct` | 70B | Tier 1+ (~40 GB) |
| `z-ai/glm-5.1` | ? (unverified) | Tier 1+ |
| `openai/gpt-oss-20b` | 20B | Tier 0.5 (~12 GB) |
| `qwen/qwen3.5-9b` | 9B | Tier 0 (~6 GB) |

**Vision workflows** (dims_vlm, docs_vlm) use a separate VLM ladder routed via OpenRouter -- see `run_experiment.py` for the full list.

**Hardware tiers:**

| Tier | Example GPU | Approx. cost | Runs |
|---|---|---|---|
| Tier 2 | 2x RTX 6000 Ada (96 GB) | ~16-18k | Everything including 120B MoE |
| Tier 1 | 1x RTX 6000 Ada (48 GB) | ~9-11k | 70-72B comfortably |
| Tier 0.5 | 1x RTX 4090 / RTX 5000 Ada (24-32 GB) | ~4-6k | 32B comfortably |
| Tier 0 | 1x RTX 4070 Ti Super (16 GB) | ~1.5-2.5k | 7-9B class |

The key question: if a smaller model still hits acceptable precision/recall, can we recommend a cheaper hardware tier?

## Setup

1. **Install [uv](https://docs.astral.sh/uv/):**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh                                     # macOS / Linux
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Configure credentials.** Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   You need at minimum a [Tensorix](https://tensorix.ai) API key. OpenRouter is optional (only required if you opt into OpenRouter-hosted models via `--models`).

4. **Verify input data** is present under `data/` (see [Project structure](#project-structure) below).

## Usage

```bash
# Smoke test -- 1 run per combination (~64 LLM calls, 5-10 minutes)
uv run python run_experiment.py --runs 1

# Full statistical run -- 10 runs per combination (~640 LLM calls, 60-120 minutes)
uv run python run_experiment.py --runs 10

# Skip OCR-heavy dims_ocr (text + PDF + xlsx workflows only)
uv run python run_experiment.py --runs 10 --no-extract

# Single workflow
uv run python run_experiment.py --workflow schedule_write --runs 10
uv run python run_experiment.py --workflow docs --pdfs llm_finetuning_report.pdf llm_finetuning_report_scanned.pdf --runs 5

# Override model list
uv run python run_experiment.py --runs 10 --models meta-llama/llama-3.3-70b-instruct qwen/qwen3.5-9b
```

## Outputs

Each run produces:

- `results/run_<timestamp>.json` -- full raw results + per-combo aggregates (mean/std/min/max for precision, recall, F1, latency, plus raw model outputs)
- `results/summary_<timestamp>.md` -- side-by-side markdown table grouped by workflow
- **stdout** -- compact summary table

## How to read the numbers

| Metric | Meaning |
|---|---|
| Precision | Of items the model flagged/extracted, what fraction were correct? |
| Recall | Of items it should have flagged/extracted, what fraction did it catch? |
| F1 | Harmonic mean of precision and recall |
| Latency mean (s) | Average end-to-end LLM call time per run |

**Verdict heuristics:**

- Both >= 0.85, std < 0.10 -- **strong pass** (capable of human-in-the-loop pilot)
- Both >= 0.70 -- **pass** (feasible with attention to false-positive triage)
- Either < 0.70 -- **weak / fail** (needs prompt iteration or a different model)

For `schedule_write`: the `forbidden_row_violations` field in the raw JSON shows whether the model touched rows it was told not to -- a real-world risk signal even if precision/recall look fine.

## Project structure

```
agen-ops-6/
├── README.md
├── LICENSE                    # MIT (code) + CC-BY-4.0 (data/prompts)
├── run_experiment.py          # The harness (multi-backend, multi-model, multi-run, 8 workflows)
├── probe_models.py            # Model-probe utility
├── pyproject.toml             # uv project (openai, easyocr, pillow, numpy, openpyxl, pymupdf)
├── .env.example               # Credential template
├── .gitignore
├── docs/
│   └── METHODOLOGY.md         # Scoring rubrics, sampling settings, limitations
├── prompts/                   # One system prompt per workflow
├── data/
│   ├── compliance/            # Spec + cert + ground truth (synthetic)
│   ├── dims/                  # Dimension CSVs + engineering drawings + ground truth
│   ├── docs/                  # 3 PDF variants: native, scanned, image-based + ground truth
│   ├── schedule_read/         # Excel timeline with injected scheduling issues + ground truth
│   └── schedule_write/        # Baseline schedule + delay scenario + ground truth
└── results/                   # Run JSONs + summary MDs land here
```

## Protocol

- **N=10 runs** per (model, task) combination for statistical reporting.
- **Temperature / sampling**: fixed across runs (see `run_experiment.py` defaults).
- **EasyOCR** runs locally (CPU is fine for the test inputs) and triggers automatically on scanned PDFs and engineering drawings.
- Hardware tier reasoning assumes **Q4 quantization**. If you use FP8/FP16, re-check VRAM requirements.

For full scoring rubrics, ground-truth construction methodology, and known limitations, see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## License

Code: MIT. Data and prompts: CC-BY-4.0. See [`LICENSE`](LICENSE) for full text.

## Author

[Agenovation](https://agenovation.ai) -- achim@agenovation.ai
