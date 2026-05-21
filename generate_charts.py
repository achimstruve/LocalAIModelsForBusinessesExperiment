"""Generate benchmark result charts for the README.

Usage:
    python generate_charts.py results/run_20260521-154821.json

Writes PNGs into assets/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
# Professional, colour-blind-friendly palette based on Tol's muted scheme.
BG       = "#FAFAFA"
GRID_CLR = "#E0E0E0"
TEXT_CLR  = "#333333"
ACCENT   = "#4477AA"  # blue

# For per-model bars — distinct, muted hues.
MODEL_COLOURS = [
    "#4477AA",  # blue
    "#EE6677",  # rose
    "#228833",  # green
    "#CCBB44",  # olive
    "#66CCEE",  # cyan
    "#AA3377",  # magenta
    "#BBBBBB",  # grey
    "#EE8866",  # orange
    "#44BB99",  # teal
]

# ── Shared styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "text.color":       TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.grid":        True,
    "grid.color":       GRID_CLR,
    "grid.linewidth":   0.5,
})

# Nice short names for models.
def short_name(model_id: str) -> str:
    parts = model_id.split("/")
    name = parts[-1] if len(parts) > 1 else parts[0]
    # Shorten common prefixes
    for prefix in ["llama-", "qwen-", "qwen", "gpt-oss-", "deepseek-", "glm-"]:
        if name.startswith(prefix):
            break
    return name


# Nice short names for workflows.
WF_LABELS = {
    "compliance":     "Compliance",
    "dims":           "Dims",
    "dims_ocr":       "Dims OCR",
    "dims_vlm":       "Dims VLM",
    "docs":           "Docs",
    "docs_vlm":       "Docs VLM",
    "schedule_read":  "Sched Read",
    "schedule_write": "Sched Write",
}


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def model_avg_f1(combos: list[dict]) -> dict[str, tuple[float, float]]:
    """Return {model: (mean_f1, mean_std)} averaged across items (drawings/PDFs)."""
    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in combos:
        m = c["model"]
        agg = c.get("aggregate", {})
        f1 = agg.get("f1", {})
        by_model[m].append((f1.get("mean", 0.0), f1.get("std", 0.0)))
    out = {}
    for m, vals in by_model.items():
        means = [v[0] for v in vals]
        stds = [v[1] for v in vals]
        out[m] = (float(np.mean(means)), float(np.mean(stds)))
    return out


def model_avg_latency(combos: list[dict]) -> dict[str, float]:
    """Return {model: mean_latency} averaged across items."""
    by_model: dict[str, list[float]] = defaultdict(list)
    for c in combos:
        m = c["model"]
        agg = c.get("aggregate", {})
        lat = agg.get("latency_s", {}).get("mean", 0.0)
        if lat > 0:
            by_model[m].append(lat)
    return {m: float(np.mean(vs)) for m, vs in by_model.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: Heatmap — F1 by model × workflow (all 8 workflows, model-averaged)
# ═══════════════════════════════════════════════════════════════════════════
def chart_heatmap(data: dict) -> None:
    workflows_order = ["compliance", "dims", "schedule_read", "schedule_write",
                       "dims_ocr", "docs", "dims_vlm", "docs_vlm"]
    workflows_present = [w for w in workflows_order if w in data["workflows"]]

    # Collect all models across all workflows and their F1s.
    all_models: set[str] = set()
    wf_model_f1: dict[str, dict[str, float]] = {}
    for wf in workflows_present:
        combos = data["workflows"][wf]["combos"]
        avg = model_avg_f1(combos)
        wf_model_f1[wf] = {m: v[0] for m, v in avg.items()}
        all_models.update(avg.keys())

    # Sort models: by max F1 across text workflows descending.
    text_wfs = [w for w in workflows_present if w in ("compliance", "dims", "schedule_read", "schedule_write")]
    def model_sort_key(m):
        scores = [wf_model_f1.get(w, {}).get(m, 0) for w in text_wfs]
        return -np.mean(scores) if scores else 0
    models = sorted(all_models, key=model_sort_key)

    # Build matrix.
    matrix = np.full((len(models), len(workflows_present)), np.nan)
    for j, wf in enumerate(workflows_present):
        for i, m in enumerate(models):
            if m in wf_model_f1.get(wf, {}):
                matrix[i, j] = wf_model_f1[wf][m]

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.55 + 1.5)))

    # Custom colormap: red → yellow → green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#D32F2F", "#FFAB00", "#388E3C"], N=256)
    cmap.set_bad(color="#E0E0E0")

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(workflows_present)))
    ax.set_xticklabels([WF_LABELS.get(w, w) for w in workflows_present], rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m) for m in models], fontsize=10)

    # Annotate cells.
    for i in range(len(models)):
        for j in range(len(workflows_present)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")
            else:
                colour = "white" if v < 0.45 else TEXT_CLR
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=colour)

    # Divider between text and vision workflows.
    n_text = len([w for w in workflows_present if w in ("compliance", "dims", "schedule_read", "schedule_write")])
    if n_text < len(workflows_present):
        ax.axvline(n_text - 0.5, color=TEXT_CLR, linewidth=1.5, linestyle="--", alpha=0.5)

    ax.set_title("F1 Score by Model and Workflow", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("F1 (mean)", fontsize=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    fig.tight_layout()
    fig.savefig(ASSETS / "heatmap_f1_all.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote assets/heatmap_f1_all.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: Grouped bar — text workflows F1 (models side-by-side)
# ═══════════════════════════════════════════════════════════════════════════
def chart_text_workflows(data: dict) -> None:
    text_wfs = ["compliance", "dims", "schedule_read", "schedule_write"]
    text_wfs = [w for w in text_wfs if w in data["workflows"]]

    # Collect models that appear in at least one text workflow.
    all_models: set[str] = set()
    wf_model: dict[str, dict[str, tuple[float, float]]] = {}
    for wf in text_wfs:
        avg = model_avg_f1(data["workflows"][wf]["combos"])
        wf_model[wf] = avg
        all_models.update(avg.keys())

    # Sort models by mean F1 across text workflows (desc).
    def sort_key(m):
        scores = [wf_model.get(w, {}).get(m, (0, 0))[0] for w in text_wfs]
        return -np.mean(scores)
    models = sorted(all_models, key=sort_key)

    n_wf = len(text_wfs)
    n_m = len(models)
    bar_w = 0.8 / n_wf
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(10, n_m * 1.1), 5.5))

    for j, wf in enumerate(text_wfs):
        f1s = [wf_model.get(wf, {}).get(m, (0, 0))[0] for m in models]
        stds = [wf_model.get(wf, {}).get(m, (0, 0))[1] for m in models]
        offset = (j - n_wf / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, f1s, bar_w * 0.9, yerr=stds,
                       label=WF_LABELS.get(wf, wf), color=MODEL_COLOURS[j % len(MODEL_COLOURS)],
                       edgecolor="white", linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("F1 (mean ± std)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Text Workflow Performance by Model")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.axhline(0.85, color="#388E3C", linewidth=0.8, linestyle="--", alpha=0.5, label=None)
    ax.axhline(0.70, color="#FFAB00", linewidth=0.8, linestyle="--", alpha=0.5, label=None)

    fig.tight_layout()
    fig.savefig(ASSETS / "bar_text_workflows.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote assets/bar_text_workflows.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: dims_ocr vs dims_vlm comparison (model-averaged F1)
# ═══════════════════════════════════════════════════════════════════════════
def chart_vision_comparison(data: dict) -> None:
    wfs = ["dims_ocr", "dims_vlm"]
    wfs = [w for w in wfs if w in data["workflows"]]
    if not wfs:
        return

    all_models: set[str] = set()
    wf_model: dict[str, dict[str, tuple[float, float]]] = {}
    for wf in wfs:
        avg = model_avg_f1(data["workflows"][wf]["combos"])
        wf_model[wf] = avg
        all_models.update(avg.keys())

    # Sort by best F1 across either workflow.
    def sort_key(m):
        return -max(wf_model.get(w, {}).get(m, (0, 0))[0] for w in wfs)
    models = sorted(all_models, key=sort_key)
    # Drop models that are all-zero (complete failures).
    models = [m for m in models if any(wf_model.get(w, {}).get(m, (0, 0))[0] > 0 for w in wfs)]

    n_wf = len(wfs)
    n_m = len(models)
    bar_w = 0.35
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(9, n_m * 0.9), 5.5))

    colours = {"dims_ocr": "#4477AA", "dims_vlm": "#EE6677"}
    for j, wf in enumerate(wfs):
        f1s = [wf_model.get(wf, {}).get(m, (0, 0))[0] for m in models]
        stds = [wf_model.get(wf, {}).get(m, (0, 0))[1] for m in models]
        offset = (j - n_wf / 2 + 0.5) * bar_w
        ax.bar(x + offset, f1s, bar_w * 0.9, yerr=stds,
               label=WF_LABELS.get(wf, wf), color=colours.get(wf, MODEL_COLOURS[j]),
               edgecolor="white", linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("F1 (mean ± std, averaged across drawings)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Dimension Extraction: OCR+LLM Hybrid vs Direct VLM")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.axhline(0.85, color="#388E3C", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.70, color="#FFAB00", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(ASSETS / "bar_vision_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote assets/bar_vision_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: docs workflow — F1 by model × PDF type
# ═══════════════════════════════════════════════════════════════════════════
def chart_docs_by_pdf(data: dict) -> None:
    if "docs" not in data["workflows"]:
        return
    combos = data["workflows"]["docs"]["combos"]

    pdf_types = sorted({c["pdf"] for c in combos if c.get("pdf")})
    pdf_short = {p: p.replace("llm_finetuning_report", "").replace(".pdf", "").lstrip("_") or "native" for p in pdf_types}

    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for c in combos:
        m = c["model"]
        p = c.get("pdf", "")
        f1 = c.get("aggregate", {}).get("f1", {}).get("mean", 0.0)
        by_model[m][p] = f1

    # Sort by mean F1 desc.
    def sort_key(m):
        return -np.mean(list(by_model[m].values()))
    models = sorted(by_model.keys(), key=sort_key)

    n_pdf = len(pdf_types)
    n_m = len(models)
    bar_w = 0.8 / n_pdf
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(10, n_m * 1.0), 5.5))

    pdf_colours = ["#4477AA", "#EE6677", "#228833"]
    for j, p in enumerate(pdf_types):
        f1s = [by_model[m].get(p, 0) for m in models]
        offset = (j - n_pdf / 2 + 0.5) * bar_w
        ax.bar(x + offset, f1s, bar_w * 0.9,
               label=pdf_short[p], color=pdf_colours[j % len(pdf_colours)],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("F1 (mean)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Document Entity Extraction by Model and PDF Type")
    ax.legend(title="PDF variant", fontsize=9, framealpha=0.9)
    ax.axhline(0.85, color="#388E3C", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.70, color="#FFAB00", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(ASSETS / "bar_docs_by_pdf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote assets/bar_docs_by_pdf.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5: Latency vs F1 scatter (text workflows)
# ═══════════════════════════════════════════════════════════════════════════
def chart_latency_vs_f1(data: dict) -> None:
    text_wfs = ["compliance", "dims", "schedule_read", "schedule_write"]
    text_wfs = [w for w in text_wfs if w in data["workflows"]]

    fig, ax = plt.subplots(figsize=(9, 6))

    wf_colours = {
        "compliance": "#4477AA", "dims": "#EE6677",
        "schedule_read": "#228833", "schedule_write": "#CCBB44",
    }
    wf_markers = {
        "compliance": "o", "dims": "s",
        "schedule_read": "^", "schedule_write": "D",
    }

    for wf in text_wfs:
        combos = data["workflows"][wf]["combos"]
        for c in combos:
            m = c["model"]
            agg = c.get("aggregate", {})
            f1 = agg.get("f1", {}).get("mean", 0)
            lat = agg.get("latency_s", {}).get("mean", 0)
            if lat <= 0:
                continue
            ax.scatter(lat, f1, c=wf_colours.get(wf, ACCENT),
                       marker=wf_markers.get(wf, "o"), s=60, alpha=0.75,
                       edgecolors="white", linewidths=0.5)

    # Legend entries for workflows.
    for wf in text_wfs:
        ax.scatter([], [], c=wf_colours.get(wf), marker=wf_markers.get(wf),
                   s=60, label=WF_LABELS.get(wf, wf))

    ax.set_xlabel("Latency (seconds, mean)")
    ax.set_ylabel("F1 (mean)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Latency vs F1 — Text Workflows")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.axhline(0.85, color="#388E3C", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.70, color="#FFAB00", linewidth=0.8, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(ASSETS / "scatter_latency_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote assets/scatter_latency_f1.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    if len(sys.argv) < 2:
        # Auto-detect latest results JSON.
        results_dir = ROOT / "results"
        jsons = sorted(results_dir.glob("run_*.json"))
        if not jsons:
            print("ERROR: no results JSON found. Pass path as argument or run the experiment first.", file=sys.stderr)
            sys.exit(1)
        path = jsons[-1]
    else:
        path = Path(sys.argv[1])

    print(f"Loading {path} ...")
    data = load_results(path)
    print(f"  {len(data['workflows'])} workflows, run={data.get('runs_per_combo')}")
    print()
    print("Generating charts:")
    chart_heatmap(data)
    chart_text_workflows(data)
    chart_vision_comparison(data)
    chart_docs_by_pdf(data)
    chart_latency_vs_f1(data)
    print()
    print("Done. Charts written to assets/.")


if __name__ == "__main__":
    main()
