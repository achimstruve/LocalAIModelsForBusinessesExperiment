# METHODOLOGY.md -- AGEN-OPS-6 Benchmark

This document describes the scoring rubrics, sampling settings, ground-truth construction methodology, intended use, and known limitations of the AGEN-OPS-6 benchmark.

---

## 1. Purpose and intended use

AGEN-OPS-6 measures whether open-source and closed-source LLMs can reliably perform six categories of assistive-AI tasks that small and mid-sized manufacturing / engineering firms (10--500 employees) actually need. Open-source models are benchmarked for local deployment on commodity GPU hardware; closed-source models (GPT-5.5, Claude Opus 4.6) serve as reference ceilings. The benchmark is designed for:

- **Procurement teams** evaluating whether a local-hosted open model can replace or augment a specific workflow before committing to hardware or SaaS spend.
- **AI consultants** positioning model recommendations with measurable, operations-grounded evidence rather than generic benchmark scores.
- **Researchers** studying the gap between public leaderboard rankings and domain-specific task performance.

The benchmark is **not** designed for:

- Ranking models on general intelligence or reasoning ability.
- Safety, alignment, or toxicity evaluation.

---

## 2. Task overview

Eight workflow variants, grouped into six core capabilities:

| Workflow | Capability | Input type | Output type |
|---|---|---|---|
| `compliance` | Multi-attribute compliance reasoning | Two text documents (spec + cert) | Structured JSON: list of deviations |
| `dims` | Tabular comparison with tolerance logic | Two CSV tables (drawing + 3D model dims) | Structured JSON: list of flagged dimensions |
| `dims_ocr` | Vision-to-table extraction (OCR + LLM) | Engineering drawing image (JPEG) | Structured JSON: extracted dimension table |
| `dims_vlm` | Vision-to-table extraction (VLM, no OCR) | Engineering drawing image (JPEG) | Structured JSON: extracted dimension table |
| `docs` | Document entity extraction (native + OCR fallback) | PDF (native text, scanned, or image-based) | Structured JSON: list of entities |
| `docs_vlm` | Document entity extraction (VLM, no OCR) | PDF pages rendered to PNG | Structured JSON: list of entities |
| `schedule_read` | Spreadsheet issue detection | Excel timeline (.xlsx) | Structured JSON: list of scheduling issues |
| `schedule_write` | Spreadsheet cascade-update editing | Excel baseline + scenario text | Structured JSON: list of cell edits |

---

## 3. Sampling and inference settings

| Parameter | Value | Notes |
|---|---|---|
| Temperature | 0.1 | Fixed across all runs. Overridable via `--temperature`. |
| Runs per (model, task) cell | 10 | Overridable via `--runs`. |
| Response format | `{"type": "json_object"}` | First attempt uses structured JSON mode. If the API rejects it or the response fails to parse, a plain-text fallback attempt is made (with JSON extracted via regex). |
| Per-call timeout | 150 seconds | Hard wall-clock cap. If the JSON-mode attempt times out, the plain-text fallback is skipped entirely. |
| Max retries | 0 | SDK-level auto-retry is disabled to prevent timeout chains. |
| Seed | Per-call nonce | A unique seed is injected per call to defeat provider-side response caching. |

**EasyOCR settings** (for `dims_ocr` and `docs` OCR fallback):

| Parameter | Value |
|---|---|
| Languages | English, German |
| Device | CPU |
| Paragraph mode | Disabled (raw text boxes) |

**PDF rendering** (for `docs` OCR fallback and `docs_vlm`):

| Parameter | Value |
|---|---|
| DPI | 200 (OCR fallback), 150 (VLM page images) |
| OCR trigger threshold | < 50 characters per page average from native extraction |

---

## 4. Scoring rubrics

All workflows produce precision, recall, and F1 scores. The definitions vary by task.

### 4.1 Compliance

**Ground truth:** 5 expected deviations (DEV-1 through DEV-5), plus 19 compliant fields that must not be falsely flagged. One compliant field (Manganese content) is a deliberate borderline trap (cert value 1.98%, spec max 2.00%).

**Matching rule:** For each model-emitted deviation, the evaluator concatenates `field + rationale + spec_requirement` (lowercased) and checks whether any keyword from a deviation's keyword set appears as a substring. Each deviation has a curated keyword set:

| Deviation | Keywords (any match) |
|---|---|
| DEV-1 (Yield strength) | "yield strength" |
| DEV-2 (Surface roughness) | "surface roughness", "surface finish", "ra " (with space), "0.95", "0.8 micrometre", "0.8 um" |
| DEV-3 (Intergranular corrosion) | "intergranular", "a262" |
| DEV-4 (Charpy impact) | "charpy", "impact at -196", "impact toughness" |
| DEV-5 (Standard revision) | "a276-17", "a276-23", "a479-23", "superseded", "revision", "standard version", "outdated standard" |

**False positives:** Any emitted deviation that does not match any keyword set across all five deviations is counted as a false positive.

**Metrics:**

```
precision = caught / max(1, caught + false_positives)
recall    = caught / 5
f1        = harmonic mean of precision and recall
```

### 4.2 Dims

**Ground truth:** 25 dimensions; 9 must be flagged because `|drawing_value - model_value| > 10% of (tolerance_plus + tolerance_minus)`. 15 are compliant. 1 is informational (surface roughness, expected as N/A from the 3D model).

**Matching rule:** The model emits a list of `flagged_dimensions`, each with a `dim_id`. Matching is by exact dimension ID (e.g., "D002").

**Metrics:**

```
TP = flagged_ids intersection must_flag_ids
FP = flagged_ids minus must_flag_ids
FN = must_flag_ids minus flagged_ids

precision = |TP| / max(1, |TP| + |FP|)
recall    = |TP| / max(1, |TP| + |FN|)
f1        = harmonic mean
```

### 4.3 Dims_ocr and dims_vlm

**Ground truth:** Per-drawing JSON files, each containing 10--13 expected dimensions with fields: `id`, `kind`, `nominal_value`, `tolerance_plus`, `tolerance_minus`, `feature`.

**Matching rule:** Greedy one-to-one matching. For each ground-truth dimension, the evaluator searches the model's extracted dimensions for a match. A match requires:

1. **Kind alignment:** The `kind` field must normalize to the same category. An alias map collapses synonyms (e.g., "od", "outer diameter", "ext. dia." all map to `diameter`; "ra", "surface finish" map to `surface_finish`).
2. **Value match:** `|extracted_value - gt_value| <= 0.05 mm` (for linear dimensions) or `<= 0.01` (for ratios/angles).
3. **Tolerance match (if present):** `|extracted_tol - gt_tol| <= 0.05 mm` for each of `tolerance_plus` and `tolerance_minus`.
4. **String values** (threads, callouts): exact case-insensitive match.

Each ground-truth dimension can match at most one extracted dimension, and vice versa.

**Metrics:**

```
TP = count of matched GT dimensions
FP = count of extracted dimensions not matched to any GT
FN = count of GT dimensions not matched

precision = TP / max(1, TP + FP)
recall    = TP / max(1, TP + FN)
f1        = harmonic mean
```

### 4.4 Docs and docs_vlm

**Ground truth:** Per-PDF JSON files containing expected entities, each with an `id`, `kind`, `value`, and `match_keywords_any_of` list.

**Matching rule:** Greedy one-to-one matching. For each ground-truth entity, the evaluator concatenates each extracted entity's `kind + value + context` (lowercased) and checks whether any keyword from the entity's keyword set appears as a substring. Each GT entity matches at most one extracted entity.

**False positive estimate:** `max(0, extracted_count - TP)`. This is conservative -- it treats every unmatched extraction as a false positive, even if the model found a real entity not in the ground truth.

**Metrics:**

```
precision = TP / max(1, TP + FP_estimate)
recall    = TP / max(1, GT_count)
f1        = harmonic mean
```

### 4.5 Schedule_read

**Ground truth:** 4 expected scheduling issues (ISS-1 through ISS-4), each with a `kind`, involved parts/stages, and a keyword set.

**Matching rule:** For each ground-truth issue, the evaluator builds a normalized blob from each model-emitted issue (`kind + part_id + stages_involved + summary`, lowercased) and counts keyword hits. A match requires:

- At least 2 keyword hits, **or**
- At least 1 keyword hit **and** a matching part ID.

Each GT issue matches at most one model flag.

**Metrics:**

```
TP = count of matched GT issues
FP = count of model flags not matched to any GT
FN = GT_count - TP

precision = TP / max(1, TP + FP)
recall    = TP / max(1, GT_count)
f1        = harmonic mean
```

### 4.6 Schedule_write

**Ground truth:** 6 expected edits (ED-1 through ED-6), plus a list of protected rows that must not be modified.

**Matching rule:** For each GT edit, find a model-emitted edit where:

1. `part_id` matches (case-insensitive).
2. `stage` matches (case-insensitive substring in either direction).
3. `column` matches exactly.
4. Value matches: either exact/substring match (for dates), or at least one keyword present (for status strings like "delay", "rebaselined"). For date fields with `expected_value_alternatives`, the evaluator also accepts any alternative value.

**Weekend edge case (ED-4, ED-5):** The ground truth expects CMM inspection on 2026-05-23 (a Saturday). Models reasoning about business days may correctly set 2026-05-25 (Monday) instead. Both dates are accepted. Manufacturing schedules can legitimately run on weekends, but a model accounting for business days is not wrong either.

**Forbidden row violations:** Any model edit touching a protected row (any column) is flagged as a violation. This is reported separately from precision/recall as `forbidden_row_violations` in the raw output.

**Metrics:**

```
TP = count of matched GT edits
FP = count of model edits not matched to any GT
FN = GT_count - TP

precision = TP / max(1, TP + FP)
recall    = TP / max(1, GT_count)
f1        = harmonic mean
```

---

## 5. Ground-truth construction

All ground-truth files were author-generated (single author: the benchmark creator). The process for each workflow:

| Workflow | GT construction method |
|---|---|
| `compliance` | Spec and cert were written with 5 deliberate deviations injected at known positions. GT is deterministic by construction. One borderline-compliant field (Mn at 1.98% vs. 2.00% max) was added as a false-positive trap. |
| `dims` | Drawing and model CSVs were generated with 9 deliberate discrepancies exceeding the 10% tolerance-band threshold. GT was verified by manual arithmetic on all 25 rows. |
| `dims_ocr` / `dims_vlm` | 6 synthetic engineering drawings were created with known dimension callouts. GT was manually authored per drawing by reading each callout. |
| `docs` / `docs_vlm` | 3 synthetic PDF variants of the same report (native text, scanned, image-based) were generated. GT entities were extracted by the author from the source document before PDF generation. |
| `schedule_read` | A synthetic Gantt timeline was constructed with 4 deliberately injected issues (supplier delay cascade, machine conflict, sequence violation, un-cascaded material delay). GT is deterministic by construction. |
| `schedule_write` | A clean baseline spreadsheet was constructed, and a supplier-delay scenario was written as a forwarded email. The 6 expected edits and protected rows were determined by the author from the scenario facts. |

**Inter-rater reliability:** Not assessed. This is a known limitation (see section 7).

---

## 6. Aggregation and reporting

For each (workflow, model) combination (or (workflow, model, drawing/PDF) triple for multi-input workflows):

1. N runs are executed (default 10).
2. Failed runs are categorized as **infrastructure failures** or **capability failures** (see 6.1).
3. Two score sets are computed: `scores_all_runs` (failures score 0.0) and `scores_successful_only` (failures excluded).
4. Per-combination aggregates include: mean, population standard deviation, 95% bootstrap confidence interval, min, max for precision, recall, F1, and latency.

### 6.1 Infrastructure vs capability failure separation

Failed runs are classified by error message pattern matching:

- **Infrastructure failures:** HTTP 402 (credit exhaustion), 404 (routing failure), 429 (rate limit), timeouts, connection errors, circuit-breaker skips. These reflect provider-side issues, not model limitations.
- **Capability failures:** JSON parse errors, empty content, or other errors where the model received the prompt but produced unusable output.

The summary reports both score sets:
- `scores_all_runs`: All failures score 0.0 on P/R/F1. This penalizes models on unreliable providers.
- `scores_successful_only`: P/R/F1 computed only over runs that produced valid output. This measures model capability independent of provider infrastructure.

The `reliability` block reports total runs, successful runs, infra failures, capability failures, and success rate.

### 6.2 Bootstrap confidence intervals

For each metric (precision, recall, F1), a 95% bootstrap confidence interval is computed:

1. Resample the N metric values with replacement, 10,000 times.
2. Compute the mean of each bootstrap sample.
3. The 2.5th and 97.5th percentiles of the bootstrap means form the 95% CI.

With N=10, CIs are wide. They make the uncertainty visible rather than hiding it behind a point estimate.

### 6.3 Verdict heuristics

**Verdict heuristics** (for human interpretation, not formal thresholds):

| Condition | Verdict |
|---|---|
| Precision >= 0.85 **and** recall >= 0.85, std < 0.10 | Strong pass |
| Precision >= 0.70 **and** recall >= 0.70 | Pass |
| Either metric < 0.70 | Weak / fail |

### 6.4 Provider response headers

The harness captures provider-identifying response headers when available:

| Header | Source | Purpose |
|---|---|---|
| `x-provider` | OpenRouter | Which backend provider served the request |
| `openrouter-processing-ms` | OpenRouter | Backend processing time |
| `x-request-id` | OpenAI, OpenRouter | Request tracing |
| `served_model` | All | Actual model ID returned (may differ from requested) |
| `x-ratelimit-*` | OpenAI, OpenRouter | Rate limit status |

These headers are stored in `metadata.response_headers` in the raw results JSON, enabling post-hoc analysis of provider routing and version consistency.

---

## 7. Known limitations

1. **Small N.** N=10 runs per cell is enough to detect large performance gaps and compute basic confidence intervals, but not enough for fine-grained significance testing between models that score similarly.

2. **Single-author ground truth.** All GT was created by one person with no independent second-rater check. Systematic blind spots in the GT construction (e.g., a keyword set that is too narrow, or a dimension tolerance that was miscalculated) would bias all models equally.

3. **Keyword-based matching.** The compliance, docs, and schedule_read evaluators rely on keyword matching rather than semantic similarity. A model that correctly identifies a deviation but uses unexpected phrasing may be scored as a miss. The keyword sets were designed to be broad, but edge cases are possible.

4. **Synthetic data.** All inputs are synthetic. While designed to mirror real-world complexity (mixed units, missing fields, borderline values, scanned-document noise), they may not capture the full distribution of messiness found in production documents.

5. **Model provenance is trust-based.** The harness requests a model by ID, but no provider offers cryptographic attestation that the served weights match the requested checkpoint. OpenRouter dispatches to downstream providers (Together, Fireworks, DeepInfra, etc.); from v0.5, the harness captures `x-provider` and `served_model` headers for traceability, but a backend could still serve a subtly different checkpoint or quantization variant. Results should be read as "performance of the model ID as served by provider X on date Y" rather than "performance of a verified model artifact."

6. **Temperature sensitivity.** The default temperature of 0.1 was chosen for reproducibility. Higher temperatures may produce different score distributions. The benchmark does not sweep temperature.

7. **No multilingual coverage beyond German.** The `docs` workflow exercises German OCR fallback via a scanned PDF variant, but no other languages are tested. SMEs in non-English/German markets may face different extraction challenges.

8. **Vision workflows depend on provider support.** The `dims_vlm` and `docs_vlm` workflows require a vision-language model API that accepts base64-encoded images. Not all providers support this uniformly.

9. **False-positive cost is not weighted.** In real-world use, a false positive on compliance (unnecessary material rejection) has very different cost from a false positive on schedule_read (investigating a non-issue). The benchmark treats all FPs equally.

10. **Drawing-1 balloon ambiguity.** Engineering drawing 1 (`technical-drawing-1.jpg`) has known dimension callout positioning ambiguity (documented in `data/dims/DRAWING_FIXES.md`). This makes it consistently the hardest drawing for all models across both dims_ocr and dims_vlm workflows. The ground truth reflects the intended reading; the ambiguity biases against all models equally.

---

## 8. Reproducibility checklist

To reproduce published results:

1. Use the exact model IDs and versions listed in the results JSON.
2. Use `temperature=0.1` and `runs=10` (the defaults).
3. Use the data files committed in this repository (ground truth + inputs).
4. Run `uv sync` to install pinned dependencies from `uv.lock`.
5. Ensure EasyOCR is running on the same platform (CPU mode) -- OCR results may vary slightly across hardware.

The harness writes full raw outputs (including model responses) to `results/run_<timestamp>.json`, enabling post-hoc re-scoring if the evaluation logic is updated.

---

## 9. Scope and roadmap

This benchmark is in its second iteration (v0.5). Known areas for future extension include:

- **Additional workflows and data.** Expand beyond the current eight workflow variants with more diverse inputs (e.g. multilingual documents, larger spreadsheets, multi-page engineering drawings).
- **Fine-tuning comparison.** Evaluate whether LoRA or full fine-tuning on a small domain-specific corpus (e.g. compliance reasoning) meaningfully shifts the score distribution vs. off-the-shelf models.
- **Multi-rater ground truth.** Add independent second-rater validation of ground-truth labels to assess and mitigate single-author bias.
- **Semantic matching.** Replace or supplement keyword-based scoring with embedding-similarity matching to reduce false negatives from unexpected phrasing.

### Implemented in v0.5

- **Model provenance logging.** Response headers (`x-provider`, `served_model`, rate limits) are now captured and stored in metadata (see section 6.4).
- **Infrastructure failure separation.** Failed runs are categorized as infra vs capability, with separate score sets (see section 6.1).
- **Bootstrap confidence intervals.** 95% CIs on all metrics (see section 6.2).
- **Closed-model anchor points.** GPT-5.5 and Claude Opus 4.6 as reference ceilings.
- **Multi-backend support.** Tensorix (open models) + OpenRouter (VLMs + closed-model anchors).
- **Automatic retry with backoff.** Transient HTTP errors (429, 502, 503) are retried up to 2 times with exponential backoff; fatal errors (402, 404) fail immediately.
