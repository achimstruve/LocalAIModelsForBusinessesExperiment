# First-Run Insights (May 21, 2026 — N=10)

This document captures lessons from the first full AGEN-OPS-6 benchmark run before any harness changes were made. It serves both as a historical record and as part of the paper narrative ("how we iterated").

---

## 1. Provider failure breakdown

Of 577 failed runs across the entire run, the overwhelming majority were **infrastructure failures** rather than model capability issues:

| Failure mode | Count | Provider | Notes |
|---|---|---|---|
| 402 (Payment Required) | 382 | OpenRouter | Credit exhaustion mid-run |
| 404 (Not Found) | 90 | OpenRouter | Routing filter too strict — `quantizations: ["fp16", "bf16", "fp8"]` excluded most VLM backends that only serve int4/int8 |
| Timeout | 39 | Both | Individual calls exceeding 150s budget |
| Circuit-breaker skip | 48 | Both | Runs skipped after 2 consecutive timeouts on same combo |
| JSON parse failure | 17 | Both | Model returned unparseable output (Tensorix: formatting issues; OpenRouter: partial responses) |
| Rate limit (429) | 1 | OpenRouter | Single occurrence |

**Key takeaway:** 472 of 577 failures (82%) were OpenRouter infrastructure issues (402 + 404), not model limitations. The docs_vlm workflow scored 0/10 across *all* VLM models — not because VLMs can't do document extraction, but because the account ran out of credits before docs_vlm ran.

---

## 2. OCR+LLM hybrid vs direct VLM

The first run tested two extraction paths for engineering drawings:

- **dims_ocr** (EasyOCR locally + LLM consolidation): Ceiling F1 ~0.70
- **dims_vlm** (raw image to VLM): Best F1 0.926 when the provider worked

The OCR bottleneck is **recall**, not precision. EasyOCR systematically loses dimension information — small callouts, overlapping text, tolerance frames — that VLMs can read directly from the image. The LLM consolidation step can't recover what the OCR stage drops.

This finding drove the decision to focus on the VLM path for v0.5: if the provider infrastructure is reliable, direct VLM is the better approach for engineering drawing extraction.

---

## 3. Model-size surprises

Size does not predict performance on structured operational tasks:

- **qwen3.5-9b** (9B parameters) achieved perfect F1 1.000 on schedule_write, beating all larger models including 120B+.
- **gpt-oss-20b** nearly matched gpt-oss-120b on dims (F1 0.933 vs 1.000).
- The 235B+ MoE models showed no consistent advantage over 70B-class models on text workflows.

**Implication for SMEs:** A 9B model on a $1,500 GPU can outperform a 120B model on a $16,000 dual-GPU rig for certain structured tasks. Hardware tier recommendations must be workflow-specific.

---

## 4. Workflow difficulty ranking

From hardest to easiest (based on best-model F1 across the ladder):

| Rank | Workflow | Best F1 | Best model | Challenge |
|---|---|---|---|---|
| 1 (hardest) | schedule_read | 0.675 | multiple | Multi-row dependency reasoning across 20+ rows |
| 2 | docs | ~0.85 | gpt-oss-120b | Varies sharply by PDF type (native vs scanned) |
| 3 | dims | 1.000 | gpt-oss-120b, glm-5.1 | 10% tolerance-band arithmetic; some models fail the rule |
| 4 | compliance | ~0.95 | multiple | Most models catch 4-5/5 deviations; DEV-5 (standard revision) is hardest |
| 5 (easiest) | schedule_write | 1.000 | qwen3.5-9b, others | Well-structured scenario → well-structured JSON edit list |

**schedule_read** stands out as the only workflow where no model exceeds F1 0.675. The task requires reasoning about temporal dependencies across multiple rows — a capability that current models handle inconsistently.

---

## 5. Ground truth review (second-reviewer audit)

All arithmetic in the ground truth was verified correct. Specific findings per workflow:

### Compliance
- All 5 deviations correctly identified. All 19 compliant fields verified.
- The Manganese trap (cert 1.98% vs spec max 2.00%) works as intended — models that flag it incur a false positive.
- **No changes needed.**

### Dims
- All 25 dimensions verified. 9 must-flag, 15 compliant, 1 informational (surface roughness as N/A from 3D model).
- Arithmetic on the 10% tolerance-band rule checked out for all 25 rows.
- **No changes needed.**

### Dims OCR/VLM drawings
- Internal structure of all 6 ground truth JSONs consistent.
- Drawing 1 has known balloon ambiguity (documented in `data/dims/DRAWING_FIXES.md`): dimension callout positioning makes it consistently the hardest drawing for all models.
- **No GT changes needed.** Drawing-1 ambiguity documented in METHODOLOGY.md.

### Docs
- 55 entities well-structured across 3 PDF ground truth files.
- Minor keyword generosity noted (e.g., E-12 "lambda" could match hyperparameter references; E-24 "paris" could match unrelated mentions), but this biases *toward* models, not against.
- **No changes needed.**

### Schedule_read
- 4 issues with clear match logic. Keyword sets are broad enough.
- **No changes needed.**

### Schedule_write
- 6 edits correct per scenario arithmetic.
- **Edge case found:** GT expects CMM inspection on 2026-05-23 (a Saturday). Models reasoning about business days would set 2026-05-25 (Monday) and score as wrong.
- **Resolution:** Accept both 2026-05-23 and 2026-05-25 as valid for ED-4 and ED-5. Manufacturing schedules can legitimately run on weekends, but a model accounting for business days is not wrong either.

---

## 6. Actionable changes for v0.5

Based on these findings, the following changes were prioritised:

1. **Harden OpenRouter reliability** — relax quantization constraints, add pre-flight credit check, better error categorization.
2. **Add closed-model anchor points** (GPT-5.5, Claude Opus 4.7) — reference ceilings for interpreting open-model scores.
3. **Separate infrastructure from capability failures** in scoring — report both "all runs" and "successful only" metrics.
4. **Focus on VLM path** — move dims_ocr to appendix; dims_vlm becomes the primary vision workflow.
5. **Add bootstrap confidence intervals** — N=10 is small; CIs make the uncertainty visible.
6. **Capture provider response headers** — strengthen model provenance traceability.
7. **Fix schedule_write weekend edge case** — accept Monday alternative for CMM dates.
