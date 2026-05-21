# Drawing fixes needed before ground-truth finalization

> Compiled 2026-05-20. These are inconsistencies found while building the ground-truth JSON files.
> Fix them in the CAD source, re-export the JPGs, and the ground-truth files will be updated to match.

## technical-drawing-1.jpg — Gearbox Adapter Housing (GAH-250-01)

The balloon callouts reuse the same number for genuinely different features. If a balloon number is meant to identify one unique feature, the following are errors:

1. **Balloon ④ is overloaded across four places.** It points to a body region on the FRONT VIEW, to `R8 TYP.` on the RIGHT-SIDE VIEW, to `R15 TYP.` on the TOP VIEW, and to `R6 TYP.` on SECTION B-B. Those are three different radii (8, 15, 6) plus an unclear front-view target. Each radius needs its own balloon number.

2. **Balloon ⑩ points to two different bores.** On SECTION A-A it labels `Ø120 H7`; on SECTION B-B it labels `Ø90 H7`. One of them needs a different balloon.

3. **Balloon ⑪ points to two different bores.** On SECTION A-A it labels `Ø170`; on SECTION B-B it labels `Ø120 H7`. One of them needs a different balloon.

4. **Front-view balloons ①, ③, ④ have unclear targets.** They point at body/casting regions rather than at a specific dimension or feature. Either retarget them to the dimension they belong to, or remove them.

Note — these reuses are *correct* and should be left alone: balloon ② (the `6 × Ø11 THRU on Ø160 B.C.D.`, shown in front view and section B-B), balloon ⑦ (the `M10×1-6H` lube port, shown in front view, top view and detail C), and balloon ⑭ (the `Ø50 H7` main bore, shown in sections A-A and B-B). Those are the *same* feature shown in multiple views, which is fine.

Suggested clean scheme for the bores: `Ø50 H7`→⑭, `Ø70`→⑬, `Ø90 H7`→ new number, `Ø120 H7`→⑩ (use in both sections), `Ø170`→⑪ — then make sections A-A and B-B reference them consistently.

## technical-drawing-3.jpg — xometry_lathe_sample v3.0

5. **Balloon ⑬ appears twice.** There are two separate balloons both numbered 13 near the keyway / right-hand shaft end. One of them should be renumbered.

6. **Balloons ⑮ and ㉕ are missing.** The sequence runs 1–31 but no balloon 15 and no balloon 25 exist on the sheet — likely the renumber that should have produced them. After fixing item 5, re-sequence so the numbering is contiguous with no gaps and no duplicates.

7. **Detail C (4:1) dimensions are un-ballooned.** The `0.5` width and the `30°` angle in detail C have no balloon, while `R0.3` in the same detail has balloon ㉛. Add balloons to `0.5` and `30°` for consistency.

## technical-drawing-4.jpg — Mounting Flange (MF-002)

8. **The large-bore counterbore callout is geometrically backwards.** It currently reads:

   ```
   2 × Ø32.0 THRU ALL
   ⌴ Ø20.0 ▽ 22.0
   ```

   A counterbore (`⌴`) must be **larger** than the hole it sits on — a Ø20 counterbore cannot exist on a Ø32 through-hole. The two diameters are almost certainly swapped. It should read:

   ```
   2 × Ø20.0 THRU ALL
   ⌴ Ø32.0 ▽ 22.0
   ```

   (Ø20 through-hole, Ø32 counterbore, 22 deep.) Please confirm which is the intended through-hole diameter, then correct the callout.

## technical-drawing-2, -5, -6 — no fixes required

- **technical-drawing-2.jpg** (Housing Bracket HB-0247) — no inconsistencies found; no balloons used, callouts are internally consistent.
- **technical-drawing-5.jpg** (Clevis Support Bracket CSB-100-01) — no hard errors found. One thing worth a glance: balloon numbers 2 and 3 were not clearly visible while reading the sheet — confirm the balloon sequence is complete (no gaps).
- **technical-drawing-6.jpg** (Clevis Mount Bracket CMB-250605-01) — no inconsistencies found; no balloons used.

## After you re-export

Re-save the corrected JPGs into this folder (`data/bm1/`). Then the ground-truth JSON files will be updated so that:
- drawing-1: bore diameters keyed to their corrected balloons
- drawing-3: dimension list re-aligned to the contiguous balloon sequence
- drawing-4: the counterbore entry corrected to the right through/counterbore diameters

Drawings 2, 5, 6 ground truth will not change unless their images change.
