# INTERACTIVE_IMAGE_REVIEW — Proposed Hybrid Extraction Approach (deferred)

> **Status: proposed, not implemented.** This document records a design idea raised on
> 2026-07-26 and deliberately deferred. Priority right now is building a first VLM prototype
> from the Song + Labrosse papers' metadata (see [`METADATA_EXTRACTION.md`](METADATA_EXTRACTION.md)
> and `feedback_panels_need_image_association` in project memory). **Pick this up only after**
> that prototype validates the overall approach — don't start building the review tool before then.

---

## The problem

`extract`'s solar-image classification (`utils/solar_classifier.py`) is a classical-CV heuristic
(Hough circle detection, HSV palette analysis, HMI grayscale/texture heuristics, a raw score
threshold) tuned against a handful of example papers. In practice this doesn't generalize:

- **False negatives**: legitimate sub-panels get silently dropped. Confirmed concretely on Song
  et al.'s Fig. 5 — 6 of 9 real panels (the HMI continuum/magnetogram/difference row) never became
  saved images because they're small (180×120) crops that hit a "too small" rejection before any
  content-based scoring ran.
- **False positives**: figures that aren't real solar observations (diagrams, schematics, plots
  that happen to trigger the circle/color/texture heuristics) get accepted and saved as if they
  were genuine observation images — observed across several other papers being processed.

The underlying issue: SDO papers come from many different journals with wildly different figure
styles, color palettes, and layouts. A fixed set of pixel-heuristic thresholds will always have
failure modes on *some* subset of that diversity — every threshold fix for one paper's failure mode
risks breaking a different paper that relied on the old behavior (already observed once this
project, with the size-rejection threshold).

Given the ultimate goal is training data for a VLM (see `project_dataset_purpose` in project
memory), **label quality matters more here than in most pipeline contexts** — a bad image/caption
training pair actively hurts the model, it doesn't just waste processing time. That raises the bar
for how much automatic misclassification is acceptable.

## The proposed approach: classifier as highlighter, human as decider

Don't replace the classifier — change its role from *sole decision-maker* to *candidate
highlighter*:

1. Render each PDF page (PyMuPDF already gives us page images and every embedded image's bbox —
   `utils/pdf_extractor.py` and `extraction_log.json` already capture this for every embedded
   image, not just the ones currently saved).
2. Run the existing classifier to produce candidate scores/bboxes, and visually highlight them on
   the rendered page in a small review UI.
3. The user confirms, rejects, or manually adds/adjusts a selection per image, per paper — nothing
   is saved to `output/images/<name>/` without that confirmation.

This keeps the classifier's value (it still does the heavy lifting of proposing candidates, so the
user isn't manually scanning every embedded object on every page — logos, watermarks, tiny icons,
etc. don't need review) while eliminating both failure modes above, since nothing reaches the
canonical output layout without a human looking at it.

### Synergy with panel-image association

This is also the natural place to resolve the multi-panel association problem recorded in
`feedback_panels_need_image_association` (project memory): if the user is already looking at the
rendered page to confirm/reject a candidate image, they can *at the same time* indicate which panel
letter(s) it corresponds to — directly, by inspection — instead of relying on the geometric
`bbox`-ordering heuristic that was being validated across more papers before this proposal came up.
One review pass could plausibly resolve both the classification-quality problem and the
panel-association problem, rather than building two separate heuristic subsystems.

## The trade-off to size before committing

This shifts pipeline cost from **compute** to **the user's review time per paper**. For a large VLM
training corpus, page-by-page manual review could become the actual bottleneck — more so than any
compute or LLM inference cost in the current pipeline. Before building this, get a real sense of the
target corpus size (reviewing 50 papers is a very different commitment than reviewing 2,000) and
decide whether the hybrid tool's review burden is sustainable at that scale, or whether some further
automation (e.g. only surfacing low-confidence classifier decisions for review, auto-accepting
high-confidence ones) is needed to keep it tractable.

## Rough shape (not designed in detail yet)

- Likely a small local web app: PyMuPDF renders each page to an image, candidate bboxes drawn as
  clickable/toggleable overlays (classifier output, color-coded by confidence), keyboard or click
  to confirm/reject/relabel-panel, writes directly into the canonical `extraction_log.json` /
  `images/<name>/` layout so it slots into the existing pipeline downstream (`metadata`, `query`)
  without changing their contract.
- `solar_classifier.py`'s scoring logic is reused as-is for the initial highlight suggestions — no
  need to redesign the CV heuristics themselves, just stop trusting them unconditionally.
