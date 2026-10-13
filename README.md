# AI Support Triage Agent (Spec Project Demo)

A self-contained demo of an AI support-ticket triage pipeline:

**intake → classify (category + urgency) → confidence check → draft first response → escalate low-confidence tickets to a human.**

## Run it

```bash
python3 triage.py
```

Standard library only. No dependencies, no network, no API keys. Deterministic.

## What it does

- Reads `tickets.json`: 48 synthetic support tickets with ground-truth
  category (`billing` / `technical` / `account` / `shipping`) and priority labels.
- Classifies each ticket's category and urgency with weighted keyword scoring.
- If classification confidence is below threshold, escalates the ticket to a
  human instead of guessing.
- Drafts a first-response reply per ticket (category-specific template,
  priority-aware tone).
- Evaluates predictions against ground truth and writes `metrics.json`
  and `results.json`.

## Latest run (2026-10-13)

| Metric | Value |
|---|---|
| Tickets evaluated | 48 |
| Category accuracy | 89.6% |
| Accuracy on auto-routed tickets (non-escalated) | 93.5% |
| Urgency detection (priority vs routine) | 79.2% |
| Escalation rate | 4.2% (2 tickets sent to a human) |
| Throughput | ~0.035 ms / ticket |

Per-category precision / recall: billing 0.86/1.00, technical 0.83/0.83,
account 0.90/0.75, shipping 1.00/1.00.

## Files

- `triage.py` — the agent: classifier, reply drafter, evaluator
- `tickets.json` — 48 labeled sample tickets (synthetic data)
- `metrics.json` — evaluation metrics from the latest run (generated)
- `results.json` — per-ticket predictions and draft replies (generated)

## Notes

This is a **spec project**: a fictional scenario built to demonstrate
capability. The keyword-based classifier is a v1 baseline; a production
deployment would swap in LLM-based classification (OpenAI/Claude API)
behind the same pipeline interface — same intake, confidence check,
drafting, and escalation contract.
