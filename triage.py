#!/usr/bin/env python3
"""
AI Support Triage Agent (Spec Project demo).

Pipeline: intake -> classify (category + priority) -> confidence check ->
draft first response -> escalate low-confidence tickets to a human.

Reads tickets.json (tickets with ground-truth labels), classifies each one,
drafts a reply, evaluates against ground truth, and writes metrics.json.

Standard library only. Deterministic: no randomness, no network, no API keys.
"""

import json
import re
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent

# --- Keyword signals: category -> [(keyword, weight)] -------------------------
CATEGORY_SIGNALS = {
    "billing": [
        ("charg", 3), ("invoice", 4), ("refund", 4), ("payment", 3),
        ("subscription", 4), ("billing", 5), ("receipt", 3), ("overcharg",
         4), ("downgrade", 3), ("upgrade", 2), ("vat", 4), ("currency", 3),
        ("discount", 3), ("trial", 2), ("renewal", 3), ("price", 2),
        ("fee", 2),
    ],
    "technical": [
        ("crash", 4), ("error", 3), ("500", 4), ("401", 4), ("bug", 4),
        ("login", 3), ("slow", 3), ("load", 2), ("sync", 3), ("api", 4),
        ("notification", 2), ("upload", 3), ("freeze", 4), ("blank page", 5),
        ("down", 3), ("search", 2), ("dark mode", 3), ("two-factor", 3),
        ("authenticator", 3),
    ],
    "account": [
        ("password", 4), ("email address", 3), ("delete my account", 5),
        ("merge", 4), ("suspend", 4), ("reactivat", 4), ("username", 4),
        ("profile", 3), ("team member", 4), ("verify my identity", 4),
        ("gdpr", 4), ("privacy", 3), ("hacked", 5), ("workspace", 3),
        ("display name", 3),
    ],
    "shipping": [
        ("order", 3), ("tracking", 5), ("package", 4), ("deliver", 4),
        ("shipment", 4), ("parcel", 4), ("return label", 5), ("exchange", 3),
        ("customs", 4), ("gift wrapping", 4), ("pickup point", 4),
        ("address", 2), ("damaged", 3), ("wrong item", 4),
    ],
}

# --- Priority signals ---------------------------------------------------------
PRIORITY_SIGNALS = {
    "urgent": [("urgent", 3), ("asap", 3), ("right now", 3), ("losing sales", 3),
               ("hacked", 3), ("can't get in", 2), ("locked out", 2)],
    "high": [("as soon as possible", 2), ("immediately", 2), ("missing", 2),
             ("late", 2), ("disappointed", 1), ("crash", 1),
             ("is down", 2), ("went down", 2), ("page down", 2),
             ("site down", 2), ("system down", 2),
             ("charged twice", 2), ("duplicate", 1)],
    "low": [("just wondering", 2), ("not urgent", 2), ("minor", 2),
            ("just flagging", 2)],
}

PRIORITY_ORDER = ["low", "medium", "high", "urgent"]

REPLY_TEMPLATES = {
    "billing": ("Thanks for reaching out about billing. I've pulled up your "
                "account and will review the charge/invoice in question. "
                "I'll follow up within one business day with the correction "
                "or refund confirmation."),
    "technical": ("Sorry you're hitting this issue. I've logged it for our "
                  "engineering team with the details you provided. As a first "
                  "step, could you confirm your app/browser version? I'll "
                  "update you as soon as there's a fix or workaround."),
    "account": ("I can help with your account request. For security, I'll "
                "need to verify your identity first — please reply from your "
                "registered email or complete the verification step in your "
                "account settings, and I'll proceed right away."),
    "shipping": ("Thanks for the details. I'm checking your shipment status "
                 "now and will get back to you with tracking updates and the "
                 "next steps (replacement, return label, or redelivery)."),
}

ESCALATION_NOTE = ("This ticket was escalated to a human agent because the "
                   "automatic classification confidence was too low.")


def score_text(text, signals):
    text = text.lower()
    total = 0
    for keyword, weight in signals:
        if keyword in text:
            total += weight
    return total


def classify(ticket):
    text = f"{ticket['subject']} {ticket['body']}"
    scores = {cat: score_text(text, sig)
              for cat, sig in CATEGORY_SIGNALS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1]
    confidence = ((top_score - second_score) / top_score
                  if top_score > 0 else 0.0)

    p_scores = {p: score_text(text, sig)
                for p, sig in PRIORITY_SIGNALS.items()}
    priority = "medium"
    best = max(p_scores.items(), key=lambda kv: kv[1])
    if best[1] > 0:
        priority = best[0]

    escalate = top_score == 0 or confidence < 0.25
    return {
        "category": top_cat,
        "priority": priority,
        "confidence": round(confidence, 3),
        "escalated": escalate,
        "scores": scores,
    }


def draft_reply(ticket, result):
    greeting = "Hi there,"
    if result["escalated"]:
        return f"{greeting} {ESCALATION_NOTE}"
    body = REPLY_TEMPLATES[result["category"]]
    if result["priority"] in ("high", "urgent"):
        body += " I've marked this as high priority on our side."
    return f"{greeting} {body}"


def evaluate(tickets, results):
    n = len(tickets)
    correct_cat = sum(1 for t, r in zip(tickets, results)
                      if r["predicted_category"] == t["category"])
    correct_prio = sum(1 for t, r in zip(tickets, results)
                       if r["predicted_priority"] == t["priority"])
    escalated = sum(1 for r in results if r["escalated"])
    auto = [r for r in results if not r["escalated"]]
    auto_correct = sum(
        1 for t, r in zip(tickets, results)
        if not r["escalated"] and r["predicted_category"] == t["category"])

    per_cat = {}
    for cat in CATEGORY_SIGNALS:
        tp = sum(1 for t, r in zip(tickets, results)
                 if r["predicted_category"] == cat and t["category"] == cat)
        fp = sum(1 for t, r in zip(tickets, results)
                 if r["predicted_category"] == cat and t["category"] != cat)
        fn = sum(1 for t, r in zip(tickets, results)
                 if r["predicted_category"] != cat and t["category"] == cat)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_cat[cat] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "support": tp + fn,
        }

    # Urgency is the binary decision the agent actually acts on
    # (high/urgent get priority handling in draft_reply); the four
    # priority levels are informational detail.
    def is_urgent(p):
        return p in ("high", "urgent")

    urgency_correct = sum(
        1 for t, r in zip(tickets, results)
        if is_urgent(r["predicted_priority"]) == is_urgent(t["priority"]))

    return {
        "tickets_evaluated": n,
        "category_accuracy": round(correct_cat / n, 3),
        "priority_accuracy": round(correct_prio / n, 3),
        "urgency_accuracy": round(urgency_correct / n, 3),
        "escalation_rate": round(escalated / n, 3),
        "auto_classified": len(auto),
        "auto_classified_accuracy": (round(auto_correct / len(auto), 3)
                                    if auto else 0.0),
        "per_category": per_cat,
    }


def main():
    tickets = json.loads((BASE / "tickets.json").read_text())

    results = []
    timings = []
    for ticket in tickets:
        start = time.perf_counter()
        result = classify(ticket)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)
        results.append({
            "id": ticket["id"],
            "predicted_category": result["category"],
            "true_category": ticket["category"],
            "predicted_priority": result["priority"],
            "true_priority": ticket["priority"],
            "confidence": result["confidence"],
            "escalated": result["escalated"],
            "draft_reply": draft_reply(ticket, result),
        })

    metrics = evaluate(tickets, results)
    metrics["avg_ms_per_ticket"] = round(sum(timings) / len(timings), 3)
    metrics["total_ms"] = round(sum(timings), 1)

    (BASE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (BASE / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Evaluated {metrics['tickets_evaluated']} tickets "
          f"in {metrics['total_ms']} ms "
          f"({metrics['avg_ms_per_ticket']} ms/ticket)")
    print(f"Category accuracy : {metrics['category_accuracy']:.1%}")
    print(f"Priority accuracy : {metrics['priority_accuracy']:.1%}")
    print(f"Escalation rate   : {metrics['escalation_rate']:.1%} "
          f"({metrics['tickets_evaluated'] - metrics['auto_classified']} "
          f"tickets sent to a human)")
    print(f"Auto-classified accuracy (non-escalated): "
          f"{metrics['auto_classified_accuracy']:.1%}")
    print("Per-category precision/recall:")
    for cat, pr in metrics["per_category"].items():
        print(f"  {cat:9s} P={pr['precision']:.2f} "
              f"R={pr['recall']:.2f} (n={pr['support']})")
    print("\nSample draft replies:")
    for r in results[:3]:
        print(f"  [{r['id']}] {r['draft_reply'][:100]}...")


if __name__ == "__main__":
    main()
