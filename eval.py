"""Score the classifier against labeled tickets. This is your Week 1 'analysis' step.

Run:  python eval.py
Edit sample_tickets.json (add 20-30 tickets, including tricky ones) and watch the
numbers change as you improve CLASSIFY_SYSTEM in ticket_router.py.
"""
import json
from pathlib import Path

from ticket_router import ClassificationError, classify_ticket


def main() -> None:
    tickets = json.loads(Path(__file__).with_name("sample_tickets.json").read_text())
    import anthropic
    client = anthropic.Anthropic()
    cat_ok = urg_ok = errors = 0
    misses = []

    for t in tickets:
        try:
            c = classify_ticket(client, t["text"])
        except ClassificationError:
            errors += 1
            continue
        cat_hit = c["category"] == t["expected_category"]
        urg_hit = c["urgency"] == t["expected_urgency"]
        cat_ok += cat_hit
        urg_ok += urg_hit
        if not (cat_hit and urg_hit):
            misses.append((t["id"], t["expected_category"], c["category"],
                           t["expected_urgency"], c["urgency"]))

    n = len(tickets)
    print(f"Tickets: {n} | Hard failures: {errors}")
    print(f"Category accuracy: {cat_ok}/{n} ({cat_ok / n:.0%})")
    print(f"Urgency accuracy:  {urg_ok}/{n} ({urg_ok / n:.0%})")
    for m in misses:
        print(f"  MISS {m[0]}: category {m[1]} -> {m[2]} | urgency {m[3]} -> {m[4]}")


if __name__ == "__main__":
    main()
