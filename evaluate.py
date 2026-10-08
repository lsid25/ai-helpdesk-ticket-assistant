"""
Measures how accurately the assistant triages the labeled sample tickets.
Run:  python evaluate.py          (offline mode)
      python evaluate.py --ai     (Claude mode, needs ANTHROPIC_API_KEY)
"""

import sys
import time

import pandas as pd

from triage import KnowledgeBase, load_kb, triage_ticket


def main():
    use_ai = "--ai" in sys.argv
    kb = KnowledgeBase(load_kb())
    tickets = pd.read_csv("data/sample_tickets.csv")

    start = time.time()
    results = [triage_ticket(t, kb, use_ai=use_ai) for t in tickets.to_dict("records")]
    elapsed = time.time() - start

    tickets["pred_category"] = [r["category"] for r in results]
    tickets["pred_priority"] = [r["priority"] for r in results]
    cat_acc = (tickets.pred_category == tickets.true_category).mean()
    pri_acc = (tickets.pred_priority == tickets.true_priority).mean()

    print(f"Mode: {results[0]['mode']}")
    print(f"Tickets triaged: {len(tickets)} in {elapsed:.2f}s")
    print(f"Category accuracy: {cat_acc:.0%}")
    print(f"Priority accuracy: {pri_acc:.0%}")

    misses = tickets[tickets.pred_category != tickets.true_category]
    if len(misses):
        print("\nCategory misses:")
        print(misses[["ticket_id", "subject", "true_category", "pred_category"]].to_string(index=False))


if __name__ == "__main__":
    main()
