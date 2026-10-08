"""
Core logic for the AI Help Desk Ticket Assistant.

For each ticket it:
  1. Finds the most relevant knowledge-base (KB) article (TF-IDF text similarity)
  2. Classifies the ticket's category and priority
  3. Drafts a reply for the technician to review

Two modes:
  - AI mode:      uses Claude (Anthropic API) when ANTHROPIC_API_KEY is set
  - Offline mode: uses KB similarity + keyword rules, so the demo works without a key
"""

import json
import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
CATEGORIES = [
    "Account & Access",
    "Network & Connectivity",
    "Hardware & Printing",
    "Software & Applications",
    "Email & Collaboration",
    "Security",
]
PRIORITIES = ["Critical", "High", "Medium", "Low"]
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")


def load_kb():
    with open(DATA_DIR / "knowledge_base.json", encoding="utf-8") as f:
        return json.load(f)


class KnowledgeBase:
    """Searches KB articles by text similarity."""

    def __init__(self, articles):
        self.articles = articles
        docs = [f"{a['title']} {a['keywords']} {' '.join(a['steps'])}" for a in articles]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, text, top_k=3):
        scores = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [(self.articles[i], float(scores[i])) for i in ranked]


# ---------- Offline (rule-based) mode ----------

URGENCY_RULES = [
    ("Critical", r"\b(nobody|everyone|whole office|all computers|entire|outage|clicked .*link|entered my password|hacked|ransomware|crash\w*)\b"),
    ("High", r"\b(can't log in|cannot log in|can't sign in|locked|patients? (are )?waiting|urgent|asap|before my shift|error)\b"),
    ("Low", r"\b(install|request|new hire|when you can|no rush|keyboard|update available)\b"),
]


def offline_priority(text):
    t = text.lower()
    for priority, pattern in URGENCY_RULES:
        if re.search(pattern, t):
            return priority
    return "Medium"


def offline_reply(ticket, article):
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(article["steps"], 1))
    return (
        f"Hi {ticket['user']},\n\n"
        f"Thanks for reaching out about \"{ticket['subject']}\". "
        f"We're looking into it now. Here's what we'll do:\n{steps}\n\n"
        f"We'll update you as soon as it's resolved.\n\nIT Support"
    )


# ---------- AI mode (Claude) ----------

def ai_triage(ticket, kb_matches):
    import anthropic

    client = anthropic.Anthropic()
    kb_text = "\n".join(
        f"- {a['id']} | {a['title']} | steps: {' / '.join(a['steps'])}" for a, _ in kb_matches
    )
    prompt = f"""You are an IT help desk triage assistant.

Ticket from {ticket['user']}:
Subject: {ticket['subject']}
Description: {ticket['description']}

Relevant knowledge-base articles:
{kb_text}

Return ONLY a JSON object with these keys:
"category": one of {CATEGORIES}
"priority": one of {PRIORITIES} (Critical = many users down, security incident, or patient care blocked)
"kb_id": the id of the most relevant article above
"reason": one short sentence explaining the priority
"reply": a short, friendly reply to the user in plain language, signed "IT Support"
"""
    msg = client.messages.create(
        model=MODEL, max_tokens=600, messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


# ---------- Main entry point ----------

def triage_ticket(ticket, kb, use_ai=None):
    """Returns a dict with category, priority, KB article, reply, and mode used."""
    text = f"{ticket['subject']} {ticket['description']}"
    matches = kb.search(text)
    best_article, score = matches[0]

    if use_ai is None:
        use_ai = bool(os.getenv("ANTHROPIC_API_KEY"))

    if use_ai:
        try:
            r = ai_triage(ticket, matches)
            article = next((a for a, _ in matches if a["id"] == r.get("kb_id")), best_article)
            return {
                "category": r["category"],
                "priority": r["priority"],
                "kb_id": article["id"],
                "kb_title": article["title"],
                "confidence": round(score, 2),
                "reason": r.get("reason", ""),
                "reply": r["reply"],
                "mode": "AI (Claude)",
            }
        except Exception as e:  # fall back so the demo never breaks
            print(f"AI mode failed, using offline mode: {e}")

    return {
        "category": best_article["category"],
        "priority": offline_priority(text),
        "kb_id": best_article["id"],
        "kb_title": best_article["title"],
        "confidence": round(score, 2),
        "reason": "Matched by knowledge-base similarity and urgency keywords.",
        "reply": offline_reply(ticket, best_article),
        "mode": "Offline",
    }
