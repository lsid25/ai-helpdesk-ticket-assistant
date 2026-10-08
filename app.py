"""
AI Help Desk Ticket Assistant - Streamlit dashboard
Run:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from triage import CATEGORIES, PRIORITIES, KnowledgeBase, load_kb, triage_ticket

st.set_page_config(page_title="AI Help Desk Assistant", page_icon="🛠️", layout="wide")

PRIORITY_COLORS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}


@st.cache_resource
def get_kb():
    return KnowledgeBase(load_kb())


@st.cache_data(show_spinner="Triaging tickets...")
def run_triage(df, use_ai):
    kb = get_kb()
    rows = []
    for t in df.to_dict("records"):
        r = triage_ticket(t, kb, use_ai=use_ai)
        rows.append({**t, **r})
    return pd.DataFrame(rows)


# ---------- Sidebar ----------
st.sidebar.title("🛠️ Help Desk AI")
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
use_ai = st.sidebar.toggle("Use Claude AI", value=has_key, disabled=not has_key,
                           help="Set ANTHROPIC_API_KEY to enable AI mode.")
st.sidebar.caption("Mode: **AI (Claude)**" if use_ai else "Mode: **Offline** (KB similarity + rules)")

uploaded = st.sidebar.file_uploader("Upload tickets CSV", type="csv",
                                    help="Columns: ticket_id, user, subject, description")
tickets = pd.read_csv(uploaded) if uploaded else pd.read_csv("data/sample_tickets.csv")

# ---------- Main ----------
st.title("AI Help Desk Ticket Assistant")
st.caption("Automatically triages IT tickets, matches knowledge-base fixes, and drafts replies for technicians.")

results = run_triage(tickets, use_ai)
order = {p: i for i, p in enumerate(PRIORITIES)}
results = results.sort_values("priority", key=lambda s: s.map(order))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickets in queue", len(results))
c2.metric("Critical", int((results.priority == "Critical").sum()))
c3.metric("High", int((results.priority == "High").sum()))
if "true_category" in results:
    acc = (results.category == results.true_category).mean()
    c4.metric("Category accuracy (labeled set)", f"{acc:.0%}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Triaged queue (sorted by priority)")
    view = results[["ticket_id", "user", "subject", "category", "priority", "kb_id"]].copy()
    view["priority"] = view["priority"].map(lambda p: f"{PRIORITY_COLORS.get(p, '')} {p}")
    st.dataframe(view, hide_index=True, width="stretch")
with right:
    st.subheader("Tickets by category")
    counts = results.category.value_counts().reindex(CATEGORIES, fill_value=0)
    st.bar_chart(counts, horizontal=True)

st.divider()
st.subheader("Ticket details")
pick = st.selectbox("Select a ticket", results.ticket_id,
                    format_func=lambda i: f"{i} - {results.set_index('ticket_id').loc[i, 'subject']}")
t = results.set_index("ticket_id").loc[pick]

a, b = st.columns(2)
with a:
    st.markdown(f"**From:** {t.user}")
    st.markdown(f"**Subject:** {t.subject}")
    st.info(t.description)
    st.markdown(f"**Category:** {t.category}  \n**Priority:** {PRIORITY_COLORS.get(t.priority, '')} {t.priority}")
    st.markdown(f"**Suggested fix:** {t.kb_id} - {t.kb_title} (match score {t.confidence})")
    st.caption(t.reason)
with b:
    st.markdown("**Drafted reply** (review before sending)")
    st.text_area("reply", t.reply, height=260, label_visibility="collapsed")

st.download_button("Download triaged queue (CSV)", results.to_csv(index=False),
                   "triaged_tickets.csv", "text/csv")
