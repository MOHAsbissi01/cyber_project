"""Dataset explorer: structure, quality problems found and fixed, class distribution, browsable records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

ui.setup("Dataset explorer", "🗂️")
svc = ui.service()
if svc is None:
    st.stop()
dq = svc.report("data_quality.json") or {}

st.markdown("**Source:** Hugging Face `AlicanKiraz0/Cybersecurity-Dataset-v1` (apache-2.0) · columns `system`, `user`, `assistant` · no labels.")
if dq:
    ui.caveat(dq["nature"])
    c = st.columns(4)
    ui.kpi(c[0], "Raw rows", f"{dq['raw_rows']:,}")
    ui.kpi(c[1], "After cleaning", f"{dq['rows_after_cleaning']:,}", f"{dq['raw_rows'] - dq['rows_after_cleaning']} dropped")
    ui.kpi(c[2], "Avg answer length", f"{dq['answer_chars_mean']:.0f}", "characters")
    ui.kpi(c[3], "Records with a CVE", dq["rows_with_cve_in_raw_text"], f"{dq['rows_with_ip_in_raw_text']} with an IP")
    with st.expander("Data-quality findings and what was done about them", expanded=True):
        st.markdown(f"""
- **Leaked chain-of-thought:** {dq['answers_with_think_tag']} answers contain `<think>`, and {dq['answers_with_unclosed_think']} of them never close it (generation cut inside the reasoning).
  Such rows have no final answer and are dropped; in the others {dq['reasoning_sentences_removed_total']} first-person reasoning sentences were removed from {dq['rows_with_reasoning_sentences_removed']} rows.
- **Literal `\\n` sequences** in {dq['rows_with_literal_backslash_n']} answers were converted to spaces.
- **Duplicates:** {dq['duplicate_questions']} duplicate questions, removed.
- **Not truncated:** the full answer is kept (the first prototype cut it at 1,500 characters and lost half of the CVE mentions).
- **Factual quality:** answers are LLM-generated and some CVE/technique explanations are wrong; relations are therefore stored with provenance and checked against a small reference KB.
""")
st.divider()

left, right = st.columns([3, 2])
with left:
    st.subheader("Browse records")
    f1, f2, f3 = st.columns([2, 2, 3])
    cats = ["(all)"] + sorted(svc.df.category.unique())
    cat = f1.selectbox("Category (weak label)", cats)
    sp = f2.selectbox("Split", ["(all)", "train", "val", "test"])
    q = f3.text_input("Text contains", placeholder="e.g. kerberos")
    page = svc.dataset_page(None if cat == "(all)" else cat, None if sp == "(all)" else sp, q or None, limit=300)
    st.caption(f"{len(page)} record(s) shown (max 300)")
    event = st.dataframe(page, hide_index=True, width="stretch", height=330, on_select="rerun", selection_mode="single-row")
with right:
    st.subheader("Class distribution")
    cc = pd.crosstab(svc.df.category, svc.df.split)[["train", "val", "test"]]
    st.bar_chart(cc, horizontal=True, height=330)
    st.caption("Stratified split: every class keeps the same proportions in train / val / test.")

rows = event.selection.rows if event and event.selection else []
if rows:
    st.subheader("Selected record")
    ui.show_incident_expander(svc, int(page.iloc[rows[0]]["id"]))
else:
    st.info("Select a row in the table to read the full question, answer and extracted entities.")
