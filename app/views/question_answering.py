"""Question answering: Graph-RAG vs hybrid RAG vs vector RAG, with the evidence behind every answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

EXAMPLES = ["Which vulnerabilities affect log4j?", "Which software does CVE-2016-5195 affect?", "Which incidents are related to Mimikatz?",
            "What does phishing target?", "How can ROP attacks be mitigated?", "What is the relationship between ASLR and ROP?",
            "Which incidents are similar to incident 3?", "What is Kerberoasting?", "Which CVE affects Zyxel firewalls?"]
MODES = {"Graph-RAG (recommended)": "graph", "Hybrid RAG (lexical + dense)": "hybrid", "Vector RAG (dense only)": "vector"}

ui.setup("Question answering", "💬")
svc = ui.service()
if svc is None:
    st.stop()
if svc.store_note:
    st.warning(svc.store_note)

with st.sidebar:
    st.markdown("**Answering options**")
    choice = st.radio("Method", list(MODES) + ["Compare all three"], index=0)
    k = st.slider("Sources per answer", 1, 10, 5)
    st.caption(f"Graph backend: **{svc.store.name}** · generator: **{svc.engine.gen.name}**")

st.write("Ask about vulnerabilities, software, malware, attacks, defences or similar incidents. Every answer lists the facts and passages it is "
         "built from, and the system **refuses** when the data does not contain the answer.")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
        st.session_state["qa_pending"] = ex
st.session_state.setdefault("qa_history", [])


def render(res: dict, title: str | None = None):
    if title:
        st.markdown(f"##### {title}")
    if res["abstained"]:
        st.warning(res["answer"])
        return
    st.markdown(res["answer"].replace("\n- ", "\n\n- "))
    meta = f"intent **{res['intent']}** · confidence **{res['confidence']}**"
    g = res.get("grounding") or {}
    if g.get("grounded_fraction") is not None:
        meta += f" · {g['claims_found_in_source']}/{g['claims_checked']} cited claims verified verbatim in the source"
    st.caption(meta)
    if res["facts"]:
        with st.expander(f"Evidence and provenance of the graph facts ({len(res['facts'])})", expanded=False):
            ui.show_facts(res["facts"])
    if res["passages"]:
        with st.expander(f"Retrieved passages ({len(res['passages'])})", expanded=False):
            for p in res["passages"]:
                if p["chunk_id"] < 0:
                    st.markdown(f"- **#{p['incident_id']}** (similarity {p['score']:.2f}) {p['question'][:140]}")
                else:
                    sents = " ".join(f"“{s}”" for s in p["sentences"])
                    st.markdown(f"**#{p['incident_id']}** · {p['question'][:110]}  \n{sents}")
    if res["incidents"]:
        st.caption("Source incidents: " + ", ".join(f"#{i}" for i in res["incidents"]))


for item in st.session_state["qa_history"]:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        if item["mode"] == "compare":
            cc = st.columns(3)
            for col, (m, label) in zip(cc, (("vector", "Vector RAG"), ("hybrid", "Hybrid RAG"), ("graph", "Graph-RAG"))):
                with col:
                    render(item["res"][m], label)
        else:
            render(item["res"])

question = st.chat_input("Ask a cybersecurity question…") or st.session_state.pop("qa_pending", None)
if question:
    if not question.strip():
        st.warning("Please type a question.")
    else:
        try:
            with st.spinner("Retrieving evidence and building the answer…"):
                if choice == "Compare all three":
                    res, mode = svc.compare(question, k), "compare"
                else:
                    res, mode = svc.answer(question, MODES[choice], k), "single"
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not answer: {type(e).__name__}: {e}")
        else:
            st.session_state["qa_history"].append({"q": question, "res": res, "mode": mode})
            st.rerun()
if st.session_state["qa_history"] and st.button("Clear conversation"):
    st.session_state["qa_history"] = []
    st.rerun()
