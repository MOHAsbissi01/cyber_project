"""Dashboard: what the system analysed, what it found, and what to be careful about."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

ui.setup("Dashboard")
svc = ui.service()
if svc is None:
    st.stop()

ov = svc.overview()
if ov.get("graph_backend_note"):
    st.warning(ov["graph_backend_note"])
ui.caveat("The dataset is an LLM-generated Q&A corpus about security <i>concepts</i>, not real incident reports, and it has no labels. "
          "Categories are keyword-based <b>weak labels</b>; accuracy figures elsewhere are always reported against an independent gold subset too.")

c = st.columns(5)
ui.kpi(c[0], "Incident records", f"{ov['incidents']:,}", "cleaned Q&A pairs")
ui.kpi(c[1], "Specific entities", f"{ov['specific_entities']:,}", f"{ov['unique_entities']:,} incl. generic targets")
ui.kpi(c[2], "Graph nodes", f"{ov['graph_nodes']:,}", f"{ov['graph_edges']:,} edges")
ui.kpi(c[3], "Semantic relations", f"{ov['semantic_relations']:,}", "with evidence")
ui.kpi(c[4], "Incident clusters", ov["communities"], "Louvain communities")

left, right = st.columns(2)
with left:
    st.subheader("Threat / topic distribution")
    cat = pd.Series(ov["category_counts"]).sort_values(ascending=False)
    st.bar_chart(cat, horizontal=True, height=330)
    st.caption("Weak labels (keyword rules). 'phishing' has very few examples, treat its metrics with caution.")
with right:
    st.subheader("Entities by type")
    ent = pd.Series(ov["entity_type_counts"]).sort_values(ascending=False)
    st.bar_chart(ent, horizontal=True, height=330)
    st.caption("Distinct (type, name) entities found by the rule-based NER.")

st.subheader("Important insights")
gs = ov["graph_stats"]
rel = svc.relations_table(source="corpus").sort_values("support", ascending=False)
claims = ov["claims_check"]
ner = ov["ner"]
a, b = st.columns(2)
with a:
    st.markdown("**Strongest relations found in the corpus**")
    top = rel.head(6)
    for r in top.itertuples():
        st.markdown(f"- **{r.head}** → *{r.relation}* → **{r.tail}** · {r.support} incidents (lift {r.lift})")
    st.markdown("**Most central entities (PageRank)**")
    st.markdown(", ".join(f"{n} ({t})" for n, t, _ in gs.get("pagerank_top", [])[:8]))
with b:
    st.markdown("**Fact checking against the reference KB**")
    if claims:
        st.markdown(f"Of the corpus CVE claims the reference KB can judge: **{claims.get('verified', 0)} verified**, "
                    f"**{claims.get('differs_from_kb', 0)} differ**, **{claims.get('contradicted_by_kb', 0)} contradicted** "
                    f"({claims.get('unverified', 0)} relations are corpus-only and unverified).")
        st.caption("The corpus answers were produced by an LLM and some CVE explanations are wrong (e.g. it calls CVE-2017-0290 'EternalBlue').")
    st.markdown("**NER coverage**")
    st.markdown(f"{ner.get('unique_cves', 0)} distinct CVE ids · {ner.get('incidents_with_specific_entity', 0):,} of {ov['incidents']:,} records "
                f"contain at least one specific entity.")

st.subheader("Where to go next")
n = st.columns(4)
ui.nav_link(n[0], "views/incident_analysis.py", "Analyse an incident text", ":material/search_insights:")
ui.nav_link(n[1], "views/graph_explorer.py", "Explore the knowledge graph", ":material/hub:")
ui.nav_link(n[2], "views/question_answering.py", "Ask a security question", ":material/forum:")
ui.nav_link(n[3], "views/evaluation.py", "See the evaluation", ":material/fact_check:")
