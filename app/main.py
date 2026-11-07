"""CyberIntel - Streamlit entry point.   Run:  streamlit run app/main.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="CyberIntel", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/incident_analysis.py", title="Incident analysis", icon=":material/search_insights:"),
    st.Page("views/graph_explorer.py", title="Knowledge graph", icon=":material/hub:"),
    st.Page("views/question_answering.py", title="Question answering", icon=":material/forum:"),
    st.Page("views/evaluation.py", title="Model evaluation", icon=":material/fact_check:"),
    st.Page("views/dataset_explorer.py", title="Dataset explorer", icon=":material/dataset:"),
]
with st.sidebar:
    st.markdown("### 🛡️ CyberIntel")
    st.caption("Cybersecurity text analysis · knowledge graph · graph ML · Graph-RAG\n\nESPRIT 5ERP-BI · BD-45 · subject S2")
    with st.expander("Backend settings"):
        use_neo = st.toggle("Use Neo4j for graph queries", value=False,
                            help="Needs a running Neo4j loaded with scripts/neo4j_import.py and NEO4J_PASSWORD set. Falls back to the in-memory graph.")
        st.session_state["store_kind"] = "neo4j" if use_neo else "networkx"
        use_llm = st.toggle("Claude answer generation", value=False, disabled=not os.getenv("ANTHROPIC_API_KEY"),
                            help="Needs ANTHROPIC_API_KEY. Default: extractive generation (offline, cannot hallucinate).")
        st.session_state["llm_kind"] = "claude" if use_llm else "none"
st.navigation(pages).run()
