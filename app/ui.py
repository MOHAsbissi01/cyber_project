"""Shared UI helpers: cached backend service, consistent colours/badges, interactive graph rendering, safe wrappers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from cyberintel.service import ArtifactsMissing, CyberIntelService  # noqa: E402
from cyberintel.viz import TYPE_COLORS  # noqa: E402

STATUS_ICON = {"verified": "✅", "reference": "📘", "unverified": "⚪", "differs_from_kb": "⚠️", "contradicted_by_kb": "❌"}
STATUS = {
    "verified": ("Verified by reference KB", "#1a7f37", "✅"),
    "reference": ("Reference KB", "#1f5fbf", "📘"),
    "unverified": ("Corpus only (unverified)", "#6b7280", "⚪"),
    "differs_from_kb": ("Differs from reference KB", "#b7791f", "⚠️"),
    "contradicted_by_kb": ("Contradicted by reference KB", "#c0392b", "❌"),
}
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.badge {display:inline-block; padding:2px 9px; border-radius:11px; font-size:0.78rem; font-weight:600; color:#fff; margin:1px 3px 1px 0;}
.card {background:#f3f6fb; border:1px solid #dfe6f1; border-radius:10px; padding:14px 16px; margin-bottom:10px; min-height:118px;}
.card h4 {margin:0 0 4px 0; font-size:0.95rem; color:#4a5668; font-weight:600;}
.card .v {font-size:1.7rem; font-weight:700; color:#1c2430; line-height:1.2;}
.card .s {font-size:0.8rem; color:#6b7280;}
.warn {background:#fff7e6; border-left:4px solid #e0a100; padding:10px 14px; border-radius:6px; font-size:0.9rem;}
.fact {border-left:3px solid #1f5fbf; padding:4px 10px; margin:6px 0; background:#fafcff;}
.small {font-size:0.82rem; color:#6b7280;}
</style>
"""


def setup(title: str, icon: str = "🛡️"):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f"{icon} {title}")


@st.cache_resource(show_spinner="Loading the knowledge base, indexes and models (first start only)…")
def _load(store: str, llm: str) -> CyberIntelService:
    return CyberIntelService(store=store, llm=llm)


def service() -> CyberIntelService | None:
    """Cached backend. On a missing build shows a clear instruction instead of a stack trace."""
    try:
        return _load(st.session_state.get("store_kind", "networkx"), st.session_state.get("llm_kind", "none"))
    except ArtifactsMissing as e:
        st.error(f"The project has not been built yet. {e}")
        st.code("python -m cyberintel.pipeline all", language="bash")
        return None
    except Exception as e:  # noqa: BLE001 - surface any loading problem to the user
        st.error(f"Could not start the backend: {type(e).__name__}: {e}")
        return None


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def type_badge(etype: str, name: str) -> str:
    return badge(f"{name}", TYPE_COLORS.get(etype, "#444"))


def status_badge(status: str) -> str:
    label, color, icon = STATUS.get(status, (status, "#6b7280", ""))
    return badge(f"{icon} {label}", color)


def kpi(col, title: str, value, sub: str = ""):
    col.markdown(f'<div class="card"><h4>{title}</h4><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def caveat(text: str):
    st.markdown(f'<div class="warn">⚠️ {text}</div>', unsafe_allow_html=True)


def render_graph(H: nx.DiGraph, center: str | None = None, height: int = 520, edge_labels: bool = True):
    """Interactive pyvis graph (drag, zoom, hover for details). H: entity DiGraph with 'relations' and 'weight' on edges."""
    if H.number_of_nodes() == 0:
        st.info("Nothing to display for this selection.")
        return
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("pyvis is not installed (`pip install pyvis`): showing the table view only.")
        return
    net = Network(height=f"{height}px", width="100%", cdn_resources="in_line", directed=True, bgcolor="#ffffff", font_color="#1c2430")
    net.barnes_hut(gravity=-6000, spring_length=140)
    net.set_options('{"nodes": {"font": {"size": 18, "strokeWidth": 3, "strokeColor": "#ffffff"}, "scaling": {"label": {"enabled": false}}}, '
                    '"edges": {"font": {"size": 11, "align": "middle", "strokeWidth": 3, "strokeColor": "#ffffff"}}, '
                    '"physics": {"stabilization": {"iterations": 200}}}')
    for n, d in H.nodes(data=True):
        net.add_node(n, label=d["label"], color=TYPE_COLORS.get(d["type"], "#444"), size=30 if n == center else 12 + 3 * min(H.degree(n), 8),
                     title=f"{d['type']}: {d['label']} ({d.get('n_incidents', 0)} incidents)", borderWidth=3 if n == center else 1)
    for a, b, d in H.edges(data=True):
        rel = "/".join(d["relations"])
        net.add_edge(a, b, label=rel if edge_labels else "", title=f"{rel} (support {d['weight']})", value=max(d["weight"], 1) ** 0.5, arrows="to",
                     color="#9aa5b1")
    html = net.generate_html(notebook=False)
    if hasattr(st, "iframe"):          # recent Streamlit; components.html is deprecated there
        st.iframe(html, height=height + 20)
    else:
        components.html(html, height=height + 20, scrolling=False)
    present = sorted({d["type"] for _, d in H.nodes(data=True)})
    st.markdown(" ".join(badge(t, TYPE_COLORS.get(t, "#444")) for t in present), unsafe_allow_html=True)


def show_facts(facts: list[dict]):
    for f in facts:
        origin = {"corpus": "corpus", "reference_kb": "reference KB", "corpus+reference_kb": "corpus + reference KB"}.get(f["source"], f["source"])
        ev = [e for e in f.get("evidence", []) if e.get("sentence")]
        st.markdown(f'<div class="fact"><b>{f["text"]}</b><br>{status_badge(f["status"])} '
                    f'<span class="small">{origin}{f" · support {f["support"]} incident(s)" if f["support"] else ""}</span></div>', unsafe_allow_html=True)
        if ev:
            with st.expander("Evidence sentences"):
                for e in ev:
                    st.markdown(f"> {e['sentence']}" + (f"  \n<span class='small'>incident #{e['incident']}</span>" if e.get("incident") is not None else ""),
                                unsafe_allow_html=True)


def show_incident_expander(svc: CyberIntelService, i: int, label: str | None = None):
    inc = svc.incident(int(i))
    with st.expander(label or f"#{inc['id']} · {inc['category']} · {inc['question'][:100]}"):
        st.markdown(f"**Question.** {inc['question']}")
        st.markdown(f"**Answer.** {inc['answer'][:2200]}{'…' if len(inc['answer']) > 2200 else ''}")
        ents = [e for e in inc["entities"] if e["type"] != "TARGET"]
        if ents:
            st.markdown(" ".join(type_badge(e["type"], e["label"]) for e in ents[:25]), unsafe_allow_html=True)
        st.caption(f"category (weak label): {inc['category']} · split: {inc['split']} · graph community: {inc['community']}")


def df_or_info(df: pd.DataFrame, empty_msg: str = "No data.", **kw):
    if df is None or len(df) == 0:
        st.info(empty_msg)
    else:
        st.dataframe(df, hide_index=True, width="stretch", **kw)


def nav_link(col, page: str, label: str, icon: str):
    """page_link only works inside st.navigation; degrade silently when a view is rendered on its own (tests)."""
    try:
        col.page_link(page, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - StreamlitPageNotFoundError
        col.markdown(f"{label}")
