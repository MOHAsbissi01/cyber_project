"""Knowledge-graph explorer: search an entity, see its relations with evidence, the interactive neighbourhood, clusters and all relations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui  # noqa: E402
from cyberintel.ner import ENTITY_TYPES  # noqa: E402

ui.setup("Knowledge graph explorer", "🕸️")
svc = ui.service()
if svc is None:
    st.stop()

tab_entity, tab_global, tab_clusters, tab_rel = st.tabs(["Explore an entity", "Global graph", "Clusters", "All relations"])

# ----------------------------------------------------------------------------- entity
with tab_entity:
    c1, c2, c3 = st.columns([2, 2, 3])
    types = c1.multiselect("Entity type", ENTITY_TYPES, default=["SOFTWARE", "ATTACK", "MALWARE", "CVE", "VULN_NAME"])
    q = c2.text_input("Search", placeholder="e.g. windows, ROP, mimikatz, CVE-2016")
    hits = svc.search_entities(q, types or None, limit=60)
    if hits.empty:
        c3.info("No entity matches. Clear the search or add types.")
        st.stop()
    labels = {r.id: f"{r.label} [{r.type}] · {r.incidents} incidents · {r.relations} linked entities" for r in hits.itertuples()}
    eid = c3.selectbox("Entity", list(labels), format_func=labels.get)
    prof = svc.entity_profile(eid)
    if not prof:
        st.warning("Entity not found in the graph backend.")
        st.stop()
    m = prof["entity"]
    st.markdown(f"### {m['label']}  " + ui.type_badge(m["type"], m["type"]), unsafe_allow_html=True)
    if m.get("kb_summary"):
        st.info(f"**{m.get('kb_name')}** (reference KB): {m['kb_summary']}")
    st.caption(f"Mentioned in {prof['n_incidents']} incident records · {len(prof['relations'])} relation(s)")

    left, right = st.columns([5, 4])
    with left:
        st.markdown("**Relations**")
        if not prof["relations"]:
            st.info("No semantic relation was extracted for this entity (it co-occurs with others only at document level).")
        else:
            rows = [{"relation": f"{r['relation']} {'→' if r['direction'] == 'out' else '←'}", "other entity": f"{r['other_label']} · {r['other_type']}",
                     "support": r["support"], "lift": round(r["lift"] or 0, 2), "source": r["source"],
                     "check": ui.STATUS_ICON.get(r["status"], "")}
                    for r in prof["relations"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("check: ✅ verified by the reference KB · 📘 reference KB fact · ⚪ corpus only (unverified) · ⚠️ differs from KB · ❌ contradicted by KB")
            with st.expander("Evidence sentences behind the strongest relations"):
                for r in prof["relations"][:6]:
                    for e in r["evidence"][:1]:
                        st.markdown(f"**{svc.G.nodes[r['head']]['label']} → {r['relation']} → {svc.G.nodes[r['tail']]['label']}**  \n> {e['sentence']}"
                                    + (f"  \n<span class='small'>incident #{e['incident']}</span>" if e.get("incident") is not None else ""),
                                    unsafe_allow_html=True)
    with right:
        st.markdown("**Neighbourhood**")
        sub = svc.subgraph(eid)
        ui.render_graph(sub, center=eid, height=420, edge_labels=sub.number_of_nodes() <= 12)
    st.markdown("**Incident records that mention it**")
    if prof["incidents"]:
        for i in prof["incidents"][:8]:
            ui.show_incident_expander(svc, i["id"], f"#{i['id']} · {i['category']} · {i['mentions']} mention(s) · {i['question'][:90]}")
    else:
        st.info("Not mentioned in any incident record (it comes from the reference KB only).")

# ----------------------------------------------------------------------------- global
with tab_global:
    c1, c2, c3 = st.columns(3)
    gtypes = c1.multiselect("Types shown", ENTITY_TYPES, default=["ATTACK", "DEFENSE", "SOFTWARE", "TECHNOLOGY", "MALWARE", "TOOL"], key="gt")
    minsup = c2.slider("Minimum support (incidents)", 1, 20, 4)
    maxn = c3.slider("Maximum nodes", 20, 120, 60)
    with st.spinner("Building the graph view…"):
        H = svc.global_entity_graph(gtypes or None, minsup, maxn)
    ui.render_graph(H, height=620)
    st.caption(f"{H.number_of_nodes()} entities · {H.number_of_edges()} relations shown. Arrow = direction (head → tail). Drag nodes, scroll to zoom, hover for details.")

# ----------------------------------------------------------------------------- clusters
with tab_clusters:
    ct = svc.communities_table()
    if ct.empty:
        st.info("Run the pipeline stage `kg` to compute communities.")
    else:
        gs = svc.report("graph_stats.json") or {}
        q = gs.get("incident_communities", {})
        st.markdown(f"**Incident communities** (Louvain on semantic-similarity + shared-entity graph): {q.get('n_communities')} clusters · "
                    f"weighted purity vs weak labels {q.get('weighted_purity')} · NMI {q.get('nmi_vs_weak_labels')}")
        st.dataframe(ct, hide_index=True, width="stretch")
        c = st.selectbox("Inspect a community", ct["community"].tolist(),
                         format_func=lambda c: f"#{c} · {ct.set_index('community').loc[c, 'top_terms']}")
        ui.df_or_info(svc.community_incidents(int(c)), "Empty community.")
    ec = pd.read_csv(ui.ROOT / "reports" / "entity_communities.csv") if (ui.ROOT / "reports" / "entity_communities.csv").exists() else pd.DataFrame()
    if len(ec):
        st.markdown("**Entity communities** (groups of related attacks / defences / software on the semantic graph)")
        st.dataframe(ec.head(15), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- relations
with tab_rel:
    c1, c2, c3 = st.columns(3)
    rel = c1.selectbox("Relation", ["(all)"] + sorted(svc.relations.relation.unique().tolist()))
    src = c2.selectbox("Source", ["(all)", "corpus", "reference_kb"])
    stt = c3.selectbox("Check", ["(all)"] + sorted(svc.relations.status.unique().tolist()))
    t = svc.relations_table(None if rel == "(all)" else rel, None if src == "(all)" else src, None if stt == "(all)" else stt)
    st.caption(f"{len(t):,} relations")
    ui.df_or_info(t.sort_values("support", ascending=False), "No relation matches these filters.")
