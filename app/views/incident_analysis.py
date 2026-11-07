"""Incident analysis: paste a text -> topic classification, entities, related incidents, knowledge-graph context."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

EXAMPLE = ("Our SOC detected a phishing campaign: employees received emails with a fake login page and several credentials were stolen. "
           "The attacker then used Mimikatz to dump credentials from memory on a Windows server and moved laterally through the Active Directory "
           "domain with pass-the-hash, abusing CVE-2020-1472 (Zerologon). Traffic to 185.220.101.4 was observed.")

TYPE_LABEL = {"CVE": "CVE", "CWE": "CWE", "IP": "IP address", "HASH": "File hash", "VULN_NAME": "Named vulnerability",
              "MITRE_TECHNIQUE": "ATT&CK technique", "TACTIC": "ATT&CK tactic", "THREAT_ACTOR": "Threat actor", "ATTACK": "Attack / malware class",
              "TARGET": "Target (generic)", "DEFENSE": "Defence", "TOOL": "Security tool", "TECHNOLOGY": "Technology / protocol",
              "SOFTWARE": "Software", "MALWARE": "Malware family"}
ui.setup("Incident analysis")
st.write("Paste an incident description or a security question. The tool classifies the topic, extracts entities, finds similar records and "
         "shows what the knowledge graph knows about the entities it recognised.")
svc = ui.service()
if svc is None:
    st.stop()

text = st.text_area("Incident text", value=st.session_state.get("ia_text", EXAMPLE), height=170, key="ia_text_area")
c1, c2 = st.columns([1, 5])
k = c1.number_input("Related incidents", 1, 10, 5)
go = c2.button("Analyse", type="primary")
if not go:
    st.info("Press **Analyse** to run the pipeline on the text above.")
    st.stop()
if not text.strip():
    st.warning("The text is empty. Paste an incident description first.")
    st.stop()
try:
    with st.spinner("Classifying, extracting entities and searching related incidents…"):
        r = svc.analyze_incident(text, int(k))
except Exception as e:  # noqa: BLE001
    st.error(f"Analysis failed: {type(e).__name__}: {e}")
    st.stop()
st.success("Analysis complete.")

a, b = st.columns([1, 1])
with a:
    st.subheader("Predicted topic")
    st.markdown(f"### {r['category'].replace('_', ' ')}")
    st.progress(min(1.0, r["confidence"]), text=f"model confidence {r['confidence']:.0%}")
    probs = pd.Series(r["probabilities"]).sort_values(ascending=False)
    st.bar_chart(probs.head(6), horizontal=True, height=230)
    if r["top_terms"]:
        st.caption("Terms that push the model towards this class: " + ", ".join(r["top_terms"][:8]))
    ui.caveat("Topics come from keyword-based weak labels; the model predicts <i>agreement with those rules</i>. "
              "See <b>Model evaluation</b> for the independent gold-set accuracy.")
with b:
    st.subheader("Extracted entities")
    if not r["entities"]:
        st.info("No entity recognised (the NER is rule-based: it only knows its gazetteers and identifier patterns).")
    for t, names in r["entities"].items():
        st.markdown(f"**{TYPE_LABEL.get(t, t.replace('_', ' ').title())}**<br>" + " ".join(ui.type_badge(t, n) for n in names), unsafe_allow_html=True)

st.subheader("What the knowledge graph knows about these entities")
if r["graph_context"]:
    df = pd.DataFrame(r["graph_context"]).rename(columns={"head": "from", "tail": "to"})
    df["check"] = df["status"].map(lambda s: ui.STATUS_ICON.get(s, "") + " " + ui.STATUS.get(s, (s,))[0])
    st.dataframe(df[["from", "relation", "to", "support", "source", "check"]], hide_index=True, width="stretch")
else:
    st.info("None of the recognised entities has a relation in the graph.")

st.subheader(f"{len(r['similar'])} related incident records")
for s in r["similar"]:
    ui.show_incident_expander(svc, s["id"], f"#{s['id']} · {s['category']} · relevance {s['relevance']:.2f} · {s['question'][:90]}")

with st.expander("Preprocessing steps applied to the first 300 characters (course Chap. 1)"):
    p = r["preprocessing"]
    st.markdown("**Tokens:** " + " ".join(p["tokens"][:40]))
    st.markdown("**After stop-word removal:** " + " ".join(p["no_stopwords"][:40]))
    st.markdown("**Stems (Porter):** " + " ".join(p["stems"][:40]))
    st.markdown("**Lemmas (WordNet):** " + " ".join(p["lemmas"][:40]))
    st.markdown("**POS tags:** " + " ".join(f"{w}/{t}" for w, t in p["pos"][:30]))
