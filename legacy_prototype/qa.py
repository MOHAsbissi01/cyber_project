"""RAG (vectoriel) vs Graph-RAG (vecteur + expansion sur le graphe) pour questions de securite."""
import os
import re

import numpy as np

from cyberlib import extract_entities
from graph_tools import nid

_SIMILAR = re.compile(r"similar|similaire|ressembl|like incident", re.I)
_VULN = re.compile(r"vuln[eé]rab|\bcve\b|\bcwe\b|faille|flaw", re.I)
_TARGET = re.compile(r"\btarget|\bcible|\baffect|\bvis[ée]|attack(s|ed)? what|what.*attack", re.I)
_INC_ID = re.compile(r"(?:incident|inc|#)\s*#?(\d+)", re.I)


class GraphRAG:
    def __init__(self, df, G, embedder, E):
        self.df, self.G, self.emb, self.E = df, G, embedder, E

    # -------- RAG vectoriel classique
    def vector_search(self, q, k=5, exclude=None):
        s = self.E @ self.emb.encode([q])[0]
        if exclude is not None:
            s[exclude] = -1
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

    # -------- Graph-RAG : amorces vectorielles + expansion via entites partagees
    def graph_search(self, q, k=5, n_seed=20, alpha=0.15, beta=0.05):
        q_ents = [nid(t, n) for t, n in extract_entities(q)]
        seeds = self.vector_search(q, n_seed)
        score = {i: s for i, s in seeds}
        for e in q_ents:                       # incidents relies directement aux entites de la question
            if e in self.G:
                for nb in self.G.neighbors(e):
                    if nb.startswith("INC:"):
                        i = int(nb[4:]); score[i] = score.get(i, 0) + alpha
        for i, s in seeds[:5]:                 # voisins des meilleures amorces (entites partagees)
            for e in self.G.neighbors(f"INC:{i}"):
                if not e.startswith("INC:"):
                    for nb in self.G.neighbors(e):
                        if nb.startswith("INC:") and int(nb[4:]) != i:
                            j = int(nb[4:]); score[j] = score.get(j, 0) + beta * s
        return sorted(score.items(), key=lambda x: -x[1])[:k]

    # -------- faits du graphe
    def facts(self, entities, limit=8):
        out = []
        for e in entities:
            if e not in self.G:
                continue
            nbs = [(nb, d) for nb, d in self.G[e].items() if d["relation"] not in ("MENTIONS", "SIMILAR_TO")]
            for nb, d in sorted(nbs, key=lambda x: -x[1]["weight"])[:limit]:
                a, b = self.G.nodes[e], self.G.nodes[nb]
                out.append(f"{a['label']} ({a['type']}) --{d['relation']}--> {b['label']} ({b['type']}) [x{d['cooc']}]")
        return out

    def _linked(self, e, types, limit=5):
        if e not in self.G:
            return []
        nbs = [(self.G.nodes[nb]["label"], d["cooc"]) for nb, d in self.G[e].items()
               if d["relation"] not in ("MENTIONS", "SIMILAR_TO") and self.G.nodes[nb]["type"] in types]
        return sorted(nbs, key=lambda x: -x[1])[:limit]

    # -------- reponse
    def answer(self, question, mode="graph", k=5, use_llm=False):
        ents = extract_entities(question)
        eids = [nid(t, n) for t, n in ents]
        m = _INC_ID.search(question)
        if _SIMILAR.search(question) and m and int(m.group(1)) < len(self.df):
            i0 = int(m.group(1))
            hits = self.vector_search(self.df.text[i0], k, exclude=i0)
            lines = [f"Incidents similaires a #{i0} ({self.df.question[i0][:70]}...) :"]
            lines += [f"- #{i} (cos={s:.2f}, {self.df.category[i]}): {self.df.question[i][:90]}" for i, s in hits]
            return {"answer": "\n".join(lines), "sources": [i for i, _ in hits], "facts": [], "entities": ents}

        hits = self.graph_search(question, k) if mode == "graph" else self.vector_search(question, k)
        facts = self.facts(eids) if mode == "graph" else []
        direct = None
        if mode == "graph":
            sw = [e for e in eids if e.startswith("SOFTWARE:")]
            at = [e for e in eids if e.startswith(("ATTACK:", "MALWARE:"))]
            if _VULN.search(question) and sw:
                v = self._linked(sw[0], {"CVE", "CWE"})
                if v:
                    direct = f"Vulnerabilites liees a {self.G.nodes[sw[0]]['label']} : " + ", ".join(f"{n} ({c} incidents)" for n, c in v)
            elif _TARGET.search(question) and at:
                v = self._linked(at[0], {"TARGET", "SOFTWARE"})
                if v:
                    direct = f"{self.G.nodes[at[0]]['label']} cible surtout : " + ", ".join(f"{n} ({c})" for n, c in v)
        ctx = [self.df.answer[i] for i, _ in hits[:3]]
        if use_llm and os.getenv("ANTHROPIC_API_KEY"):
            text = self._llm(question, ctx, facts)
        else:
            first = re.split(r"(?<=[.!?])\s", ctx[0])[:2] if ctx else []
            text = ((direct + "\n") if direct else "") + " ".join(first)
        return {"answer": text, "sources": [i for i, _ in hits], "facts": facts, "entities": ents}

    @staticmethod
    def _llm(question, ctx, facts):
        import anthropic
        prompt = ("Reponds a la question de cybersecurite en t'appuyant UNIQUEMENT sur le contexte et les faits du graphe.\n\n"
                  "FAITS DU GRAPHE:\n" + "\n".join(facts) + "\n\nCONTEXTE:\n" + "\n---\n".join(c[:1200] for c in ctx) +
                  f"\n\nQUESTION: {question}")
        r = anthropic.Anthropic().messages.create(model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=500,
                                                  messages=[{"role": "user", "content": prompt}])
        return r.content[0].text


def retrieval_eval(rag, df, test_idx, k=5):
    """Hit@k par categorie (label faible) : le top-k contient-il un incident de meme categorie ?
    Les incidents de test sont exclus du corpus de recherche. Compare RAG vs Graph-RAG."""
    res = {"vector": [], "graph": []}
    test = set(int(i) for i in test_idx)
    for i in test_idx:
        q = df.question[i]
        fns = (("vector", lambda q, n: rag.vector_search(q, n)),
               ("graph", lambda q, n: rag.graph_search(q, n, n_seed=n)))
        for mode, fn in fns:
            hits = [j for j, _ in fn(q, k + len(test)) if j not in test][:k]
            res[mode].append(any(df.category[j] == df.category[i] for j in hits))
    return {m: round(float(np.mean(v)), 4) for m, v in res.items()}
