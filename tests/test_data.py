import numpy as np
import pandas as pd

from cyberintel.data import assign_splits, clean_text, make_chunks
from cyberintel.textutil import sentence_spans, split_sentences


def test_clean_removes_closed_and_unclosed_think():
    t, _ = clean_text("<think>secret plan</think> The real answer is here and it is long enough to keep around for tests.")
    assert "secret" not in t and "real answer" in t
    t, _ = clean_text("<think>Okay, so I need to explain X. The model was cut off here")
    assert t.strip() == ""                      # unfinished reasoning has no answer


def test_clean_removes_reasoning_sentences_and_literal_newlines():
    t, n = clean_text("Okay, so I need to explain this. ASLR randomises addresses.\\n\\nIt helps against ROP. Let me think about it.")
    assert n == 2 and "ASLR randomises" in t and "\\n" not in t and "Let me" not in t


def test_sentence_split_and_spans_are_consistent():
    text = "First sentence here. Second one follows! 3) A numbered item starts. Last."
    spans = sentence_spans(text)
    assert [text[a:b].strip() for a, b in spans] == split_sentences(text)
    assert len(spans) >= 3


def test_split_is_stratified_deterministic_and_disjoint():
    rng = np.random.RandomState(0)
    labels = pd.Series(rng.choice(["a", "b", "c"], 300, p=[0.6, 0.3, 0.1]))
    df = pd.DataFrame({"id": range(300)})
    s1, s2 = assign_splits(df, labels, seed=1), assign_splits(df, labels, seed=1)
    assert (s1 == s2).all()                                         # deterministic
    assert set(s1) == {"train", "val", "test"}
    shares = pd.crosstab(labels, s1, normalize="columns")
    assert (shares.max(axis=1) - shares.min(axis=1)).max() < 0.06     # same class proportions in every split


def test_chunks_cover_all_sentences(mini_df):
    ch = make_chunks(mini_df, target_chars=80)
    assert set(ch.incident_id) == set(mini_df.id)
    for i, g in ch.groupby("incident_id"):
        assert " ".join(g.sort_values("order").text).replace(" ", "") == mini_df.answer[i].replace(" ", "")
