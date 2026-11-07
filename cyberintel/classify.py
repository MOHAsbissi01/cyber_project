"""Incident/topic classification: classical models, preprocessing ablation, evaluation on weak AND gold labels.

Protocol (identical for every model of the project):
  * one deterministic stratified split (train/val/test) stored in the corpus;
  * every vectoriser/embedder is fitted on TRAIN only;
  * hyper-parameters are chosen on VAL only; TEST is touched once per model;
  * metrics are reported on (a) the weak-label test set and (b) the independent gold subset of the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .embeddings import Embedder
from .preprocessing import make_analyzer


def metrics_block(y_true, y_pred, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0, output_dict=True)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": {c: {k: round(float(rep[c][k]), 4) if k != "support" else int(rep[c][k])
                          for k in ("precision", "recall", "f1-score", "support")} for c in classes},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _encode(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return df.category.map(idx).values


def _eval(name_pred: np.ndarray, df: pd.DataFrame, y: np.ndarray, gold: pd.DataFrame | None, classes: list[str]) -> dict:
    """name_pred: predictions for ALL rows of df (aligned to df.index)."""
    te = np.where(df.split.values == "test")[0]
    out = {"test_weak": metrics_block(y[te], name_pred[te], classes)}
    if gold is not None and len(gold):
        gi = df.reset_index(drop=True).reset_index().merge(gold[["id", "gold_label"]], on="id")
        yg = gi.gold_label.map({c: i for i, c in enumerate(classes)}).values
        out["test_gold"] = metrics_block(yg, name_pred[gi["index"].values], classes)
        out["test_gold"]["labels_source"] = sorted(gold.gold_source.unique().tolist())
    return out


def _grid(make, grid, Xtr, ytr, Xva, yva):
    best = (-1, None, None)
    for params in grid:
        m = make(**params).fit(Xtr, ytr)
        f = f1_score(yva, m.predict(Xva), average="macro", zero_division=0)
        if f > best[0]:
            best = (f, params, m)
    return best


def train_and_evaluate(df: pd.DataFrame, classes: list[str], gold: pd.DataFrame | None = None, seed: int = 42):
    """Fit all classical models. Returns (results_dict, fitted_artifacts_for_serving)."""
    y = _encode(df, classes)
    tr, va = (df.split.values == "train"), (df.split.values == "val")
    results: dict = {"classes": classes, "models": {}, "baselines": {}}

    # ---- baselines
    maj = np.bincount(y[tr]).argmax()
    results["baselines"]["majority_class"] = _eval(np.full(len(df), maj), df, y, gold, classes)
    if gold is not None and len(gold):
        gm = gold.merge(df[["id", "category"]], on="id")
        results["rules_vs_gold"] = {
            "n": int(len(gm)), "agreement": round(float((gm.gold_label == gm.category).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(gm.gold_label, gm.category)), 4),
            "note": "How often the keyword rules (weak labels) agree with the independent gold labels."}

    # ---- TF-IDF models
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english", max_features=60000)
    X = vec.fit(df.text[tr]).transform(df.text)
    Xtr, Xva = X[tr], X[va]
    specs = {
        "tfidf_logreg": (lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced"), [{"C": c} for c in (0.5, 2, 8, 32)]),
        "tfidf_linear_svm": (lambda C: LinearSVC(C=C, class_weight="balanced"), [{"C": c} for c in (0.1, 0.5, 2)]),
        "tfidf_complement_nb": (lambda alpha: ComplementNB(alpha=alpha), [{"alpha": a} for a in (0.1, 0.3, 1.0)]),
    }
    fitted = {}
    for name, (make, grid) in specs.items():
        f, params, m = _grid(make, grid, Xtr, y[tr], Xva, y[va])
        results["models"][name] = {"val_macro_f1": round(float(f), 4), "params": params, **_eval(m.predict(X), df, y, gold, classes)}
        fitted[name] = m

    # ---- LSA + MLP (dense features, same kind of input the GNNs receive)
    emb = Embedder("tfidf").fit(df.text[tr])
    E = emb.encode(df.text)
    f, params, m = _grid(lambda h: MLPClassifier((h,), max_iter=400, early_stopping=True, random_state=seed),
                         [{"h": h} for h in (64, 128)], E[tr], y[tr], E[va], y[va])
    results["models"]["lsa_mlp"] = {"val_macro_f1": round(float(f), 4), "params": params, **_eval(m.predict(E), df, y, gold, classes)}

    # ---- top words per class from the logistic regression (interpretability)
    lr = fitted["tfidf_logreg"]
    vocab = np.array(vec.get_feature_names_out())
    results["top_terms"] = {c: vocab[np.argsort(-lr.coef_[i])[:10]].tolist() for i, c in enumerate(classes)}
    serving = {"vectorizer": vec, "model": lr, "classes": classes}
    return results, serving


def preprocessing_ablation(df: pd.DataFrame, classes: list[str]) -> dict:
    """AA3: effect of each preprocessing step (raw tokens / +stop words / +stemming / +lemmatisation) on macro-F1.
    Same model (LogReg C=2, unigram TF-IDF), same split; fitted on train; reported on val and test (weak labels)."""
    y = _encode(df, classes)
    tr, va, te = (df.split.values == s for s in ("train", "val", "test"))
    out = {}
    for mode in ("raw", "stop", "stem", "lemma"):
        v = TfidfVectorizer(analyzer=make_analyzer(mode), min_df=2, sublinear_tf=True)
        X = v.fit(df.text[tr]).transform(df.text)
        m = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        p = m.predict(X)
        out[mode] = {"vocab_size": len(v.vocabulary_),
                     "val_macro_f1": round(float(f1_score(y[va], p[va], average="macro", zero_division=0)), 4),
                     "test_macro_f1": round(float(f1_score(y[te], p[te], average="macro", zero_division=0)), 4)}
    out["note"] = "Weak labels (keyword rules): differences between variants are small and partly an artefact of the rules."
    return out


def predict_proba(serving: dict, texts: list[str]) -> pd.DataFrame:
    P = serving["model"].predict_proba(serving["vectorizer"].transform(texts))
    return pd.DataFrame(P, columns=[serving["classes"][c] for c in serving["model"].classes_])
