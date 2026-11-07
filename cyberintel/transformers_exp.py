"""Transformer experiments (AA5/AA6): sentence-transformer embeddings and DistilBERT fine-tuning, on the SHARED split.

    python -m cyberintel.transformers_exp minilm        # all-MiniLM-L6-v2 embeddings + LogReg   (~90 MB model, minutes on CPU)
    python -m cyberintel.transformers_exp distilbert    # DistilBERT fine-tuning                 (~270 MB model, ~tens of minutes on CPU)

Both write predictions for every row (reports/<name>_preds.csv) so they can be scored on the gold labels later, plus a metrics json.
Model selection uses VAL only (epoch / C); TEST is evaluated once.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .classify import metrics_block
from .pipeline import art, load_corpus, rep, save_json
from .labels import CLASSES


def _prep():
    df = load_corpus()
    classes = sorted(df.category.unique().tolist())
    y = df.category.map({c: i for i, c in enumerate(classes)}).values
    return df, classes, y


def _finish(name, df, classes, y, pred, extra):
    te, va = df.split.values == "test", df.split.values == "val"
    out = {"model": name, "classes": classes, "val_macro_f1": round(float(f1_score(y[va], pred[va], average="macro", zero_division=0)), 4),
           "test_weak": metrics_block(y[te], pred[te], classes), **extra}
    save_json(out, rep(f"{name}.json"))
    pd.DataFrame({"id": df.id, "split": df.split, "pred": [classes[i] for i in pred]}).to_csv(rep(f"{name}_preds.csv"), index=False)
    print(f"[{name}] val macro-F1={out['val_macro_f1']}  test macro-F1={out['test_weak']['macro_f1']}  acc={out['test_weak']['accuracy']}")
    return out


def run_minilm():
    from .embeddings import Embedder
    df, classes, y = _prep()
    t0 = time.time()
    emb = Embedder("st")                      # fitting = loading the pretrained model; no training on our data (unsupervised)
    emb.fit([])
    E = emb.encode(df.text.tolist(), batch_size=32)
    np.save(art("embeddings_minilm.npy"), E)
    tr, va = df.split.values == "train", df.split.values == "val"
    best = (-1, None, None)
    for C in (1, 4, 16, 64):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(E[tr], y[tr])
        f = f1_score(y[va], m.predict(E[va]), average="macro", zero_division=0)
        if f > best[0]:
            best = (f, C, m)
    return _finish("minilm_logreg", df, classes, y, best[2].predict(E), {"params": {"C": best[1]}, "encoder": emb.name,
                                                                          "seconds": round(time.time() - t0, 1)})


def run_distilbert(epochs: int = 3, batch: int = 16, max_len: int = 160, lr: float = 5e-5, seed: int = 42):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    df, classes, y = _prep()
    torch.manual_seed(seed); np.random.seed(seed)
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(classes))
    tr_i, va_i = np.where(df.split.values == "train")[0], np.where(df.split.values == "val")[0]
    cnt = np.bincount(y[tr_i], minlength=len(classes)).astype(float)
    w = torch.tensor((cnt.sum() / (len(classes) * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)   # sqrt-inverse-frequency
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr_i) / batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))

    def enc(idx):
        return tok(df.question.iloc[idx].tolist(), df.answer.iloc[idx].tolist(), truncation="only_second", max_length=max_len,
                   padding=True, return_tensors="pt")

    def predict(idx):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                out.append(model(**enc(idx[i:i + 64])).logits.argmax(1).numpy())
        return np.concatenate(out)

    t0, best_f, best_state, history = time.time(), -1.0, None, []
    for ep in range(epochs):
        model.train(); rng = np.random.RandomState(seed + ep); order = rng.permutation(tr_i); tot = 0.0
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            loss = loss_fn(model(**enc(b)).logits, torch.tensor(y[b]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); tot += loss.item() * len(b)
        f = f1_score(y[va_i], predict(va_i), average="macro", zero_division=0)
        history.append({"epoch": ep + 1, "train_loss": round(tot / len(order), 4), "val_macro_f1": round(float(f), 4),
                        "elapsed_s": round(time.time() - t0)})
        print(history[-1], flush=True)
        if f > best_f:
            best_f, best_state = f, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict(np.arange(len(df)))
    return _finish("distilbert", df, classes, y, pred, {"history": history, "base_model": name, "max_len": max_len, "lr": lr,
                                                        "epochs": epochs, "seconds": round(time.time() - t0, 1),
                                                        "device": "cpu"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["minilm", "distilbert"])
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    run_minilm() if a.which == "minilm" else run_distilbert(epochs=a.epochs)
