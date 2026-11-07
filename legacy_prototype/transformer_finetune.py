"""Fine-tuning DistilBERT pour la classification d'incidents (AA5/AA6). NON teste dans cette livraison :
necessite le telechargement du modele (distilbert-base-uncased, ~270 Mo) -> a lancer sur votre machine/Colab (GPU conseille).

  python transformer_finetune.py --source hf --epochs 3
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cyberlib import load_data, weak_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    df = load_data(a.source)
    df["category"] = [weak_label(q, r) for q, r in zip(df.question, df.answer)]
    lab = df.category.astype("category"); y = lab.cat.codes.values; names = list(lab.cat.categories)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0)  # meme split que run_pipeline

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(names))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    enc = lambda idx: tok(df.answer.iloc[idx].tolist(), truncation=True, max_length=256, padding=True, return_tensors="pt")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for ep in range(a.epochs):
        model.train(); np.random.shuffle(tr)
        for i in range(0, len(tr), a.bs):
            b = tr[i:i + a.bs]
            x = {k: v.to(dev) for k, v in enc(b).items()}
            loss = model(**x, labels=torch.tensor(y[b]).to(dev)).loss
            loss.backward(); opt.step(); opt.zero_grad()
        print(f"epoch {ep + 1} loss {loss.item():.3f}")
    model.eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(te), 32):
            x = {k: v.to(dev) for k, v in enc(te[i:i + 32]).items()}
            pred += model(**x).logits.argmax(1).cpu().tolist()
    print(classification_report(y[te], pred, labels=range(len(names)), target_names=names, zero_division=0))
    print("macro-F1:", f1_score(y[te], pred, average="macro"))


if __name__ == "__main__":
    main()
