"""One-time download of the small NLTK resources (WordNet for lemmatisation, the POS tagger) into ./nltk_data.

NLTK's default folder is blocked on some Windows Store Python installs ("Security Violation ... Unauthorized path"), so the project keeps
its own copy next to the code; `cyberintel/preprocessing.py` adds that folder to NLTK's search path.

    python scripts/setup_nltk.py
"""
from pathlib import Path

import nltk

target = Path(__file__).resolve().parent.parent / "nltk_data"
target.mkdir(exist_ok=True)
for resource in ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"):
    ok = nltk.download(resource, download_dir=str(target), quiet=True)
    print(f"{resource:35s} {'ok' if ok else 'FAILED'}")
