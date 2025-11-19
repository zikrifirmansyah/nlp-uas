# NLP Preprocessing Web App

Simple Flask app to preprocess text from `dataNew.csv` using Pandas, NumPy, and NLTK.

Features:

- Load text data from a CSV (default: `dataNew.csv`).
- Cleaning: remove URLs, emojis, mentions, punctuation, digits.
- Tokenization using NLTK.
- Stopword removal (attempts to use NLTK Indonesian stopwords; fallback list included).
- Simple web UI to preview original data and see preprocessing results.

Requirements:

```bash
python3 -m pip install -r requirements.txt
```

Run:

```bash
python main.py
# then open http://127.0.0.1:5500 in your browser
```

Notes:

- On first run, NLTK may download `punkt` and `stopwords` datasets automatically.
- By default the UI points to `dataNew.csv` in the same folder.
