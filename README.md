# BITS Information Retrieval — Assignment 1

## Deliverables

- `app.py` — Streamlit front end for ingestion, preprocessing, indexing, phrase queries, tolerant retrieval, and the dictionary experiment.
- `dictionary_search.py` — BST and B-tree dictionaries and in-memory benchmark.
- `normalization_evaluation.py` and `evaluation_queries.json` — retrieval-quality comparison and its curated relevance judgments.
- `css/styles.css` — BITS color palette and Streamlit layout styling.
- `dataset/corpus.json` — 60-document teaching corpus.
- `report.md` — implementation explanation, experimental results, inferences, and limitations.
- `assets/bits_logo.png` — BITS branding extracted from the supplied assignment brief.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet averaged_perceptron_tagger_eng
streamlit run app.py
```

The sidebar lets you select either Stemming or Lemmatization; only one normalization method is applied at a time.

At startup, the app loads every supported file in `dataset/`, including the supplied 60-document `corpus.json`. To include PDFs in the default collection, place them in `dataset/` beside `corpus.json`; each PDF is indexed as one document using its extracted text. Scanned/image-only PDFs need OCR before they can be searched; the app displays a warning when a PDF has no extractable text. JSON, TXT, Markdown, CSV, XLSX, and DOCX files in that folder are also loaded. Document IDs must be unique.

Alternatively, use the sidebar uploader to replace the current collection for the session. All required actions are available from the Streamlit interface; no backend code is needed during a demonstration.

## Demo checklist

1. Enter a query in the centered home-page search box and select a retrieval method in the sidebar.
2. Browse uploaded document contents on **Documents**.
3. Show the document count and collection table on **Overview**.
4. Select a document on **Preprocessing & Index** and show tokenization, lowercasing, hyphen handling, stop-word removal, the selected stemming/lemmatization result, and the inverse index. Run **Compare normalization methods** to see per-query precision, recall and F1 and macro averages on the judged teaching corpus; edit the JSON judgments to evaluate the current collection instead.
5. Search a phrase on **Phrase Queries** to compare **Biword** and **Positional** results side by side, inspect their index samples, and see any biword false positives. Phrase adjacency is evaluated after preprocessing; turn off stop-word removal to retain stop words.
6. Try `retrievel algorihm` on **Tolerant Retrieval**.
7. Run the **BST vs B-tree** benchmark with one term per line and capture its lookup and retrieval timing table.
8. Use **Inferences** as the discussion slide.

## Reproducibility

The corpus includes short, original information-retrieval teaching notes. Timings are measured in the Python process, so values vary by machine. The report records an example experiment and its protocol. To rerun that experiment with 10,000 repetitions per measurement:

```bash
python - <<'PY'
import app
from dictionary_search import run_benchmark

queries = ["retrieval", "index", "streamlit", "phrase", "ranking", "nonexistentterm"]
normalized = [(query, app.process(query)[0]) for query in queries]
print("Documents:", len(app.docs), "Vocabulary:", len(app.inverted))
for row in run_benchmark(app.inverted, app.docs, normalized, repetitions=10000):
    print(row)
PY
```

The command uses the default dataset and preprocessing settings. Standalone import of the Streamlit app may emit bare-mode warnings. Lookup times exclude tree construction; retrieval includes lookup and fetching already-loaded documents, not disk I/O.

For the normalization comparison, select the judged teaching corpus in **Preprocessing & Index**, enable stop-word removal, select split hyphens, and run the comparison. The judgments and information needs are in `evaluation_queries.json`; documents not listed as relevant are treated as non-relevant. For another collection, select **Current collection** and provide complete relevance judgments using that collection's document IDs. Lemmatization requires the NLTK resources installed in the setup commands above.
