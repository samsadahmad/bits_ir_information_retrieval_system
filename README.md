# BITS Information Retrieval — Assignment 1

## Deliverables

- `app.py` — Streamlit front end for ingestion, preprocessing, indexing, phrase queries, tolerant retrieval, and the dictionary experiment.
- `css/styles.css` — BITS color palette and Streamlit layout styling.
- `dataset/corpus.json` — 60-document teaching corpus.
- `report.md` — implementation explanation, experimental results, inferences, and limitations.
- `assets/bits_logo.png` — BITS branding extracted from the supplied assignment brief.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens with the supplied 60-document collection. Use the sidebar uploader to replace it with JSON, TXT, CSV, XLSX, PDF, or DOCX files. All required actions are available from the Streamlit interface; no backend code is needed during a demonstration.

## Demo checklist

1. Show the document count and collection table on **Overview**.
2. Select a document on **Preprocessing & Index** and show tokens, normalized tokens, and the inverse index.
3. Search `information retrieval` with both **Biword** and **Positional** modes.
4. Try `retrievel algorihm` on **Tolerant Retrieval**.
5. Run the five-query **BST vs B-tree** benchmark and capture its table.
6. Use **Inferences** as the discussion slide.

## Reproducibility

The corpus is generated from short, original information-retrieval teaching notes. Timings are measured in the browser session, so values vary by machine; the report records the qualitative result and the exact protocol.
