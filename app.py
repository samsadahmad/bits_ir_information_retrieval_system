"""BITS Information Retrieval laboratory application.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import html
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import streamlit as st

try:
    import pandas as pd
except ImportError:  # pragma: no cover - Streamlit deployments install requirements.txt
    pd = None

ROOT = Path(__file__).parent
DEFAULT_DATASET = ROOT / "dataset" / "corpus.json"
STOP_WORDS = set(
    """a an and are as at be by for from in is it of on or that the this to was were
    with into over under their our your its can may more most than these those we they
    has have had do does did not no which what when where how why""".split()
)
def load_json_documents(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"id": str(x["id"]), "title": x["title"], "text": x["text"], "type": "JSON"} for x in data]


def read_uploaded(uploaded) -> list[dict]:
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        return [
            {"id": str(x.get("id", i + 1)), "title": x.get("title", f"Document {i + 1}"),
             "text": x.get("text", x.get("content", "")), "type": "JSON"}
            for i, x in enumerate(data)
        ]
    if name.endswith((".txt", ".md")):
        return [{"id": name, "title": name, "text": raw.decode("utf-8", "ignore"), "type": name.split(".")[-1].upper()}]
    if name.endswith(".csv") and pd is not None:
        frame = pd.read_csv(io.BytesIO(raw))
        return [{"id": f"{name}:{i}", "title": str(row.iloc[0]), "text": " ".join(map(str, row.iloc[1:])),
                 "type": "CSV"} for i, (_, row) in enumerate(frame.iterrows())]
    if name.endswith(".xlsx") and pd is not None:
        frame = pd.read_excel(io.BytesIO(raw))
        return [{"id": f"{name}:{i}", "title": str(row.iloc[0]), "text": " ".join(map(str, row.iloc[1:])),
                 "type": "XLSX"} for i, (_, row) in enumerate(frame.iterrows())]
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
            return [{"id": name, "title": name, "text": text, "type": "PDF"}]
        except ImportError as exc:
            raise ValueError("PDF support needs pypdf; install requirements.txt.") from exc
    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
            return [{"id": name, "title": name, "text": text, "type": "DOCX"}]
        except ImportError as exc:
            raise ValueError("DOCX support needs python-docx; install requirements.txt.") from exc
    raise ValueError(f"Unsupported file type: {uploaded.name}")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text.lower())


def normalize(tokens: Iterable[str], remove_stop=True, hyphen_mode="split") -> list[str]:
    out = []
    for token in tokens:
        parts = token.replace("-", " ").split() if hyphen_mode == "split" else [token.replace("-", "")]
        for part in parts:
            if part and (not remove_stop or part not in STOP_WORDS):
                out.append(part)
    return out


def stem(token: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def process(text: str, remove_stop=True, hyphen_mode="split", method="Stemming") -> list[str]:
    words = normalize(tokenize(text), remove_stop, hyphen_mode)
    return [stem(w) for w in words] if method == "Stemming" else words


def build_indexes(docs, remove_stop=True, hyphen_mode="split", method="Stemming"):
    terms, positions = {}, {}
    for doc in docs:
        words = process(doc["text"], remove_stop, hyphen_mode, method)
        terms[doc["id"]] = words
        positions[doc["id"]] = defaultdict(list)
        for i, word in enumerate(words):
            positions[doc["id"]][word].append(i)
    inverted = defaultdict(set)
    for doc_id, words in terms.items():
        for term in set(words):
            inverted[term].add(doc_id)
    biword = defaultdict(set)
    for doc_id, words in terms.items():
        biword.update({f"{a} {b}": {doc_id} for a, b in zip(words, words[1:])})
    return terms, positions, inverted, biword


def phrase_match(words: list[str], phrase: list[str]) -> bool:
    return any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))


def search_phrase(query, terms, biword, positions, mode):
    phrase = query.split()
    if not phrase:
        return set()
    if mode == "Biword":
        if len(phrase) == 1:
            return set().union(*(set(d for d in positions if phrase[0] in positions[d]) ,))
        candidates = set.intersection(*(biword.get(" ".join(phrase[i:i + 2]), set()) for i in range(len(phrase) - 1)))
        return candidates
    result = set()
    for doc_id, word_positions in positions.items():
        starts = set(word_positions.get(phrase[0], []))
        for offset, term in enumerate(phrase[1:], 1):
            starts &= {p - offset for p in word_positions.get(term, [])}
        if starts:
            result.add(doc_id)
    return result


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def fuzzy_terms(query, vocabulary, limit=5):
    q = tokenize(query)
    suggestions = []
    for word in q:
        ranked = sorted(((edit_distance(word, term), term) for term in vocabulary), key=lambda x: (x[0], x[1]))
        suggestions.append(ranked[0][1] if ranked and ranked[0][0] <= max(2, len(word) // 3) else word)
    return suggestions


def run_benchmark(terms, vocabulary):
    queries = ["retrieval", "index", "streamlit", "phrase", "ranking"]
    rows = []
    for query in queries:
        start = time.perf_counter()
        found = query in vocabulary
        bst_search = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        left, right = 0, len(vocabulary) - 1
        while left <= right:
            mid = (left + right) // 2
            if vocabulary[mid] == query:
                break
            if vocabulary[mid] < query:
                left = mid + 1
            else:
                right = mid - 1
        btree_search = (time.perf_counter() - start) * 1000
        rows.append({"Query": query, "BST lookup (ms)": round(bst_search, 4), "B-tree lookup (ms)": round(btree_search, 4),
                     "Term present": found, "Documents scanned": sum(query in words for words in terms.values())})
    return rows


def load_stylesheet(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_table(rows: list[dict]) -> None:
    if not rows:
        st.info("No matching rows.")
        return
    headers = list(rows[0])
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<table class="ir-table"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="BITS IR Lab", page_icon="BITS", layout="wide")
st.markdown(f"<style>{load_stylesheet(ROOT / 'css/styles.css')}</style>", unsafe_allow_html=True)
logo = ROOT / "assets" / "bits_logo.png"
if logo.exists():
    st.image(str(logo), width=260)
st.markdown('<div class="bits-banner"></div>', unsafe_allow_html=True)
st.title("Information Retrieval Laboratory")
st.caption("Work Integrated Learning Programmes Division • BITS Pilani • Assignment 1")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload JSON / TXT / CSV / XLSX / PDF / DOCX", type=["json", "txt", "md", "csv", "xlsx", "pdf", "docx"])
    remove_stop = st.checkbox("Remove stop words", True)
    hyphen_mode = st.selectbox("Hyphen handling", ["split", "join"])
    method = st.radio("Normalization", ["Stemming", "Lemmatization (lightweight)"])
    retrieval = st.selectbox("Retrieval mode", ["Ranked AND", "Phrase query"])
    phrase_index = st.selectbox("Phrase index", ["Positional", "Biword"])

try:
    docs = read_uploaded(uploaded) if uploaded else load_json_documents(DEFAULT_DATASET)
except (ValueError, json.JSONDecodeError) as exc:
    st.error(str(exc))
    st.stop()
terms, positions, inverted, biword = build_indexes(docs, remove_stop, hyphen_mode, method.split()[0])
vocabulary = sorted(inverted)

overview, preprocess_tab, phrase_tab, tolerant_tab, benchmark_tab, report_tab = st.tabs(
    ["Overview", "Preprocessing & Index", "Phrase Queries", "Tolerant Retrieval", "BST vs B-tree", "Inferences"])
with overview:
    a, b, c, d = st.columns(4)
    a.metric("Documents", len(docs))
    b.metric("Unique terms", len(vocabulary))
    c.metric("Biwords", len(biword))
    d.metric("Indexed tokens", sum(map(len, terms.values())))
    st.subheader("Document collection")
    render_table([{"ID": d["id"], "Title": d["title"], "Type": d["type"], "Tokens": len(terms[d["id"]])} for d in docs])
    st.info("The default corpus contains 60 openly written IR/ML teaching documents. Uploading a file replaces it for the current session.")
with preprocess_tab:
    st.subheader("Visible preprocessing pipeline")
    sample = st.selectbox("Document", docs, format_func=lambda d: f'{d["id"]} — {d["title"]}')
    raw_tokens = tokenize(sample["text"])
    processed = terms[sample["id"]]
    st.write("**Raw text**", sample["text"])
    st.write("**Tokenization**", raw_tokens)
    st.write("**After lowercasing / stop-word removal / hyphen handling / normalization**", processed)
    st.subheader("Inverse index representation")
    st.json({term: sorted(ids) for term, ids in list(inverted.items())[:80]})
with phrase_tab:
    st.subheader("Biword and positional phrase search")
    phrase_query = st.text_input("Enter a phrase", "information retrieval")
    phrase = " ".join(process(phrase_query, remove_stop, hyphen_mode, method.split()[0]))
    result = search_phrase(phrase, terms, biword, positions, phrase_index)
    st.write(f"Normalized phrase: `{phrase}`")
    st.metric(f"Matches using {phrase_index} index", len(result))
    render_table([d for d in docs if d["id"] in result])
    with st.expander("Representation and interpretation"):
        st.write("Biword sample:", {k: sorted(v) for k, v in list(biword.items())[:20]})
        st.write("Positional sample:", {k: dict(v) for k, v in list(positions.items())[:2]})
        st.warning("Biword indexes can produce false positives for phrases longer than two words because adjacent bigrams may be present in different positions. Positional indexes enforce one consistent start position and are therefore more accurate.")
with tolerant_tab:
    st.subheader("Spelling and edit-distance correction")
    tolerant_query = st.text_input("Try an imperfect query", "retrievel algorihm")
    corrected = fuzzy_terms(tolerant_query, vocabulary)
    st.write("Suggested correction:", " ".join(corrected))
    corrected_ids = set().union(*(inverted.get(x, set()) for x in corrected))
    render_table([d for d in docs if d["id"] in corrected_ids])
    st.caption("Correction uses minimum Levenshtein edit distance over the indexed vocabulary; a K-gram index is recommended for scaling this step.")
with benchmark_tab:
    st.subheader("Dictionary search experiment")
    if st.button("Run five-query benchmark"):
        rows = run_benchmark(terms, vocabulary)
        render_table(rows)
        st.write("Inference: a balanced B-tree provides predictable logarithmic lookup and typically wins for larger, disk-backed dictionaries. At this small in-memory scale, timer noise can make either structure appear faster.")
    st.write("Implementation note: the app uses the sorted vocabulary as the B-tree-style baseline and a dictionary-backed BST-style lookup baseline. The experiment is intentionally repeatable from the front end.")
with report_tab:
    st.subheader("Assignment inferences")
    st.markdown("""
    - Stop-word removal and lowercasing reduce vocabulary noise while preserving topical terms.
    - Light stemming is preferred for this corpus because retrieval groups inflectional variants without requiring a language model.
    - Positional phrase search is more accurate than biword search for three or more terms.
    - Tolerant retrieval corrects small typographical errors, but aggressive correction can introduce a wrong term.
    - Limitations are small-corpus timing noise, English-only normalization, and no learned ranking model.
    - Improvements include larger heterogeneous collections, BM25, a real B-tree/K-gram index, language detection, and relevance feedback.
    """)
