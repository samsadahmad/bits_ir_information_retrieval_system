"""BITS Information Retrieval laboratory application.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import streamlit as st

from dictionary_search import run_benchmark
from normalization_evaluation import evaluate_normalizations

try:
    import pandas as pd
except ImportError:  # pragma: no cover - Streamlit deployments install requirements.txt
    pd = None

ROOT = Path(__file__).parent
DEFAULT_DATASET = ROOT / "dataset"
SUPPORTED_DATASET_SUFFIXES = {".json", ".txt", ".md", ".csv", ".xlsx", ".pdf", ".docx"}
STOP_WORDS = set(
    """a an and are as at be by for from in is it of on or that the this to was were
    with into over under their our your its can may more most than these those we they
    has have had do does did not no which what when where how why""".split()
)

def read_document_file(name: str, raw: bytes) -> list[dict]:
    name = name.lower()
    suffix = Path(name).suffix
    if suffix == ".json":
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        return [
            {"id": str(x.get("id", i + 1)), "title": x.get("title", f"Document {i + 1}"),
             "text": x.get("text", x.get("content", "")), "type": "JSON"}
            for i, x in enumerate(data)
        ]
    if suffix in (".txt", ".md"):
        return [{"id": name, "title": name, "text": raw.decode("utf-8", "ignore"), "type": suffix[1:].upper()}]
    if suffix == ".csv" and pd is not None:
        frame = pd.read_csv(io.BytesIO(raw))
        return [{"id": f"{name}:{i}", "title": str(row.iloc[0]), "text": " ".join(map(str, row.iloc[1:])),
                 "type": "CSV"} for i, (_, row) in enumerate(frame.iterrows())]
    if suffix == ".xlsx" and pd is not None:
        frame = pd.read_excel(io.BytesIO(raw))
        return [{"id": f"{name}:{i}", "title": str(row.iloc[0]), "text": " ".join(map(str, row.iloc[1:])),
                 "type": "XLSX"} for i, (_, row) in enumerate(frame.iterrows())]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
            return [{"id": name, "title": name, "text": text, "type": "PDF"}]
        except ImportError as exc:
            raise ValueError("PDF support needs pypdf; install requirements.txt.") from exc
    if suffix == ".docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
            return [{"id": name, "title": name, "text": text, "type": "DOCX"}]
        except ImportError as exc:
            raise ValueError("DOCX support needs python-docx; install requirements.txt.") from exc
    raise ValueError(f"Unsupported file type: {name}")


def read_uploaded(uploaded) -> list[dict]:
    return read_document_file(uploaded.name, uploaded.getvalue())


def load_dataset(path: Path) -> list[dict]:
    if not path.is_dir():
        raise ValueError(f"Dataset folder not found: {path}")
    documents = []
    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES:
            documents.extend(read_document_file(file_path.name, file_path.read_bytes()))
    if not documents:
        raise ValueError(f"No supported documents found in {path}. Add JSON, TXT, MD, CSV, XLSX, PDF, or DOCX files.")
    document_ids = [document["id"] for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("The dataset contains duplicate document IDs. Make each document ID unique.")
    return documents


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text)


def lowercase_tokens(tokens: Iterable[str]) -> list[str]:
    return [token.lower() for token in tokens]


def handle_hyphens(tokens: Iterable[str], hyphen_mode: str) -> list[str]:
    out = []
    for token in tokens:
        parts = token.replace("-", " ").split() if hyphen_mode == "split" else [token.replace("-", "")]
        out.extend(parts)
    return out


def remove_stop_words(tokens: Iterable[str], remove_stop: bool) -> list[str]:
    return [token for token in tokens if token and (not remove_stop or token not in STOP_WORDS)]


def normalize(tokens: Iterable[str], remove_stop=True, hyphen_mode="split") -> list[str]:
    lowercased = lowercase_tokens(tokens)
    hyphen_handled = handle_hyphens(lowercased, hyphen_mode)
    return remove_stop_words(hyphen_handled, remove_stop)


def stem(token: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


class LemmatizationSetupError(RuntimeError):
    pass


def lemmatize(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    try:
        from nltk import pos_tag
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise LemmatizationSetupError(
            "Lemmatization needs NLTK in the Python environment running Streamlit. "
            "Activate that environment, then run `python -m pip install -r requirements.txt`."
        ) from exc

    try:
        tagged_tokens = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        pos_map = {
            "J": wordnet.ADJ,
            "V": wordnet.VERB,
            "N": wordnet.NOUN,
            "R": wordnet.ADV,
        }
        return [
            lemmatizer.lemmatize(token, pos_map.get(tag[0], wordnet.NOUN))
            for token, tag in tagged_tokens
        ]
    except LookupError as exc:
        raise LemmatizationSetupError(
            "Lemmatization data is missing. Run "
            "`python -m nltk.downloader wordnet averaged_perceptron_tagger_eng`."
        ) from exc


def process(text: str, remove_stop=True, hyphen_mode="split", method="Stemming") -> list[str]:
    words = normalize(tokenize(text), remove_stop, hyphen_mode)
    if method == "Stemming":
        return [stem(word) for word in words]
    if method == "Lemmatization":
        return lemmatize(words)
    return words


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
        for a, b in zip(words, words[1:]):
            biword[f"{a} {b}"].add(doc_id)
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
    q = lowercase_tokens(tokenize(query))
    suggestions = []
    for word in q:
        ranked = sorted(((edit_distance(word, term), term) for term in vocabulary), key=lambda x: (x[0], x[1]))
        suggestions.append(ranked[0][1] if ranked and ranked[0][0] <= max(2, len(word) // 3) else word)
    return suggestions


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


def render_document_preview(doc: dict, key: str) -> None:
    st.caption(f'{doc["type"]} · Document ID: {doc["id"]}')
    if doc["text"].strip():
        st.text_area("Extracted document text", doc["text"], height=420, disabled=True, key=key)
    else:
        st.info("No text could be extracted from this document.")


st.set_page_config(page_title="BITS IR Lab", page_icon="BITS", layout="wide")
st.markdown(f"<style>{load_stylesheet(ROOT / 'css/styles.css')}</style>", unsafe_allow_html=True)
logo = ROOT / "assets" / "bits_logo.png"
st.markdown('<div class="search-home-spacer"></div>', unsafe_allow_html=True)
_, brand_column, _ = st.columns([1, 2, 1])
with brand_column:
    if logo.exists():
        st.image(str(logo), width=210)
st.markdown('<div class="search-home-title">Information Retrieval</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="search-home-subtitle">Search the document collection</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Collection and options")
    uploaded = st.file_uploader("Upload JSON / TXT / CSV / XLSX / PDF / DOCX", type=["json", "txt", "md", "csv", "xlsx", "pdf", "docx"])
    st.subheader("Preprocessing")
    remove_stop = st.checkbox(
        "Remove stop words",
        True,
        help="Exclude common words such as 'the' and 'is' from the index and queries.",
    )
    hyphen_mode = st.selectbox(
        "Hyphen handling",
        ["split", "join"],
        format_func=lambda mode: "Split into separate terms" if mode == "split" else "Keep as one term",
        help="Choose whether a hyphenated word becomes multiple terms or stays joined.",
    )
    method = st.radio(
        "Word normalization",
        ["Stemming", "Lemmatization"],
        help="Choose one method: suffix-rule stemming or POS-aware WordNet lemmatization.",
    )
    st.subheader("Retrieval")
    retrieval = st.selectbox(
        "Retrieval method",
        ["Ranked AND", "Phrase query"],
        help="Ranked AND requires every query term. Phrase query requires the terms to occur next to each other in order.",
    )
    phrase_index = st.selectbox(
        "Phrase index",
        ["Positional", "Biword"],
        disabled=retrieval != "Phrase query",
        help="Select which phrase-search index to use when Phrase query is selected.",
    )
    st.caption("Options apply to the search and analysis tools.")

try:
    docs = read_uploaded(uploaded) if uploaded else load_dataset(DEFAULT_DATASET)
except (ValueError, json.JSONDecodeError) as exc:
    st.error(str(exc))
    st.stop()
pdfs_without_text = [
    doc for doc in docs
    if doc["type"] == "PDF" and not tokenize(doc["text"])
]
try:
    terms, positions, inverted, biword = build_indexes(docs, remove_stop, hyphen_mode, method)
except LemmatizationSetupError as exc:
    st.error(str(exc))
    st.stop()
vocabulary = sorted(inverted)

_, search_column, _ = st.columns([1, 8, 1])
with search_column:
    with st.form("home_search_form"):
        home_query = st.text_input(
            "Search documents",
            placeholder="Type a query and press Search",
            label_visibility="collapsed",
            key="home_search_input",
        )
        _, button_center, _ = st.columns([1, 1, 1])
        with button_center:
            search_submitted = st.form_submit_button("Search", use_container_width=True)
if search_submitted:
    st.session_state["last_search_query"] = home_query.strip()

last_search_query = st.session_state.get("last_search_query", "")
if last_search_query:
    query_terms = process(last_search_query, remove_stop, hyphen_mode, method)
    unique_terms = set(query_terms)
    if retrieval == "Ranked AND":
        matches = (
            set.intersection(*(inverted.get(term, set()) for term in unique_terms))
            if unique_terms else set()
        )
        scores = {
            doc_id: sum(terms[doc_id].count(term) for term in unique_terms)
            for doc_id in matches
        }
    else:
        matches = search_phrase(" ".join(query_terms), terms, biword, positions, phrase_index)
        scores = {doc_id: 1 for doc_id in matches}
    search_results = sorted(
        (doc for doc in docs if doc["id"] in matches),
        key=lambda doc: (-scores[doc["id"]], doc["title"]),
    )
    st.subheader(f"Results for: {last_search_query}")
    st.caption(f"{len(search_results)} matching documents · {retrieval}"
               + (f" · {phrase_index} index" if retrieval == "Phrase query" else ""))
    with st.expander("Search details"):
        st.write("Normalized query:", " ".join(query_terms))
        st.write(
            "Preprocessing:",
            f"stop-word removal {'on' if remove_stop else 'off'}, "
            f"hyphens {hyphen_mode}, {method.lower()}",
        )
    render_table([
        {
            "ID": doc["id"],
            "Title": doc["title"],
            "Score": scores[doc["id"]],
            "Type": doc["type"],
        }
        for doc in search_results
    ])
if pdfs_without_text:
    st.warning(
        f"{len(pdfs_without_text)} PDF(s) have no extractable text. "
        "They may be scanned or image-only; run OCR on them or use text-searchable PDFs."
    )

overview, documents_tab, preprocess_tab, phrase_tab, tolerant_tab, benchmark_tab, report_tab = st.tabs(
    ["Overview", "Browse Documents", "Preprocessing & Index", "Phrase Queries", "Tolerant Retrieval", "BST vs B-tree", "Inferences"])
with overview:
    a, b, c, d = st.columns(4)
    a.metric("Documents", len(docs))
    b.metric("Unique terms", len(vocabulary))
    c.metric("Biwords", len(biword))
    d.metric("Indexed tokens", sum(map(len, terms.values())))
    st.subheader("Document collection")
    render_table([{"ID": d["id"], "Title": d["title"], "Type": d["type"], "Tokens": len(terms[d["id"]])} for d in docs])
    st.info("The default collection includes 60 IR/ML teaching documents and other supported files in the dataset folder. Uploading a file replaces the collection for the current session.")
with documents_tab:
    st.subheader("Browse document contents")
    selected_doc = st.selectbox(
        "Choose a document to view",
        docs,
        format_func=lambda d: f'{d["id"]} — {d["title"]}',
    )
    render_document_preview(selected_doc, "document_preview")
with preprocess_tab:
    st.subheader("Visible preprocessing pipeline")
    sample = st.selectbox("Document", docs, format_func=lambda d: f'{d["id"]} — {d["title"]}')
    raw_tokens = tokenize(sample["text"])
    lowercased_tokens = lowercase_tokens(raw_tokens)
    hyphen_handled_tokens = handle_hyphens(lowercased_tokens, hyphen_mode)
    stop_filtered_tokens = remove_stop_words(hyphen_handled_tokens, remove_stop)
    processed = terms[sample["id"]]
    st.write("**Raw text**", sample["text"])
    st.write("**Tokenization**", raw_tokens)
    st.write("**Lowercasing**", lowercased_tokens)
    st.write(f"**Hyphen handling ({hyphen_mode})**", hyphen_handled_tokens)
    st.write(
        f"**Stop-word removal ({'on' if remove_stop else 'off'})**",
        stop_filtered_tokens,
    )
    st.write(f"**After {method.lower()}**", processed)
    st.subheader("Inverse index representation")
    st.json({term: sorted(ids) for term, ids in list(inverted.items())[:80]})
    st.subheader("Stemming vs lemmatization: retrieval quality")
    st.caption(
        "Both methods use the current stop-word and hyphen settings and ranked-AND "
        "matching. The evaluation treats every document not listed as relevant "
        "for a query as non-relevant. Supply complete judgments for the chosen collection."
    )
    curated_path = DEFAULT_DATASET / "corpus.json"
    curated_docs = read_document_file("corpus.json", curated_path.read_bytes()) if curated_path.is_file() else []
    current_by_id = {doc["id"]: doc for doc in docs}
    curated_available = bool(curated_docs) and all(
        doc["id"] in current_by_id
        and current_by_id[doc["id"]]["text"] == doc["text"]
        and current_by_id[doc["id"]]["title"] == doc["title"]
        for doc in curated_docs
    )
    evaluation_options = ["Current collection"]
    if curated_available:
        evaluation_options.insert(0, "Judged teaching corpus (60 documents)")
    evaluation_scope = st.selectbox("Evaluation collection", evaluation_options)
    if evaluation_scope == "Judged teaching corpus (60 documents)":
        evaluation_docs = [current_by_id[doc["id"]] for doc in curated_docs]
        initial_judgments = (ROOT / "evaluation_queries.json").read_text(encoding="utf-8")
        st.caption(
            "The supplied judgments identify notes whose main topic matches each intent. "
            "The PDF and TXT in the default collection are excluded because they are not judged."
        )
    else:
        evaluation_docs = docs
        initial_judgments = "[]"
        st.caption(
            "Enter judgments for the current collection as a JSON list. Example: "
            '[{"query": "example", "relevant_ids": ["document-id"]}]. '
            "Every relevant document ID must be listed."
        )
    with st.form("normalization_evaluation"):
        judgment_text = st.text_area(
            "Query relevance judgments (JSON)", initial_judgments, height=240,
            key=f"judgments_{evaluation_scope}",
        )
        compare_submitted = st.form_submit_button("Compare normalization methods")
    if compare_submitted:
        try:
            judgments = json.loads(judgment_text)
            comparison_rows, comparison_summary = evaluate_normalizations(
                evaluation_docs, judgments,
                lambda text, selected_method: process(text, remove_stop, hyphen_mode, selected_method),
            )
        except (json.JSONDecodeError, ValueError, LemmatizationSetupError) as exc:
            st.error(f"Cannot run comparison: {exc}")
        else:
            def format_metrics(rows):
                return [
                    {key: f"{value:.3f}" if isinstance(value, float) else value for key, value in row.items()}
                    for row in rows
                ]

            render_table(format_metrics(comparison_rows))
            st.write("**Macro averages across queries**")
            render_table(format_metrics(comparison_summary))
            stem_f1, lemma_f1 = (summary["Macro F1"] for summary in comparison_summary)
            if abs(stem_f1 - lemma_f1) < 1e-12:
                conclusion = "The methods tie on macro F1 for these judgments."
            else:
                winner = "Stemming" if stem_f1 > lemma_f1 else "Lemmatization"
                conclusion = f"{winner} has the higher macro F1 for these judgments."
            st.write(
                f"{conclusion} This conclusion applies only to the selected documents, "
                "queries, relevance judgments and preprocessing settings."
            )
with phrase_tab:
    st.subheader("Biword and positional phrase search")
    phrase_query = st.text_input("Enter a phrase", "information retrieval")
    phrase = " ".join(process(phrase_query, remove_stop, hyphen_mode, method))
    biword_result = search_phrase(phrase, terms, biword, positions, "Biword")
    positional_result = search_phrase(phrase, terms, biword, positions, "Positional")
    false_positives = biword_result - positional_result
    st.write(f"Normalized phrase: `{phrase}`")
    st.caption(
        "Both searches use normalized tokens. Stop-word removal can make words adjacent "
        "that were not adjacent in the original text; turn it off to retain stop words. "
        "Stemming or lemmatization and punctuation handling can also change what matches."
    )
    if not phrase:
        st.info("The phrase contains no searchable terms after preprocessing.")
    biword_column, positional_column = st.columns(2)
    with biword_column:
        st.metric("Matches using Biword index", len(biword_result))
        render_table([d for d in docs if d["id"] in biword_result])
    with positional_column:
        st.metric("Matches using Positional index", len(positional_result))
        render_table([d for d in docs if d["id"] in positional_result])
    st.write("Biword false positives for this query:", ", ".join(sorted(false_positives)) or "None")
    with st.expander("Representation and interpretation"):
        st.write("Biword sample:", {k: sorted(v) for k, v in list(biword.items())[:20]})
        st.write("Positional sample:", {k: dict(v) for k, v in list(positions.items())[:2]})
        st.warning("Biword indexes can produce false positives for phrases longer than two words because adjacent bigrams may be present in different positions. Positional indexes enforce one consistent start position and are therefore more accurate.")
        st.write(
            'Example: for the query "new york city", the document "new york is big york city" '
            'contains both biwords ("new york" and "york city") but not the complete phrase. '
            "The biword index returns it; the positional index does not."
        )
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
    st.caption(
        "Enter one single-term query per line. Lookup finds the term and its postings; "
        "retrieval includes lookup and fetching the matching documents from memory. "
        "Times are averages of 1,000 repetitions per query, excluding index construction."
    )
    benchmark_queries = st.text_area(
        "Benchmark queries", "retrieval\nindex\nstreamlit\nphrase\nranking",
        key="benchmark_queries",
    )
    if st.button("Run dictionary benchmark"):
        query_lines = [line.strip() for line in benchmark_queries.splitlines() if line.strip()]
        normalized = [(query, process(query, remove_stop, hyphen_mode, method)) for query in query_lines]
        invalid = [query for query, words in normalized if len(words) != 1]
        if not query_lines:
            st.error("Enter at least one query.")
        elif invalid:
            st.error("Each query must produce exactly one term after preprocessing: " + ", ".join(invalid))
        else:
            rows = run_benchmark(inverted, docs, [(query, words[0]) for query, words in normalized])
            render_table([
                {key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()}
                for row in rows
            ])
            bst_mean = sum(row["BST lookup (ms)"] for row in rows) / len(rows)
            btree_mean = sum(row["B-tree lookup (ms)"] for row in rows) / len(rows)
            st.write(
                f"Mean lookup over these queries: BST {bst_mean:.6f} ms; "
                f"B-tree {btree_mean:.6f} ms. These in-memory results depend on the "
                "queries, collection and machine; they do not measure disk performance."
            )
    st.write(
        "Both trees store the same term-to-document postings. The BST is built in "
        "median-first order; the B-tree has minimum degree 8."
    )
with report_tab:
    st.subheader("Assignment inferences")
    st.markdown("""
    - Stop-word removal and lowercasing reduce vocabulary noise while preserving topical terms.
    - Compare stemming and lemmatization against relevance judgments in the Preprocessing & Index tab before choosing a normalization method.
    - Positional phrase search is more accurate than biword search for three or more terms.
    - Tolerant retrieval corrects small typographical errors, but aggressive correction can introduce a wrong term.
    - Limitations are small-corpus timing noise, English-only normalization, and no learned ranking model.
    - Improvements include larger heterogeneous collections, BM25, a K-gram index, language detection, and relevance feedback.
    """)
