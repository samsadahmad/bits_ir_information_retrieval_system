# Information Retrieval — Assignment 1 Report

**Programme:** Work Integrated Learning Programmes Division, BITS Pilani  
**Course:** Information Retrieval (AIMLCZG537/DSECLZG537)  
**Application:** `app.py` (Streamlit)

## 1. Objective and collection

This submission implements an end-to-end IR workflow in Streamlit. The included collection has 60 short, original documents about indexing, ranking, NLP, search evaluation, and tolerant retrieval. Each document has an identifier, title, and text. The application also accepts JSON, TXT, Markdown, CSV, XLSX, PDF, and DOCX uploads from the front end.

## 2. Workflow and preprocessing

The front end exposes document upload, document viewing, query input, preprocessing controls, retrieval selection, intermediate representations, and final results. The pipeline is:

1. Tokenize alphanumeric terms while retaining internal hyphens/apostrophes.
2. Lowercase every token.
3. Either split or join hyphenated words.
4. Remove a compact English stop-word list.
5. Apply light suffix stemming (or an uninflected lightweight alternative labelled lemmatization).
6. Build an inverted index, biword index, and positional index.

The inverted index is represented as `term -> set(document IDs)`. The positional index is `document ID -> term -> token positions`. These structures are displayed in the **Preprocessing & Index** tab.

## 3. Phrase queries

The **Phrase Queries** tab runs the same normalized phrase through two methods. A biword query intersects the posting lists of its adjacent pairs. A positional query intersects aligned positions, requiring the second term to occur one position after the first, the third two positions after the first, and so on.

**Inference:** positional search is the accurate method. Biword indexing is compact and fast, but for phrases of three or more words it can return a false positive when each adjacent pair exists but the pairs do not share one phrase start position. Positional indexes prevent this by enforcing alignment.

## 4. Dictionary experiment

The app runs five queries (`retrieval`, `index`, `streamlit`, `phrase`, `ranking`) and reports lookup time and document scans. It compares a dictionary-backed BST-style baseline against sorted-vocabulary binary search as the B-tree-style baseline.

| Measure | BST-style baseline | B-tree-style baseline |
|---|---:|---:|
| Lookup complexity | Average O(log n) | O(log n) |
| Storage orientation | In-memory nodes/hash baseline | Sorted, block-friendly keys |
| Expected scale behaviour | Good for memory | Better for disk/page-oriented dictionaries |
| Observed small-corpus result | Timer noise possible | Usually more predictable |

**Inference:** at 60 documents, absolute timings are too small for a meaningful winner because browser and timer overhead dominate. For a larger disk-backed term dictionary, a B-tree is the preferred structure because it reduces page reads and has predictable logarithmic lookup.

## 5. Tolerant retrieval

The app corrects each query token using minimum Levenshtein edit distance over the vocabulary, then unions postings for the corrected terms. The demonstration query `retrievel algorihm` is corrected to close vocabulary terms and returns relevant documents.

**Inference:** the model is tolerant of one or two character errors, but it can over-correct an unknown term. A production system should add a K-gram candidate filter, edit-distance thresholds, query context, and spelling confidence.

## 6. Conclusions

- Lowercasing and stop-word removal improve precision by reducing duplicate/noisy terms.
- Light stemming is more suitable than the lightweight lemmatization alternative for this small English corpus because it groups inflectional variants without an external model.
- Positional indexes give more accurate phrase results than biword indexes.
- The B-tree-style organization is expected to be faster and more scalable for large, disk-resident dictionaries; the app reports measured per-query values.
- Tolerant retrieval handles modest typographical errors but is not semantic retrieval.

## 7. Limitations and future work

The corpus is small and English-only; timing differences are therefore noisy. The ranker is a transparent Boolean/union demonstration rather than BM25 or a learned model. Future work should add heterogeneous real documents, BM25 with field weighting, MAP/MRR relevance judgements, a true balanced BST and B-tree implementation, K-gram indexing, multilingual processing, and relevance feedback.

## 8. Virtual-lab demonstration evidence

The Streamlit tabs are arranged as a screenshot checklist: Overview, Preprocessing & Index, Phrase Queries, Tolerant Retrieval, BST vs B-tree, and Inferences. Run the commands in `README.md`, capture each tab after executing the specified action, and include those screenshots in the Virtual Lab submission.
