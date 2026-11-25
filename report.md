# Information Retrieval — Assignment 1 Report

**Programme:** Work Integrated Learning Programmes Division, BITS Pilani  
**Course:** Information Retrieval (AIMLCZG537/DSECLZG537)  
**Application:** `app.py` (Streamlit)

## 1. Objective and collection

This submission implements an end-to-end IR workflow in Streamlit. The collection includes 60 short, original documents about indexing, ranking, NLP, search evaluation, and tolerant retrieval, plus other supported files in the dataset folder. Each document has an identifier, title, and text. The application also accepts JSON, TXT, Markdown, CSV, XLSX, PDF, and DOCX uploads from the front end.

## 2. Workflow and preprocessing

The front end exposes document upload, document viewing, query input, preprocessing controls, retrieval selection, intermediate representations, and final results. The pipeline is:

1. Tokenize alphanumeric terms while retaining internal hyphens/apostrophes.
2. Lowercase every token.
3. Either split or join hyphenated words.
4. Remove a compact English stop-word list.
5. Apply suffix-rule stemming or POS-aware WordNet lemmatization, selected one at a time in the sidebar.
6. Build an inverted index, biword index, and positional index.

The **Preprocessing & Index** tab displays the tokenized output and the effect of each subsequent stage: lowercasing, the selected hyphen policy, stop-word removal, and the selected stemmer or lemmatizer. The inverted index is represented as `term -> set(document IDs)` and displayed in the same tab. The positional index is `document ID -> term -> token positions`.

### Stemming versus lemmatization experiment

The evaluation uses the 60 teaching documents in `dataset/corpus.json`, excluding the separately loaded PDF and TXT because they were not judged. For each of the seven queries in `evaluation_queries.json`, the listed IDs are the notes judged relevant to its stated information need; all other teaching documents are treated as non-relevant. For the topical queries, relevance means that the note's main topic matches the intent; the suffix-rules query seeks notes explaining suffix rules for stemming. These small, manually specified judgments are illustrative rather than an independent benchmark.

Both methods index the same documents and process each query with stop-word removal and split hyphens, then retrieve documents containing **all** normalized query terms, as in the application's ranked-AND candidate selection. Precision = relevant retrieved / retrieved (zero when no documents are retrieved); recall = relevant retrieved / relevant; F1 is their harmonic mean. Macro scores average the individual query scores, giving each query equal weight. The observed results using the installed WordNet and POS tagger are:

| Query | Relevant IDs | Stemming P / R / F1 | Lemmatization P / R / F1 |
|---|---|---|---|
| rank | D09, D39 | 0.333 / 1.000 / 0.500 | 0.333 / 1.000 / 0.500 |
| phrase queries | D06, D36 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| spelling correction | D15, D45 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| document frequencies | D16, D46 | 0.333 / 1.000 / 0.500 | 0.333 / 1.000 / 0.500 |
| positional indexes | D07, D37 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| stemming | D04, D34 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| use suffix rules | D04, D34 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 |
| **Macro average** | | **0.810 / 1.000 / 0.857** | **0.667 / 0.857 / 0.714** |

**Conclusion:** for this judged teaching subset and these queries, stemming is more suitable: it has higher macro F1 and recall, while both methods otherwise return the same results. For `use suffix rules`, the stemmer matches `use` and `uses`; in this context the lemmatizer outputs `us` for the document's `uses`, so it misses both relevant notes. This result is sensitive to POS tagging, queries and judgments, and does not establish that stemming is universally better. The two versions of each teaching note have the same text, and the evaluation is small; use additional independently judged documents and queries before generalizing.

The **Preprocessing & Index** tab runs the comparison and displays normalized queries, relevant IDs, retrieved IDs and per-query metrics. It also permits custom judgments for the current collection. The sidebar's stop-word and hyphen settings apply to both methods; changing them can change the results.

## 3. Phrase queries

The **Phrase Queries** tab displays sample representations and the results of both methods side by side for the same normalized phrase. The biword index maps each adjacent pair to the set of document IDs containing it. The positional index maps each document ID to terms and their zero-based token positions. A biword query intersects the posting lists of its adjacent pairs. A positional query intersects aligned positions, requiring the second term to occur one position after the first, the third two positions after the first, and so on. The tab also identifies documents returned only by the biword method.

**Example:** For the query `new york city`, the document `new york is big york city` contains both `new york` and `york city`, so the biword method returns it. The pairs are not aligned to form the complete phrase, so the positional method rejects it.

**Inference:** positional search accurately checks adjacency in the normalized token stream. For phrases of three or more words, biword search can return a false positive when each adjacent pair exists but the pairs do not share one phrase start position. Positional indexes prevent this by enforcing alignment. Both methods operate after preprocessing: stop-word removal can make originally separated terms adjacent. Disable stop-word removal to retain those terms; stemming or lemmatization and punctuation handling still affect matching.

## 4. Dictionary experiment

The inverted index supplies the term dictionary and document-ID postings. Two actual in-memory structures store the same postings: a binary search tree (BST) and a B-tree with minimum degree 8. Both receive terms in the same median-first insertion order; this initially balances the BST, which does not self-balance on later insertions. Queries use the same preprocessing as the index.

The following experiment used the default dataset as present on 25 November 2026: 62 documents (60 JSON documents, one TXT and one PDF) and 3,912 unique normalized terms, with stop-word removal, split hyphens and stemming. It ran on macOS ARM64, Python 3.13.3. Each value is the mean of 10,000 repetitions in milliseconds; construction and preprocessing are excluded. **Lookup** finds the term and its postings. **Retrieval** includes lookup plus collecting the corresponding already-loaded documents by ID; it does not measure disk access or result rendering. The final query tests a missing term.

| Query | Normalized term | Matches | BST lookup (ms) | B-tree lookup (ms) | BST retrieval (ms) | B-tree retrieval (ms) |
|---|---|---:|---:|---:|---:|---:|
| retrieval | retrieval | 60 | 0.000743 | 0.000615 | 0.002166 | 0.002579 |
| index | index | 62 | 0.000934 | 0.000332 | 0.001626 | 0.002111 |
| streamlit | streamlit | 2 | 0.000566 | 0.001152 | 0.001018 | 0.001010 |
| phrase | phrase | 6 | 0.000389 | 0.000519 | 0.000561 | 0.001342 |
| ranking | rank | 7 | 0.001240 | 0.000605 | 0.000992 | 0.000632 |
| nonexistentterm | nonexistentterm | 0 | 0.000374 | 0.000428 | 0.000782 | 0.000932 |

**Inference:** in this run, each tree had the lower lookup time for three queries; BST retrieval was lower for four and B-tree retrieval for two. The high-match queries generally required more time to collect documents than low-match queries. These very small in-memory timings are noisy and do not establish a general winner or demonstrate disk performance. B-trees can reduce page accesses when implemented with disk-backed nodes, but this experiment does not measure that benefit.

The app's **BST vs B-tree** tab accepts one query per line and repeats each measurement 1,000 times. Its values will vary between runs and may differ from the table above. The exact 10,000-repetition report experiment can be rerun using the command in the README.

## 5. Tolerant retrieval

The app corrects each query token using minimum Levenshtein edit distance over the vocabulary, then unions postings for the corrected terms. The demonstration query `retrievel algorihm` is corrected to close vocabulary terms and returns relevant documents.

**Inference:** the model is tolerant of one or two character errors, but it can over-correct an unknown term. A production system should add a K-gram candidate filter, edit-distance thresholds, query context, and spelling confidence.

## 6. Conclusions

- Lowercasing and stop-word removal improve precision by reducing duplicate/noisy terms.
- Stemming produced higher macro F1 than POS-aware WordNet lemmatization on the seven judged teaching-corpus queries; this is a limited, dataset-specific result.
- Positional indexes give more accurate phrase results than biword indexes.
- The BST and B-tree comparison reports measured lookup and in-memory document retrieval times; neither is consistently faster in the recorded run.
- Tolerant retrieval handles modest typographical errors but is not semantic retrieval.

## 7. Limitations and future work

The corpus is small and English-only; timing differences are therefore noisy. The trees are in-memory and the BST is not self-balancing. The ranker is a transparent Boolean/union demonstration rather than BM25 or a learned model. Future work should add heterogeneous real documents, BM25 with field weighting, MAP/MRR relevance judgements, disk-backed indexing, K-gram indexing, multilingual processing, and relevance feedback.

## 8. Virtual-lab demonstration evidence

The Streamlit tabs are arranged as a screenshot checklist: Overview, Preprocessing & Index, Phrase Queries, Tolerant Retrieval, BST vs B-tree, and Inferences. Run the commands in `README.md`, capture each tab after executing the specified action, and include those screenshots in the Virtual Lab submission.
