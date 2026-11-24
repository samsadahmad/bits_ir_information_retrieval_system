from pathlib import Path
import json

topics = [
    ("inverted index", "An inverted index maps each term to the documents containing that term and supports fast candidate generation."),
    ("tokenization", "Tokenization converts a document into terms; punctuation, hyphens, and numbers require explicit policy decisions."),
    ("stop words", "Stop word removal can reduce index size, but negation and task-specific words should not be removed blindly."),
    ("stemming", "Stemming uses simple suffix rules to group inflectional variants and is inexpensive for a teaching corpus."),
    ("lemmatization", "Lemmatization uses vocabulary and morphology to return a base form, often improving linguistic precision."),
    ("phrase queries", "Phrase queries require terms to occur in an exact order and adjacent positions."),
    ("positional index", "A positional index stores token positions and verifies phrase alignment without false adjacency matches."),
    ("biword index", "A biword index stores adjacent term pairs and is compact, though longer phrases need several pair checks."),
    ("BM25 ranking", "BM25 combines term frequency, document frequency, document length, and saturation to rank results."),
    ("precision recall", "Precision measures relevant returned documents while recall measures relevant documents found."),
    ("mean average precision", "Mean average precision aggregates precision at relevant ranks across a set of information needs."),
    ("edit distance", "Levenshtein edit distance counts insertions, deletions, and substitutions between two strings."),
    ("k gram index", "A k-gram index generates character fragments that efficiently filter vocabulary candidates for tolerant retrieval."),
    ("wildcard search", "Wildcard search can use a permuterm or k-gram index to resolve a pattern such as comput*."),
    ("spell correction", "Spelling correction proposes a vocabulary term when a query contains a small typographical error."),
    ("document frequency", "Document frequency is the number of documents containing a term and helps estimate term discrimination."),
    ("term frequency", "Term frequency counts occurrences in one document and contributes to relevance scoring."),
    ("tf idf", "TF-IDF increases weight for frequent terms in one document and decreases weight for terms common to the collection."),
    ("query expansion", "Query expansion adds related terms, synonyms, or spelling variants to improve recall."),
    ("relevance feedback", "Relevance feedback uses user judgements to adjust a query toward useful documents."),
    ("search evaluation", "Search evaluation needs a test collection, information needs, relevance judgements, and repeatable metrics."),
    ("crawling", "A crawler discovers documents, respects access policies, and supplies content to an indexing pipeline."),
    ("document parsing", "Document parsing extracts text and metadata from formats such as PDF, Word, spreadsheets, and plain text."),
    ("metadata fields", "Metadata fields such as title and author support fielded search and can receive separate ranking weights."),
    ("normalization", "Normalization makes equivalent surface forms comparable before terms enter the dictionary."),
    ("case folding", "Case folding merges uppercase and lowercase spellings, improving matching for ordinary English queries."),
    ("hyphen handling", "Hyphen handling may split, join, or preserve a compound, and the choice affects phrase and term recall."),
    ("caching", "Caching repeated query results reduces response latency but requires invalidation when an index changes."),
    ("streamlit interface", "A Streamlit interface exposes controls and intermediate outputs while keeping the demonstration reproducible."),
    ("information need", "An information need describes the user's task and is translated into a query for retrieval."),
]

docs = []
for i in range(60):
    topic, explanation = topics[i % len(topics)]
    variant = i // len(topics) + 1
    docs.append({"id": f"D{i+1:02d}", "title": f"{topic.title()} note {variant}", "text": f"{explanation} This laboratory example connects {topic} with information retrieval, indexing, and transparent experiment design."})
Path("dataset").mkdir(exist_ok=True)
Path("dataset/corpus.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")
