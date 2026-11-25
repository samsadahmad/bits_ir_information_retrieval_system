"""Evaluate normalized Boolean retrieval against explicit relevance judgments."""

from collections import defaultdict
from statistics import mean


def evaluate_normalizations(docs, judgments, process):
    """Return per-query precision, recall and F1 for both normalization methods."""
    doc_ids = {doc["id"] for doc in docs}
    if not docs or len(doc_ids) != len(docs):
        raise ValueError("The evaluation collection must contain unique document IDs.")
    if not isinstance(judgments, list) or not judgments:
        raise ValueError("Provide a non-empty JSON list of query judgments.")
    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError("Each judgment must be a JSON object.")
        query = judgment.get("query")
        relevant = judgment.get("relevant_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each judgment needs a non-empty query.")
        if not isinstance(relevant, list) or not relevant or not all(isinstance(item, str) for item in relevant):
            raise ValueError(f"Query {query!r} needs a non-empty list of relevant document IDs.")
        unknown = set(relevant) - doc_ids
        if unknown:
            raise ValueError(f"Query {query!r} references unknown document IDs: {', '.join(sorted(unknown))}.")

    rows = []
    for method in ("Stemming", "Lemmatization"):
        normalized_queries = []
        for judgment in judgments:
            words = process(judgment["query"], method)
            if not words:
                raise ValueError(f"Query {judgment['query']!r} has no searchable terms after {method.lower()}.")
            normalized_queries.append(words)
        inverted = defaultdict(set)
        for doc in docs:
            for term in set(process(doc["text"], method)):
                inverted[term].add(doc["id"])
        for judgment, words in zip(judgments, normalized_queries):
            retrieved = set.intersection(*(inverted.get(word, set()) for word in set(words)))
            relevant = set(judgment["relevant_ids"])
            true_positives = len(retrieved & relevant)
            precision = true_positives / len(retrieved) if retrieved else 0.0
            recall = true_positives / len(relevant)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({
                "Query": judgment["query"],
                "Method": method,
                "Normalized query": " ".join(words),
                "Relevant IDs": ", ".join(sorted(relevant)),
                "Retrieved IDs": ", ".join(sorted(retrieved)) or "(none)",
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            })
    summary = []
    for method in ("Stemming", "Lemmatization"):
        method_rows = [row for row in rows if row["Method"] == method]
        summary.append({
            "Method": method,
            "Macro precision": mean(row["Precision"] for row in method_rows),
            "Macro recall": mean(row["Recall"] for row in method_rows),
            "Macro F1": mean(row["F1"] for row in method_rows),
        })
    return rows, summary
