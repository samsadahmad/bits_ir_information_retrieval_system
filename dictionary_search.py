"""Term dictionaries and repeatable in-memory lookup/retrieval benchmarks."""

from bisect import bisect_left
from dataclasses import dataclass, field
from time import perf_counter_ns


@dataclass
class BSTNode:
    key: str
    postings: tuple[str, ...]
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


class BinarySearchTree:
    def __init__(self):
        self.root: BSTNode | None = None

    def insert(self, key: str, postings: tuple[str, ...]) -> None:
        if self.root is None:
            self.root = BSTNode(key, postings)
            return
        node = self.root
        while True:
            if key == node.key:
                node.postings = postings
                return
            side = "left" if key < node.key else "right"
            child = getattr(node, side)
            if child is None:
                setattr(node, side, BSTNode(key, postings))
                return
            node = child

    def search(self, key: str) -> tuple[str, ...] | None:
        node = self.root
        while node is not None:
            if key == node.key:
                return node.postings
            node = node.left if key < node.key else node.right
        return None


@dataclass
class BTreeNode:
    keys: list[str] = field(default_factory=list)
    postings: list[tuple[str, ...]] = field(default_factory=list)
    children: list["BTreeNode"] = field(default_factory=list)


class BTree:
    def __init__(self, min_degree: int = 8):
        if min_degree < 2:
            raise ValueError("B-tree minimum degree must be at least 2.")
        self.min_degree = min_degree
        self.root = BTreeNode()

    def search(self, key: str) -> tuple[str, ...] | None:
        node = self.root
        while True:
            index = bisect_left(node.keys, key)
            if index < len(node.keys) and node.keys[index] == key:
                return node.postings[index]
            if not node.children:
                return None
            node = node.children[index]

    def insert(self, key: str, postings: tuple[str, ...]) -> None:
        root = self.root
        if len(root.keys) == 2 * self.min_degree - 1:
            self.root = BTreeNode(children=[root])
            self._split_child(self.root, 0)
        self._insert_nonfull(self.root, key, postings)

    def _split_child(self, parent: BTreeNode, index: int) -> None:
        degree = self.min_degree
        left = parent.children[index]
        right = BTreeNode(
            keys=left.keys[degree:],
            postings=left.postings[degree:],
            children=left.children[degree:] if left.children else [],
        )
        parent.keys.insert(index, left.keys[degree - 1])
        parent.postings.insert(index, left.postings[degree - 1])
        parent.children.insert(index + 1, right)
        left.keys = left.keys[:degree - 1]
        left.postings = left.postings[:degree - 1]
        if left.children:
            left.children = left.children[:degree]

    def _insert_nonfull(self, node: BTreeNode, key: str, postings: tuple[str, ...]) -> None:
        index = bisect_left(node.keys, key)
        if index < len(node.keys) and node.keys[index] == key:
            node.postings[index] = postings
            return
        if not node.children:
            node.keys.insert(index, key)
            node.postings.insert(index, postings)
            return
        if len(node.children[index].keys) == 2 * self.min_degree - 1:
            self._split_child(node, index)
            if key == node.keys[index]:
                node.postings[index] = postings
                return
            if key > node.keys[index]:
                index += 1
        self._insert_nonfull(node.children[index], key, postings)


def build_dictionaries(inverted):
    """Build both trees with the same median-first insertion order and postings."""
    entries = [(key, tuple(sorted(inverted[key]))) for key in sorted(inverted)]
    bst, btree = BinarySearchTree(), BTree()
    intervals = [(0, len(entries))]
    while intervals:
        start, end = intervals.pop()
        if start >= end:
            continue
        middle = (start + end) // 2
        key, postings = entries[middle]
        bst.insert(key, postings)
        btree.insert(key, postings)
        intervals.append((middle + 1, end))
        intervals.append((start, middle))
    return bst, btree


def retrieve(tree, query, documents_by_id):
    postings = tree.search(query)
    return [documents_by_id[doc_id] for doc_id in postings] if postings is not None else []


def run_benchmark(inverted, docs, queries, repetitions=1000):
    """Measure mean lookup and lookup-plus-document retrieval in milliseconds."""
    if repetitions < 1:
        raise ValueError("Benchmark repetitions must be positive.")
    bst, btree = build_dictionaries(inverted)
    documents_by_id = {doc["id"]: doc for doc in docs}
    rows = []
    for original, query in queries:
        bst_result = bst.search(query)
        btree_result = btree.search(query)
        if bst_result != btree_result:
            raise RuntimeError(f"Dictionary results differ for {query!r}.")
        row = {
            "Query": original,
            "Normalized term": query,
            "Term present": bst_result is not None,
            "Matching documents": len(bst_result) if bst_result is not None else 0,
        }
        for label, tree in (("BST", bst), ("B-tree", btree)):
            retrieve(tree, query, documents_by_id)
            start = perf_counter_ns()
            for _ in range(repetitions):
                tree.search(query)
            row[f"{label} lookup (ms)"] = (perf_counter_ns() - start) / repetitions / 1_000_000
            start = perf_counter_ns()
            for _ in range(repetitions):
                retrieve(tree, query, documents_by_id)
            row[f"{label} retrieval (ms)"] = (perf_counter_ns() - start) / repetitions / 1_000_000
        rows.append(row)
    return rows
