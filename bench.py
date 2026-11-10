#!/usr/bin/env python3
"""BM25, dense, hybrid and reranked retrieval on three BeIR datasets. Writes results/."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

import retrieval as r

BEIR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
EMBEDDER = "BAAI/bge-small-en-v1.5"
RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"
DEPTH = 100
RERANK_DEPTH = 20


def load(name: str):
    root = Path("data") / name
    if not root.exists():
        Path("data").mkdir(exist_ok=True)
        archive = Path("data") / f"{name}.zip"
        with urllib.request.urlopen(BEIR.format(name), timeout=300) as response:  # noqa: S310
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as z:
            z.extractall("data")
    ids, texts = [], []
    for line in (root / "corpus.jsonl").read_text().splitlines():
        doc = json.loads(line)
        ids.append(doc["_id"])
        texts.append(f"{doc.get('title', '')} {doc['text']}".strip())
    position = {d: i for i, d in enumerate(ids)}
    qrels: dict[str, dict[int, int]] = {}
    with (root / "qrels" / "test.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0 and row["corpus-id"] in position:
                qrels.setdefault(row["query-id"], {})[position[row["corpus-id"]]] = int(
                    row["score"]
                )
    queries = {}
    for line in (root / "queries.jsonl").read_text().splitlines():
        query = json.loads(line)
        if query["_id"] in qrels:
            queries[query["_id"]] = query["text"]
    return texts, queries, qrels


def rerank(encoder, query: str, ranking: list[int], texts: list[str]) -> list[int]:
    head = ranking[:RERANK_DEPTH]
    scores = list(encoder.rerank(query, [texts[d] for d in head], batch_size=32))
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [head[i] for i in order] + ranking[RERANK_DEPTH:]


def evaluate(dataset: str, embedder, encoder, rng) -> dict:
    texts, queries, qrels = load(dataset)
    query_ids = list(queries)
    bm25 = r.BM25([r.tokenize(t) for t in texts])
    cache = Path("data") / f"{dataset}-{EMBEDDER.split('/')[-1]}.npy"
    started = time.perf_counter()
    if cache.exists():
        corpus = np.load(cache)
    else:
        corpus = np.array(list(embedder.passage_embed(texts, batch_size=64)), dtype=np.float32)
        np.save(cache, corpus)
    embed_seconds = time.perf_counter() - started
    vectors = np.array(
        list(embedder.query_embed([queries[q] for q in query_ids])), dtype=np.float32
    )

    runs: dict[str, list[list[int]]] = {
        k: [] for k in ("bm25", "dense", "rrf", "convex", "dense+rerank", "rrf+rerank")
    }
    timing = {"bm25": 0.0, "dense": 0.0, "rerank": 0.0}
    for query_id, vector in zip(query_ids, vectors, strict=True):
        text = queries[query_id]
        started = time.perf_counter()
        lexical = bm25.scores(r.tokenize(text))
        lexical_rank = r.top(lexical, DEPTH)
        timing["bm25"] += time.perf_counter() - started
        started = time.perf_counter()
        semantic = corpus @ vector
        dense_rank = r.top(semantic, DEPTH)
        timing["dense"] += time.perf_counter() - started
        fused = r.rrf([lexical_rank, dense_rank])
        runs["bm25"].append(lexical_rank)
        runs["dense"].append(dense_rank)
        runs["rrf"].append(fused)
        runs["convex"].append(
            r.convex(
                {d: float(lexical[d]) for d in lexical_rank},
                {d: float(semantic[d]) for d in dense_rank},
            )
        )
        runs["dense+rerank"].append(rerank(encoder, text, dense_rank, texts))
        started = time.perf_counter()
        runs["rrf+rerank"].append(rerank(encoder, text, fused, texts))
        timing["rerank"] += time.perf_counter() - started

    per_query = {
        name: np.array([r.ndcg(rank, qrels[q]) for rank, q in zip(ranks, query_ids, strict=True)])
        for name, ranks in runs.items()
    }
    table = []
    for name, ranks in runs.items():
        table.append(
            {
                "system": name,
                "ndcg@10": round(float(per_query[name].mean()), 4),
                "recall@100": round(
                    float(
                        np.mean(
                            [r.recall(k, qrels[q]) for k, q in zip(ranks, query_ids, strict=True)]
                        )
                    ),
                    4,
                ),
                "mrr": round(
                    float(
                        np.mean(
                            [
                                r.reciprocal_rank(k, qrels[q])
                                for k, q in zip(ranks, query_ids, strict=True)
                            ]
                        )
                    ),
                    4,
                ),
            }
        )
    comparisons = []
    for base, candidate in (
        ("dense", "rrf"),
        ("bm25", "rrf"),
        ("dense", "dense+rerank"),
        ("rrf", "rrf+rerank"),
        ("dense", "rrf+rerank"),
        ("convex", "rrf"),
    ):
        a, b = per_query[base], per_query[candidate]
        low, high = r.paired_bootstrap(a, b, rng)
        comparisons.append(
            {
                "from": base,
                "to": candidate,
                "delta_ndcg@10": round(float((b - a).mean()), 4),
                "ci95": [round(low, 4), round(high, 4)],
                "queries_improved": int((b > a).sum()),
                "queries_worse": int((b < a).sum()),
                "queries_tied": int((b == a).sum()),
            }
        )
    n = len(query_ids)
    return {
        "dataset": dataset,
        "documents": len(texts),
        "queries": n,
        "embed_seconds": round(embed_seconds, 1),
        "ms_per_query": {k: round(v / n * 1000, 1) for k, v in timing.items()},
        "table": table,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact"])
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    embedder, encoder = TextEmbedding(EMBEDDER), TextCrossEncoder(RERANKER)
    rng = np.random.default_rng(args.seed)
    results = []
    for dataset in args.datasets:
        results.append(evaluate(dataset, embedder, encoder, rng))
        print(json.dumps(results[-1], indent=1), flush=True)
        Path("results").mkdir(exist_ok=True)
        meta = {"embedder": EMBEDDER, "reranker": RERANKER, "rerank_depth": RERANK_DEPTH}
        Path("results/retrieval.json").write_text(
            json.dumps({**meta, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
