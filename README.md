# hybrid-retrieval-bench

BM25, dense retrieval, reciprocal rank fusion, convex score fusion and cross-encoder reranking compared on BeIR SciFact, with paired bootstrap confidence intervals and per-query win/loss counts.

Companion to [Hybrid Search Measured: Where Fusion and Reranking Actually Help](https://cloudandsre.com/blog/hybrid-search-rrf-reranking-measured/).

## Method

`retrieval.py` implements BM25 (k1 = 0.9, b = 0.4, Snowball stemming), reciprocal rank fusion with k = 60 as in [Cormack, Clarke and Büttcher](https://doi.org/10.1145/1571941.1572114), min-max convex fusion, nDCG@10, recall@100, MRR and a paired bootstrap over queries.

Dense retrieval uses `BAAI/bge-small-en-v1.5`; the reranker is `Xenova/ms-marco-MiniLM-L-6-v2`, applied to the top 20. Dataset: SciFact from BeIR. `bench.py --datasets scifact nfcorpus` adds NFCorpus if your machine can embed it in reasonable time.

## Results

| System | nDCG@10 | Recall@100 | MRR |
|---|---:|---:|---:|
| BM25 | 0.680 | 0.922 | 0.652 |
| Dense (bge-small) | 0.720 | 0.953 | 0.688 |
| RRF fusion | 0.734 | 0.965 | 0.706 |
| Convex fusion | 0.743 | 0.962 | 0.717 |
| Dense + rerank | 0.702 | 0.953 | 0.672 |
| RRF + rerank | 0.706 | 0.965 | 0.676 |

| Change | ΔnDCG@10 | 95% CI | Queries better / worse / tied |
|---|---:|---|---:|
| Dense (bge-small) → RRF fusion | +0.013 | -0.010 to +0.037 | 57 / 38 / 205 |
| BM25 → RRF fusion | +0.053 | +0.031 to +0.076 | 82 / 21 / 197 |
| Dense (bge-small) → Dense + rerank | -0.018 | -0.044 to +0.008 | 44 / 55 / 201 |
| RRF fusion → RRF + rerank | -0.028 | -0.054 to -0.002 | 42 / 60 / 198 |
| Dense (bge-small) → RRF + rerank | -0.014 | -0.045 to +0.016 | 56 / 62 / 182 |
| Convex fusion → RRF fusion | -0.009 | -0.025 to +0.006 | 30 / 33 / 237 |

Per query: BM25 2.1 ms, dense 1.2 ms, reranking 1,436.6 ms on CPU. SciFact only: the NFCorpus run did not finish on the laptop used (see Limits). Full numbers in [`results/retrieval.json`](results/retrieval.json).

## Run it

```bash
pip install -r requirements-ci.txt
pytest -q
python bench.py --datasets scifact          # small, a few minutes
python bench.py --datasets scifact nfcorpus # adds NFCorpus
```

## Limits

One dataset of scientific abstracts, one small embedder, one small reranker trained on MS MARCO. Larger models shift the absolute numbers. BM25 parameters are not tuned per dataset.

## License

Apache-2.0
