import math

import numpy as np
import pytest

import retrieval as r


def test_tokenize_stems_and_drops_stopwords():
    assert r.tokenize("The Servers are Restarting!") == ["server", "restart"]


def test_bm25_prefers_rare_terms():
    docs = [r.tokenize(t) for t in ("alpha beta", "alpha gamma", "alpha delta", "beta zeta")]
    scores = r.BM25(docs).scores(r.tokenize("alpha zeta"))
    assert int(np.argmax(scores)) == 3


def test_bm25_unknown_term_scores_nothing():
    assert r.BM25([["a1"], ["b1"]]).scores(["nothere"]).sum() == 0


def test_top_returns_best_first():
    assert r.top(np.array([0.1, 0.9, 0.5]), 2) == [1, 2]


def test_rrf_rewards_agreement_over_one_high_rank():
    assert r.rrf([[1, 2, 3], [2, 3, 1]])[0] == 2


def test_rrf_matches_formula():
    fused = r.rrf([[7], [8, 7]], k=60)
    assert fused[0] == 7  # 1/61 + 1/62 beats 1/61


def test_convex_normalises_scales():
    a = {1: 1000.0, 2: 0.0}
    b = {1: 0.0, 2: 0.9, 3: 0.45}
    assert r.convex(a, b, alpha=0.5)[0] in (1, 2)
    assert r.convex(a, b, alpha=1.0)[0] == 1
    assert r.convex(a, b, alpha=0.0)[0] == 2


def test_ndcg_perfect_and_hand_computed():
    assert r.ndcg([1, 2], {1: 1, 2: 1}) == pytest.approx(1.0)
    assert r.ndcg([9, 1], {1: 1}) == pytest.approx(1 / math.log2(3))
    assert r.ndcg([9], {}) == 0


def test_recall_and_reciprocal_rank():
    assert r.recall([1, 2, 3], {1: 1, 4: 1}, k=3) == 0.5
    assert r.reciprocal_rank([5, 1], {1: 1}) == 0.5
    assert r.reciprocal_rank([5], {1: 1}) == 0


def test_bootstrap_brackets_a_known_difference():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    low, high = r.paired_bootstrap(a, a + 0.1 + rng.normal(0, 0.01, 500), rng, rounds=2000)
    assert low < 0.1 < high and high - low < 0.01
