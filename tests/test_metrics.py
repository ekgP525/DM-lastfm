import numpy as np
import pandas as pd

from lastfm_rec.metrics import evaluate, head_items, paired_bootstrap, per_user_scores, unreachable_fraction


def test_recall_ndcg_hand_computed():
    recs = {0: np.array([5, 1, 7]), 1: np.array([9, 8, 3])}
    truth = {0: {1, 2}, 1: {4}}
    s = per_user_scores(recs, truth, k=3)
    assert np.isclose(s.loc[0, "recall"], 0.5)
    # 맞힌 것이 2위 → DCG = 1/log2(3), IDCG = 1 + 1/log2(3)
    assert np.isclose(s.loc[0, "ndcg"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert s.loc[1, "recall"] == 0 and s.loc[1, "ndcg"] == 0


def test_head_is_top_20_percent_of_interactions():
    pop = np.array([50, 30, 10, 5, 5])  # 합 100, 누적 20%는 첫 아티스트에서 넘음
    assert head_items(pop).tolist() == [True, False, False, False, False]


def test_unreachable_fraction():
    fit = pd.DataFrame({"u": [0, 1], "i": [0, 1], "w": [1.0, 1.0]})
    ev = pd.DataFrame({"u": [0, 1, 1, 0], "i": [1, 2, 3, 0], "w": 1.0})
    assert unreachable_fraction(fit, ev, n_items=4) == 0.5


def test_reachable_scope_drops_zero_listener_items():
    fit = pd.DataFrame({"u": [0, 1, 1], "i": [0, 1, 0], "w": 1.0})
    ev = pd.DataFrame({"u": [0, 0], "i": [1, 3], "w": 1.0})  # 3번은 학습 청취자 0명
    recs = {0: np.array([1, 2])}
    a, _ = evaluate(recs, ev, fit, n_items=4, k=2)
    r, _ = evaluate(recs, ev, fit, n_items=4, k=2, restrict_reachable=True)
    assert np.isclose(a["recall@2"], 0.5) and np.isclose(r["recall@2"], 1.0)


def test_bootstrap_detects_clear_difference():
    idx = pd.Index(range(300))
    a = pd.Series(np.full(300, 0.3), index=idx)
    b = pd.Series(np.random.default_rng(0).normal(0.1, 0.05, 300), index=idx)
    out = paired_bootstrap(a, b)
    assert out["significant"] and out["ci_low"] > 0
