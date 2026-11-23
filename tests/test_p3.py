import numpy as np
import pandas as pd

from lastfm_rec.graph import build_graph, column_stochastic
from lastfm_rec.models import P3
from lastfm_rec.split import split_per_user


def _fit(toy, **kw):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    return P3(**kw).fit(sp.train, n_users, n_items), sp


def test_hand_computed_tiny_graph():
    # 사용자 0: 아티스트 0(w1), 1(w1) / 사용자 1: 아티스트 1(w1), 2(w1)
    # 사용자 0 → 1/2씩 아티스트 0, 1 → 아티스트 0은 사용자 0으로만, 아티스트 1은 사용자 0·1 반반
    # 2걸음: 사용자 0 = 1/2 + 1/4 = 3/4, 사용자 1 = 1/4
    # 3걸음: 아티스트 0 = 3/8, 1 = 3/8 + 1/8 = 1/2, 2 = 1/8
    df = pd.DataFrame({"u": [0, 0, 1, 1], "i": [0, 1, 1, 2], "w": [1.0, 1.0, 1.0, 1.0]})
    m = P3().fit(df, 2, 3)
    assert np.allclose(m.score(np.array([0]))[0], [3 / 8, 1 / 2, 1 / 8])


def test_alpha1_equals_three_step_walk(toy):
    # α = 1, β = 0이면 RWR과 같은 그래프에서 3걸음 걸은 분포의 아티스트 구간과 같다
    m, sp = _fit(toy)
    A = build_graph(sp.train, m.n_users, m.n_items)
    M = column_stochastic(A).toarray()
    users = np.arange(8)
    E = np.zeros((A.shape[0], len(users)))
    E[users, np.arange(len(users))] = 1.0
    walk = (M @ M @ M @ E)[m.n_users:m.n_users + m.n_items].T
    assert np.allclose(m.score(users), walk)


def test_score_shape_and_finite(toy):
    m, _ = _fit(toy, alpha=0.5, beta=0.5)
    sc = m.score(np.arange(5))
    assert sc.shape == (5, m.n_items) and np.isfinite(sc).all() and (sc >= 0).all()


def test_beta_lowers_recommended_popularity(toy):
    base, _ = _fit(toy, beta=0.0)
    corr, _ = _fit(toy, beta=1.0)
    users = np.arange(base.n_users)

    def mean_pop(model):
        rec = model.recommend(users, k=10)
        return np.mean([model.popularity[r].mean() for r in rec.values()])

    assert mean_pop(corr) < mean_pop(base)


def test_heldout_pairs_scored_as_unseen(toy):
    m, sp = _fit(toy)
    held = set(zip(sp.test["u"], sp.test["i"]))
    seen = set(zip(*m.seen.nonzero()))
    assert not held & seen
