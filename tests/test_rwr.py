import numpy as np
import pandas as pd

from lastfm_rec.models import RWR
from lastfm_rec.split import split_per_user


def _fit(toy, **kw):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    return RWR(**kw).fit(sp.train, n_users, n_items), sp


def test_matches_closed_form_on_tiny_graph():
    # 사용자 2명, 아티스트 3개: power iteration 결과가 r = c (I - (1-c) M)^-1 e_u 와 같아야 한다
    df = pd.DataFrame({"u": [0, 0, 1, 1], "i": [0, 1, 1, 2], "w": [1.0, 2.0, 1.0, 3.0]})
    m = RWR(restart=0.3, tol=1e-12, max_iter=1000).fit(df, 2, 3)
    R = m._propagate(np.array([0, 1]))
    M = m.M.toarray()
    exact = 0.3 * np.linalg.solve(np.eye(m.n_nodes) - 0.7 * M, np.eye(m.n_nodes)[:, [0, 1]])
    assert np.allclose(R, exact, atol=1e-9)


def test_score_shape_and_mass(toy):
    m, _ = _fit(toy)
    users = np.arange(7)
    sc = m.score(users)
    assert sc.shape == (7, m.n_items) and np.isfinite(sc).all() and (sc >= 0).all()
    # 학습 간선이 있는 사용자는 확률이 새지 않아 전체 노드 점수 합이 1
    R = m._propagate(users)
    assert np.allclose(R.sum(axis=0), 1.0, atol=1e-5)
    assert m.n_iter_[-1] <= m.max_iter


def test_popularity_from_train_only(toy):
    m, sp = _fit(toy)
    expected = np.bincount(sp.train["i"], minlength=m.n_items)
    assert np.array_equal(m.popularity, expected)


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
