import numpy as np

from lastfm_rec.models import MF
from lastfm_rec.split import split_per_user


def _fit(toy, seed=0, **kw):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    params = dict(n_factors=8, epochs=10, seed=seed)
    params.update(kw)
    return MF(**params).fit(sp.train, n_users, n_items), sp


def test_score_shape_and_finite(toy):
    m, _ = _fit(toy)
    sc = m.score(np.arange(5))
    assert sc.shape == (5, m.n_items) and np.isfinite(sc).all()


def test_same_seed_same_model(toy):
    a, _ = _fit(toy, seed=1)
    b, _ = _fit(toy, seed=1)
    c, _ = _fit(toy, seed=2)
    users = np.arange(10)
    assert np.array_equal(a.score(users), b.score(users))
    assert not np.allclose(a.score(users), c.score(users))


def test_negatives_never_training_pairs(toy):
    m, sp = _fit(toy, epochs=1)
    rng = np.random.default_rng(0)
    u_neg, i_neg = m._sample_negatives(sp.train["u"].to_numpy(dtype=np.int64), rng)
    assert len(u_neg) == m.neg_ratio * len(sp.train)
    train = set(zip(sp.train["u"], sp.train["i"]))
    assert not train & set(zip(u_neg, i_neg))


def test_training_reduces_error_and_ranks_observed_higher(toy):
    m, sp = _fit(toy, epochs=20)
    assert m.rmse_history[-1] < m.rmse_history[0]
    sc = m.score(np.arange(m.n_users))
    seen = m.seen.toarray().astype(bool)
    assert sc[seen].mean() > sc[~seen].mean()


def test_heldout_pairs_scored_as_unseen(toy):
    # validation/test 쌍은 학습에 쓰지 않았으므로 추천 후보에 남는다
    m, sp = _fit(toy)
    held = set(zip(sp.test["u"], sp.test["i"]))
    seen = set(zip(*m.seen.nonzero()))
    assert not held & seen
