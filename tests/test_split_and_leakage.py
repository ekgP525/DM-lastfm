import numpy as np
import pandas as pd

from lastfm_rec.graph import build_graph, column_stochastic
from lastfm_rec.metrics import item_popularity
from lastfm_rec.models import Popularity
from lastfm_rec.split import split_per_user


def _pairs(df):
    return set(zip(df["u"], df["i"]))


def test_split_disjoint_and_complete(toy):
    inter, _, _ = toy
    sp = split_per_user(inter, seed=0)
    tr, va, te = _pairs(sp.train), _pairs(sp.val), _pairs(sp.test)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == _pairs(inter)
    # 모든 사용자에게 train이 남는다
    assert set(sp.train["u"]) == set(inter["u"])


def test_split_reproducible(toy):
    inter, _, _ = toy
    a, b = split_per_user(inter, seed=3), split_per_user(inter, seed=3)
    assert _pairs(a.test) == _pairs(b.test)
    assert _pairs(a.test) != _pairs(split_per_user(inter, seed=4).test)


def test_graph_has_no_heldout_edges(toy):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    A = build_graph(sp.train, n_users, n_items)
    for u, i in _pairs(sp.test) | _pairs(sp.val):
        assert A[u, n_users + i] == 0 and A[n_users + i, u] == 0


def test_graph_tags_of_heldout_pairs_removed(toy):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    held = pd.concat([sp.val, sp.test])
    u, i = int(held.iloc[0]["u"]), int(held.iloc[0]["i"])
    tags = pd.DataFrame({"u": [u], "i": [i], "t": [0]})  # 테스트 쌍에만 달린 태그
    A = build_graph(sp.train, n_users, n_items, tags=tags, n_tags=1, heldout_pairs=held)
    assert A[n_users + i, n_users + n_items] == 0


def test_transition_is_column_stochastic(toy):
    inter, n_users, n_items = toy
    A = build_graph(inter, n_users, n_items)
    M = column_stochastic(A)
    s = np.asarray(M.sum(axis=0)).ravel()
    deg = np.asarray(A.sum(axis=0)).ravel()
    assert np.allclose(s[deg > 0], 1.0)


def test_popularity_uses_train_only(toy):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    m = Popularity().fit(sp.train, n_users, n_items)
    assert np.array_equal(m.pop, item_popularity(sp.train, n_items))
    assert not np.array_equal(m.pop, item_popularity(inter, n_items))


def test_recommend_excludes_seen(toy):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    m = Popularity().fit(sp.train, n_users, n_items)
    recs = m.recommend(np.arange(n_users), k=10)
    seen = sp.train.groupby("u")["i"].apply(set)
    for u, items in recs.items():
        assert len(items) == 10 and not (set(items) & seen[u])
