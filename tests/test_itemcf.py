import numpy as np
import pandas as pd
import pytest

from lastfm_rec.models import ItemCF
from lastfm_rec.split import split_per_user

K = 10


def _fit(toy, **kw):
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    return ItemCF(n_neighbors=K, **kw).fit(sp.train, n_users, n_items), sp


def _dense_cosine(X):
    """정답용: dense 행렬로 직접 계산한 아티스트끼리 코사인 (대각선 0)."""
    n = np.linalg.norm(X, axis=0)
    nn = np.outer(n, n)
    C = (X.T @ X) / np.where(nn > 0, nn, 1.0)
    np.fill_diagonal(C, 0.0)
    return C


def test_no_heldout_pairs_in_matrix(toy):
    # leakage: validation·test 쌍은 학습 행렬 X에 들어가지 않는다
    m, sp = _fit(toy)
    X = m.X.toarray()
    held = pd.concat([sp.val, sp.test])
    assert all(X[u, i] == 0 for u, i in zip(held["u"], held["i"]))


def test_similarity_uses_train_only(toy):
    # leakage: 유사도 S는 train만으로 계산한 값과 같고, 전체 데이터로 계산한 값과는 다르다
    inter, n_users, n_items = toy
    m, sp = _fit(toy)
    full = ItemCF(n_neighbors=K).fit(inter, n_users, n_items)
    train_only = ItemCF(n_neighbors=K).fit(sp.train.sample(frac=1.0, random_state=1), n_users, n_items)
    assert np.allclose(m.S.toarray(), train_only.S.toarray())
    assert not np.allclose(m.S.toarray(), full.S.toarray())


def test_heldout_changes_do_not_change_scores(toy):
    # leakage: val·test의 값을 바꿔도(학습에 안 쓰이니) 점수는 그대로다
    inter, n_users, n_items = toy
    sp = split_per_user(inter, seed=0)
    a = ItemCF(n_neighbors=K).fit(sp.fit_set("val"), n_users, n_items)
    sp.val["w"] = 100.0
    sp.test["w"] = 100.0
    b = ItemCF(n_neighbors=K).fit(sp.fit_set("val"), n_users, n_items)
    users = np.arange(n_users)
    assert np.array_equal(a.score(users), b.score(users))


def test_neighbors_at_most_k_no_self(toy):
    m, _ = _fit(toy)
    S = m.S.toarray()
    assert ((S != 0).sum(axis=1) <= K).all()
    assert np.all(np.diag(S) == 0)


def test_neighbors_are_top_k_cosine(toy):
    m, _ = _fit(toy)
    S, C = m.S.toarray(), _dense_cosine(m.X.toarray())
    kept = S != 0
    assert np.allclose(S[kept], C[kept])
    for r in range(S.shape[0]):
        if kept[r].any():
            dropped = C[r][~kept[r]]
            assert S[r][kept[r]].min() >= dropped.max() - 1e-12


@pytest.mark.parametrize("normalize", [False, True])
def test_score_matches_formula(toy, normalize):
    m, _ = _fit(toy, normalize=normalize)
    users = np.arange(10)
    sc = m.score(users)
    assert sc.shape == (len(users), m.n_items) and np.isfinite(sc).all()
    X, S = m.X.toarray(), m.S.toarray()
    for u in users:
        for i in range(m.n_items):
            js = (S[i] != 0) & (X[u] != 0)
            num, den = (S[i, js] * X[u, js]).sum(), S[i, js].sum()
            expected = (num / den if den > 0 else 0.0) if normalize else num
            assert sc[u, i] == pytest.approx(expected)


def test_recommend_excludes_seen(toy):
    m, sp = _fit(toy)
    recs = m.recommend(np.arange(m.n_users), k=10)
    seen = sp.train.groupby("u")["i"].apply(set)
    for u, items in recs.items():
        assert len(items) == 10 and not (set(items) & seen[u])


def test_neighbor_fn_all_pairs_equals_exact(toy):
    # 모든 쌍을 후보로 주면 정확한 Item-CF와 유사도가 같다 (이웃을 자르지 않는 k에서 비교)
    inter, n_users, n_items = toy
    tr = split_per_user(inter, seed=0).train
    pairs = np.array([(i, j) for i in range(n_items) for j in range(i + 1, n_items)])
    exact = ItemCF(n_neighbors=n_items).fit(tr, n_users, n_items)
    approx = ItemCF(n_neighbors=n_items, neighbor_fn=lambda f, n: pairs).fit(tr, n_users, n_items)
    assert np.allclose(exact.S.toarray(), approx.S.toarray())


def test_neighbor_fn_no_candidates_means_no_neighbors(toy):
    # 후보가 하나도 없으면 이웃도 없다 (후보 밖 쌍은 유사도를 계산하지 않음)
    inter, n_users, n_items = toy
    tr = split_per_user(inter, seed=0).train
    m = ItemCF(n_neighbors=K, neighbor_fn=lambda f, n: np.empty((0, 2), dtype=np.int64)).fit(tr, n_users, n_items)
    assert m.S.nnz == 0