import numpy as np
import pandas as pd

from lastfm_rec.models import MinHashLSH
from lastfm_rec.split import split_per_user


def _listener_sets(df, n_items):
    sets = [set() for _ in range(n_items)]
    for u, i in zip(df["u"], df["i"]):
        sets[i].add(u)
    return sets


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def test_signature_match_rate_estimates_jaccard(toy):
    # MinHash 핵심 성질: 시그니처 한 칸이 같을 확률 = 두 청취자 집합의 Jaccard
    inter, _, n_items = toy
    tr = split_per_user(inter, seed=0).train
    lsh = MinHashLSH(b=500, r=1, seed=0)
    sig = lsh.signatures(tr, n_items)
    sets = _listener_sets(tr, n_items)
    active = np.flatnonzero(lsh.has_listener)
    errs = [abs((sig[:, i] == sig[:, j]).mean() - _jaccard(sets[i], sets[j]))
            for k, i in enumerate(active) for j in active[k + 1:]]
    # n = 500이면 표준오차가 최대 √(0.25/500) ≈ 0.022라 평균 오차는 이보다 작아야 한다
    assert np.mean(errs) < 0.02


def test_signature_uses_train_only(toy):
    # leakage: val, test만 들은 아티스트는 청취자 없음(-1)이고, held-out을 바꿔도 시그니처는 그대로다
    inter, _, n_items = toy
    sp = split_per_user(inter, seed=0)
    lsh = MinHashLSH(b=10, r=2, seed=0)
    sig = lsh.signatures(sp.train, n_items)
    only_heldout = sorted(set(pd.concat([sp.val, sp.test])["i"]) - set(sp.train["i"]))
    assert not lsh.has_listener[only_heldout].any()
    assert (sig[:, only_heldout] == -1).all()

    sp.val["u"] = 0
    sp.test["u"] = 0
    assert np.array_equal(sig, MinHashLSH(b=10, r=2, seed=0).signatures(sp.train, n_items))


def test_candidate_format(toy):
    # 후보 쌍: i < j, 중복 없음, 학습 청취자가 있는 아티스트만
    inter, _, n_items = toy
    tr = split_per_user(inter, seed=0).train
    lsh = MinHashLSH(b=20, r=2, seed=0)
    pairs = lsh.candidates(tr, n_items)
    assert pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > 0
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    assert lsh.has_listener[pairs].all()


def test_identical_sets_always_and_disjoint_sets_never():
    # Jaccard 1인 쌍은 (b, r)과 상관없이 항상 후보, 공통 청취자가 없는 쌍은 절대 후보가 아니다
    # 아티스트 0·1: 청취자 {0, 1, 2}로 같음 / 아티스트 2: {3, 4} / 아티스트 3: {5} / 아티스트 4·5: 청취자 없음(둘 다 -1이라 걸러내지 않으면 후보가 됨)
    df = pd.DataFrame({"u": [0, 1, 2, 0, 1, 2, 3, 4, 5],
                       "i": [0, 0, 0, 1, 1, 1, 2, 2, 3],
                       "w": 1.0})
    for b, r in [(1, 1), (5, 4), (20, 10)]:
        pairs = MinHashLSH(b=b, r=r, seed=0).candidates(df, n_items=6)
        assert {tuple(p) for p in pairs} == {(0, 1)}


def test_candidates_reproducible_by_seed(toy):
    inter, _, n_items = toy
    tr = split_per_user(inter, seed=0).train
    a = MinHashLSH(b=20, r=3, seed=0).candidates(tr, n_items)
    b = MinHashLSH(b=20, r=3, seed=0).candidates(tr, n_items)
    c = MinHashLSH(b=20, r=3, seed=1).candidates(tr, n_items)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_banding_rate_follows_s_curve(toy):
    # 쌍이 후보가 될 확률 1 - (1 - J^r)^b: seed 여러 개 평균이 이론값과 가깝다
    inter, _, n_items = toy
    tr = split_per_user(inter, seed=0).train
    sets = _listener_sets(tr, n_items)
    b, r = 8, 3
    active = [i for i in range(n_items) if sets[i]]
    all_pairs = [(i, j) for k, i in enumerate(active) for j in active[k + 1:] if sets[i] & sets[j]]
    J = np.array([_jaccard(sets[i], sets[j]) for i, j in all_pairs])
    keys = np.array([i * n_items + j for i, j in all_pairs])
    hit = np.zeros(len(all_pairs))
    n_seeds = 30
    for s in range(n_seeds):
        p = MinHashLSH(b=b, r=r, seed=s).candidates(tr, n_items)
        hit += np.isin(keys, p[:, 0] * n_items + p[:, 1])
    observed, theory = hit / n_seeds, 1 - (1 - J ** r) ** b
    assert abs(observed.mean() - theory.mean()) < 0.03