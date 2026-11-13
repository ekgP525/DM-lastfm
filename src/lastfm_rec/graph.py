"""RWR용 그래프 구성. 학습 데이터만 받는다.

노드 순서: [사용자 0..n_users-1 | 아티스트 | 태그]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_graph(fit_df: pd.DataFrame, n_users: int, n_items: int,
                friends: pd.DataFrame | None = None,
                tags: pd.DataFrame | None = None, n_tags: int = 0,
                heldout_pairs: pd.DataFrame | None = None) -> sp.csr_matrix:
    """무방향 가중 인접행렬.

    G1: fit_df(사용자–아티스트, 가중치 w)만
    G2: + friends(u, v 인코딩된 사용자 쌍), 가중치 1
    G3: + tags(i, t 인코딩, 태그 부여 행), 아티스트–태그 가중치 log1p(부여 횟수)
        heldout_pairs(u, i)에 해당하는 (사용자, 아티스트)의 태그 부여는 제외한다.
    """
    n = n_users + n_items + n_tags
    rows = [fit_df["u"].to_numpy(), n_users + fit_df["i"].to_numpy()]
    cols = [n_users + fit_df["i"].to_numpy(), fit_df["u"].to_numpy()]
    vals = [fit_df["w"].to_numpy(), fit_df["w"].to_numpy()]

    if friends is not None and len(friends):
        f = friends[["u", "v"]].to_numpy()
        rows += [f[:, 0], f[:, 1]]
        cols += [f[:, 1], f[:, 0]]
        vals += [np.ones(len(f)), np.ones(len(f))]

    if tags is not None and len(tags):
        t = tags
        if heldout_pairs is not None and len(heldout_pairs):
            key = pd.MultiIndex.from_frame(heldout_pairs[["u", "i"]])
            t = t[~pd.MultiIndex.from_frame(t[["u", "i"]]).isin(key)]
        agg = t.groupby(["i", "t"]).size().reset_index(name="c")
        it = n_users + agg["i"].to_numpy()
        tt = n_users + n_items + agg["t"].to_numpy()
        w = np.log1p(agg["c"].to_numpy(dtype=float))
        rows += [it, tt]
        cols += [tt, it]
        vals += [w, w]

    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def column_stochastic(A: sp.csr_matrix) -> sp.csr_matrix:
    """열 합이 1인 전이 행렬 M (M[j, i] = i → j 확률). 고립 노드 열은 0으로 둔다."""
    deg = np.asarray(A.sum(axis=0)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return (A @ sp.diags(inv)).tocsr()
