"""모든 추천기가 따르는 인터페이스."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Recommender:
    name = "base"

    def fit(self, fit_df: pd.DataFrame, n_users: int, n_items: int) -> "Recommender":
        self.n_users, self.n_items = n_users, n_items
        self.seen = sp.csr_matrix(
            (np.ones(len(fit_df)), (fit_df["u"].to_numpy(), fit_df["i"].to_numpy())),
            shape=(n_users, n_items))
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """(len(users), n_items) 점수 행렬. 하위 클래스가 구현한다."""
        raise NotImplementedError

    def recommend(self, users: np.ndarray, k: int = 10, batch: int = 256) -> dict[int, np.ndarray]:
        """이미 들은 아티스트를 뺀 상위 k개."""
        out = {}
        for s in range(0, len(users), batch):
            ub = users[s:s + batch]
            sc = np.asarray(self.score(ub), dtype=float)
            sc[self.seen[ub].nonzero()] = -np.inf
            top = np.argpartition(-sc, kth=min(k, sc.shape[1] - 1), axis=1)[:, :k]
            order = np.take_along_axis(sc, top, axis=1).argsort(axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, axis=1)
            out.update({int(u): top[j] for j, u in enumerate(ub)})
        return out
