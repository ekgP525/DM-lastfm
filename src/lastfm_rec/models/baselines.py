"""단순 규칙 baseline. 알고리즘 개수에는 포함하지 않는다."""
from __future__ import annotations

import numpy as np

from .base import Recommender


class Popularity(Recommender):
    """학습 청취자 수가 많은 순."""
    name = "popularity"

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        self.pop = np.bincount(fit_df["i"].to_numpy(), minlength=n_items).astype(float)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class RandomRec(Recommender):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, users):
        return self.rng.random((len(users), self.n_items))
