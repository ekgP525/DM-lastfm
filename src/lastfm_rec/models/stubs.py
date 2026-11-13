"""담당자가 채울 알고리즘. 인터페이스는 Recommender를 따른다.

fit(fit_df, n_users, n_items)에서 super().fit(...)을 먼저 호출하고,
score(users)가 (len(users), n_items) 점수 행렬을 반환하면 run_experiment.py와 평가 코드가 그대로 동작한다.
"""
from __future__ import annotations

from .base import Recommender


class ItemCF(Recommender):
    """DM_09 Recom1. 아티스트–아티스트 코사인 유사도(log 가중치) → 이웃 n_neighbors개 가중합.
    neighbor_fn을 주면 그 함수가 고른 후보 안에서만 이웃을 찾는다(MinHash-LSH 근사 이웃)."""
    name = "itemcf"

    def __init__(self, n_neighbors: int = 50, neighbor_fn=None):
        self.n_neighbors, self.neighbor_fn = n_neighbors, neighbor_fn

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        raise NotImplementedError("행렬 담당: 코사인 유사도 + top-n 이웃")

    def score(self, users):
        raise NotImplementedError


class MinHashLSH:
    """DM_04 LSH. 아티스트 = 학습 청취자 집합 → MinHash 시그니처(n = b × r) → banding 후보 쌍.
    candidates()는 아티스트별 후보 이웃 인덱스를 돌려주고 ItemCF(neighbor_fn=...)에 넘긴다.
    기록할 것: 정확한 top-k 이웃 대비 recall, 이웃 계산 시간, 청취자 1명 아티스트 처리 방식."""

    def __init__(self, b: int = 20, r: int = 5, seed: int = 0):
        self.b, self.r, self.seed = b, r, seed

    def candidates(self, fit_df, n_items):
        raise NotImplementedError("행렬 담당")


class MF(Recommender):
    """DM_10 Recom2. bias 포함 regularized MF를 SGD로 직접 구현.
    관측 쌍(w) + 사용자별로 샘플링한 미관측 쌍(0, 관측 수의 neg_ratio배)으로 학습."""
    name = "mf"

    def __init__(self, n_factors: int = 32, lr: float = 0.01, reg: float = 0.05,
                 epochs: int = 30, neg_ratio: int = 3, seed: int = 0):
        self.__dict__.update(n_factors=n_factors, lr=lr, reg=reg, epochs=epochs,
                             neg_ratio=neg_ratio, seed=seed)

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        raise NotImplementedError("행렬 담당")

    def score(self, users):
        raise NotImplementedError


class RWR(Recommender):
    """DM_11 Link Analysis. graph.build_graph → column_stochastic M.
    r = (1 - c) M r + c e_u 를 power iteration(L1 변화 < 1e-6 또는 100회)으로 계산.
    여러 사용자의 e_u를 열로 묶어 한 번에 계산. 점수는 아티스트 노드 구간만 사용.
    beta > 0이면 점수를 (학습 청취자 수 + 1)^beta로 나눈다(인기도 보정, H2)."""
    name = "rwr"

    def __init__(self, restart: float = 0.3, beta: float = 0.0, graph: str = "G1",
                 tol: float = 1e-6, max_iter: int = 100):
        self.__dict__.update(restart=restart, beta=beta, graph=graph, tol=tol, max_iter=max_iter)

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        raise NotImplementedError("그래프 담당")

    def score(self, users):
        raise NotImplementedError
