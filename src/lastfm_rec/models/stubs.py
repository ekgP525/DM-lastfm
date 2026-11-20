"""담당자가 채울 알고리즘. 인터페이스는 Recommender를 따른다.

fit(fit_df, n_users, n_items)에서 super().fit(...)을 먼저 호출하고,
score(users)가 (len(users), n_items) 점수 행렬을 반환하면 run_experiment.py와 평가 코드가 그대로 동작한다.
"""
from __future__ import annotations

import numpy as np

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
    관측 쌍(w) + 사용자별로 샘플링한 미관측 쌍(0, 관측 수의 neg_ratio배)으로 학습.

    예측 r_ui = mu + b_u + b_i + p_u · q_i, 손실 = 제곱 오차 + reg * (b_u² + b_i² + |p_u|² + |q_i|²).
    미관측 쌍은 epoch마다 균등 샘플링으로 새로 뽑고, 학습 데이터에 있는 쌍은 다시 뽑는다.
    갱신은 batch_size개씩 묶어 numpy로 처리한다(배치 안에서는 같은 파라미터로 기울기를 계산하는 미니배치 SGD)."""
    name = "mf"

    def __init__(self, n_factors: int = 32, lr: float = 0.01, reg: float = 0.05,
                 epochs: int = 30, neg_ratio: int = 3, seed: int = 0, batch_size: int = 1024,
                 init_std: float = 0.1):
        self.__dict__.update(n_factors=n_factors, lr=lr, reg=reg, epochs=epochs,
                             neg_ratio=neg_ratio, seed=seed, batch_size=batch_size, init_std=init_std)

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        rng = np.random.default_rng(self.seed)
        u_obs = fit_df["u"].to_numpy(dtype=np.int64)
        i_obs = fit_df["i"].to_numpy(dtype=np.int64)
        w_obs = fit_df["w"].to_numpy(dtype=float)
        self._seen_keys = np.unique(u_obs * n_items + i_obs)

        self.P = rng.normal(0.0, self.init_std, (n_users, self.n_factors))
        self.Q = rng.normal(0.0, self.init_std, (n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        # 관측 + 미관측(0) 전체 학습 목표값의 평균
        self.mu = float(w_obs.sum() / (len(w_obs) * (1 + self.neg_ratio)))

        self.rmse_history = []
        for _ in range(self.epochs):
            u_neg, i_neg = self._sample_negatives(u_obs, rng)
            u = np.concatenate([u_obs, u_neg])
            i = np.concatenate([i_obs, i_neg])
            y = np.concatenate([w_obs, np.zeros(len(u_neg))])
            order = rng.permutation(len(u))
            sq = 0.0
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                sq += self._step(u[idx], i[idx], y[idx])
            self.rmse_history.append(float(np.sqrt(sq / len(u))))
        return self

    def _sample_negatives(self, u_obs, rng):
        """사용자별 관측 수 × neg_ratio개의 미관측 아티스트. 학습 쌍과 겹치면 다시 뽑는다."""
        u_neg = np.repeat(u_obs, self.neg_ratio)
        i_neg = rng.integers(0, self.n_items, len(u_neg))
        bad = np.isin(u_neg * self.n_items + i_neg, self._seen_keys)
        while bad.any():
            i_neg[bad] = rng.integers(0, self.n_items, int(bad.sum()))
            bad[bad] = np.isin(u_neg[bad] * self.n_items + i_neg[bad], self._seen_keys)
        return u_neg, i_neg

    def _step(self, u, i, y) -> float:
        pu, qi = self.P[u], self.Q[i]
        err = y - (self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", pu, qi))
        lr, reg = self.lr, self.reg
        # 같은 사용자·아티스트가 배치에 여러 번 나오면 np.add.at으로 갱신을 누적한다
        np.add.at(self.bu, u, lr * (err - reg * self.bu[u]))
        np.add.at(self.bi, i, lr * (err - reg * self.bi[i]))
        np.add.at(self.P, u, lr * (err[:, None] * qi - reg * pu))
        np.add.at(self.Q, i, lr * (err[:, None] * pu - reg * qi))
        return float(err @ err)

    def score(self, users):
        users = np.asarray(users)
        return self.mu + self.bu[users, None] + self.bi[None, :] + self.P[users] @ self.Q.T


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
