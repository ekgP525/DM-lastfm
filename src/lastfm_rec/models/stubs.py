"""담당자가 채울 알고리즘. 인터페이스는 Recommender를 따른다.

fit(fit_df, n_users, n_items)에서 super().fit(...)을 먼저 호출하고,
score(users)가 (len(users), n_items) 점수 행렬을 반환하면 run_experiment.py와 평가 코드가 그대로 동작한다.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..graph import build_graph, column_stochastic
from .base import Recommender


class ItemCF(Recommender):
    """DM_09 Recom1. 아티스트–아티스트 코사인 유사도(log 가중치) → 이웃 n_neighbors개 가중합.
    neighbor_fn을 주면 그 함수가 고른 후보 안에서만 이웃을 찾는다(MinHash-LSH 근사 이웃)."""
    name = "itemcf"

    def __init__(self, n_neighbors: int = 50, neighbor_fn=None, normalize: bool = False):
        self.n_neighbors, self.neighbor_fn, self.normalize = n_neighbors, neighbor_fn, normalize

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        # 1) 사용자×아티스트 행렬 X. 값은 w = log(1 + plays), fit_df(학습 부분)만 사용
        u = fit_df["u"].to_numpy(dtype=np.int64)
        i = fit_df["i"].to_numpy(dtype=np.int64)
        w = fit_df["w"].to_numpy(dtype=float)
        self.X = sp.csr_matrix((w, (u, i)), shape=(n_users, n_items))

        # 2) 아티스트 열을 길이 1로 맞춘 뒤 Xn^T Xn = 아티스트끼리 코사인 유사도
        norms = np.sqrt(np.asarray(self.X.power(2).sum(axis=0)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        Xn = (self.X @ sp.diags(inv)).tocsc()
        if self.neighbor_fn is None:
            sim = (Xn.T @ Xn).tocsr()
        else:
            pairs = self.neighbor_fn(fit_df, n_items)
            sim = self._pair_cosine(Xn, pairs, n_items)
            self.n_candidate_pairs = len(pairs)
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        # 3) 아티스트마다 유사도 상위 n_neighbors개만 남긴다. S[i, j] = i의 이웃 j의 유사도
        self.S = self._keep_top_k(sim, self.n_neighbors)
        return self

    @staticmethod
    def _pair_cosine(Xn, pairs, n_items):
        """후보 쌍 (i, j)의 코사인만 계산해 대칭 희소 행렬로 돌려준다. 후보가 아닌 쌍은 0(이웃 아님)."""
        if len(pairs) == 0:
            return sp.csr_matrix((n_items, n_items))
        XnT = Xn.T.tocsr()
        pi, pj = pairs[:, 0], pairs[:, 1]
        vals = np.asarray(XnT[pi].multiply(XnT[pj]).sum(axis=1)).ravel()
        rows = np.concatenate([pi, pj])
        cols = np.concatenate([pj, pi])
        return sp.csr_matrix((np.concatenate([vals, vals]), (rows, cols)), shape=(n_items, n_items))

    @staticmethod
    def _keep_top_k(sim, k):
        """희소 행렬의 각 행에서 값이 큰 k개만 남긴다."""
        rows, cols, vals = [], [], []
        for r in range(sim.shape[0]):
            s, e = sim.indptr[r], sim.indptr[r + 1]
            idx, val = sim.indices[s:e], sim.data[s:e]
            if len(val) > k:
                top = np.argpartition(-val, k - 1)[:k]
                idx, val = idx[top], val[top]
            rows.append(np.full(len(idx), r))
            cols.append(idx)
            vals.append(val)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=sim.shape)

    def score(self, users):
        Xu = self.X[np.asarray(users)]
        sc = (Xu @ self.S.T).toarray()
        if self.normalize:
            heard = Xu.copy()
            heard.data[:] = 1.0
            den = (heard @ self.S.T).toarray()
            sc = np.divide(sc, den, out=np.zeros_like(sc), where=den > 0)
        return sc


class MinHashLSH:
    """DM_04 LSH. 아티스트 = 학습 청취자 집합 → MinHash 시그니처(n = b × r) → banding 후보 쌍.
    candidates()는 아티스트별 후보 이웃 인덱스를 돌려주고 ItemCF(neighbor_fn=...)에 넘긴다.
    기록할 것: 정확한 top-k 이웃 대비 recall, 이웃 계산 시간, 청취자 1명 아티스트 처리 방식."""

    P = 2_147_483_647

    def __init__(self, b: int = 20, r: int = 5, seed: int = 0):
        self.b, self.r, self.seed = b, r, seed
        self.n_hash = b * r

    def signatures(self, fit_df, n_items):
        """(n_hash, n_items) MinHash 시그니처. 학습 청취자가 없는 아티스트 열은 -1.
        h_k(u) = (a_k * u + c_k) mod P 를 순열 대신 쓰고, 아티스트별로 청취자 해시값의 최솟값을 취한다."""
        rng = np.random.default_rng(self.seed)
        a = rng.integers(1, self.P, self.n_hash, dtype=np.int64)
        c = rng.integers(0, self.P, self.n_hash, dtype=np.int64)

        # (아티스트, 사용자) 쌍을 중복 없이 아티스트 순으로 정렬 → 아티스트마다 청취자가 연속 구간
        pairs = np.unique(np.stack([fit_df["i"].to_numpy(np.int64),
                                    fit_df["u"].to_numpy(np.int64)], axis=1), axis=0)
        items, users = pairs[:, 0], pairs[:, 1]
        starts = np.flatnonzero(np.r_[True, items[1:] != items[:-1]])

        sig = np.full((self.n_hash, n_items), -1, dtype=np.int64)
        for s in range(0, self.n_hash, 32):
            H = (a[s:s + 32, None] * users[None, :] + c[s:s + 32, None]) % self.P
            sig[s:s + 32, items[starts]] = np.minimum.reduceat(H, starts, axis=1)
        self.has_listener = np.zeros(n_items, dtype=bool)
        self.has_listener[items[starts]] = True
        return sig

    def candidates(self, fit_df, n_items):
        """banding 후보 쌍. (후보 쌍 수, 2) int 배열, 각 행은 i < j인 (i, j), 중복 없음.
        어느 band 하나에서라도 r행이 모두 같은 아티스트끼리 후보가 된다.
        학습 청취자가 없는 아티스트(시그니처 -1)는 후보에서 뺀다."""
        sig = self.signatures(fit_df, n_items)
        active = np.flatnonzero(self.has_listener)
        keys = []
        for band in range(self.b):
            rows = sig[band * self.r:(band + 1) * self.r][:, active]
            # r칸이 완전히 같은 열끼리 같은 버킷 번호 받음
            _, bucket = np.unique(rows.T, axis=0, return_inverse=True)
            bucket = bucket.ravel()
            order = np.argsort(bucket, kind="stable")
            sb = bucket[order]
            starts = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1]])
            sizes = np.diff(np.r_[starts, len(sb)])
            # 버킷에 2명 이상 있으면 그 안의 모든 쌍이 후보
            for st, size in zip(starts[sizes > 1], sizes[sizes > 1]):
                members = active[order[st:st + size]]
                a, b = np.triu_indices(size, k=1)
                keys.append(members[a] * n_items + members[b])
        keys = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)
        return np.stack([keys // n_items, keys % n_items], axis=1)


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
        if self.graph != "G1":
            raise NotImplementedError("G2·G3는 친구·태그 입력 연결 후 지원")
        A = build_graph(fit_df, n_users, n_items)
        self.M = column_stochastic(A)
        self.n_nodes = A.shape[0]
        # 아티스트별 학습 청취자 수 (인기도 보정·분석용, 학습 데이터만 사용)
        self.popularity = np.bincount(fit_df["i"].to_numpy(dtype=np.int64), minlength=n_items)
        self.n_iter_ = []
        return self

    def _propagate(self, users) -> np.ndarray:
        """사용자별 RWR 정상 분포. (노드 수, len(users)) 행렬, 각 열이 한 사용자."""
        users = np.asarray(users, dtype=np.int64)
        c = self.restart
        E = np.zeros((self.n_nodes, len(users)))
        E[users, np.arange(len(users))] = 1.0
        R = E.copy()
        for it in range(1, self.max_iter + 1):
            R_new = (1.0 - c) * (self.M @ R) + c * E
            delta = np.abs(R_new - R).sum(axis=0).max()  # 열(사용자)별 L1 변화 중 최대
            R = R_new
            if delta < self.tol:
                break
        self.n_iter_.append(it)
        return R

    def score(self, users):
        R = self._propagate(users)
        sc = R[self.n_users:self.n_users + self.n_items].T  # 아티스트 노드 구간만
        if self.beta > 0:
            sc = sc / (self.popularity + 1.0) ** self.beta
        return sc


class P3(Recommender):
    """P3α / RP3β baseline (Dacrema et al. 2019, 2.2절의 정의를 따름).
    사용자 → 아티스트 → 사용자 → 아티스트 3걸음 전이확률로 점수를 매긴다.
    전이확률은 RWR과 같은 가중치 w로 p_ui = (w_ui / Σ_i w_ui)^α, p_iu = (w_ui / Σ_u w_ui)^α.
    beta > 0이면 RWR과 똑같이 (학습 청취자 수 + 1)^beta로 나눈다(β = 0이면 P3α, β > 0이면 RP3β).
    원 구현의 아이템 이웃 top-K 자르기는 하지 않는다(전체 3걸음 확률 사용)."""
    name = "p3"

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha, self.beta = alpha, beta

    def fit(self, fit_df, n_users, n_items):
        super().fit(fit_df, n_users, n_items)
        u = fit_df["u"].to_numpy(dtype=np.int64)
        i = fit_df["i"].to_numpy(dtype=np.int64)
        w = fit_df["w"].to_numpy(dtype=float)
        X = sp.csr_matrix((w, (u, i)), shape=(n_users, n_items))
        X.sum_duplicates()
        row = np.asarray(X.sum(axis=1)).ravel()
        col = np.asarray(X.sum(axis=0)).ravel()
        self.P_ui = (sp.diags(np.divide(1.0, row, out=np.zeros_like(row), where=row > 0)) @ X).tocsr()
        self.P_iu = (X @ sp.diags(np.divide(1.0, col, out=np.zeros_like(col), where=col > 0))).T.tocsr()
        if self.alpha != 1.0:
            self.P_ui = self.P_ui.power(self.alpha)
            self.P_iu = self.P_iu.power(self.alpha)
        self.popularity = np.bincount(i, minlength=n_items)
        return self

    def score(self, users):
        users = np.asarray(users, dtype=np.int64)
        first = self.P_ui[users]                        # (b, 아티스트) 1걸음
        second = (first @ self.P_iu).toarray()          # (b, 사용자) 2걸음
        sc = np.asarray(self.P_ui.T @ second.T).T       # (b, 아티스트) 3걸음
        if self.beta > 0:
            sc = sc / (self.popularity + 1.0) ** self.beta
        return sc
