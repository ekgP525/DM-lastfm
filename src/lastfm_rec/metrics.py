"""정확도·다양성 지표와 paired bootstrap.

모든 인기도 관련 값(평균 인기도, head/tail 구분)은 학습 데이터로만 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def truth_sets(eval_df: pd.DataFrame) -> dict[int, set[int]]:
    return eval_df.groupby("u")["i"].apply(set).to_dict()


def item_popularity(fit_df: pd.DataFrame, n_items: int) -> np.ndarray:
    """학습 청취자 수(아티스트별 학습 데이터 사용자 수)."""
    return np.bincount(fit_df["i"].to_numpy(), minlength=n_items)


def head_items(pop: np.ndarray, head_share: float = 0.2) -> np.ndarray:
    """청취 건수 누적 head_share를 차지하는 상위 아티스트 = head. bool 마스크 반환."""
    order = np.argsort(-pop, kind="stable")
    cum = np.cumsum(pop[order]) / max(pop.sum(), 1)
    n_head = int(np.searchsorted(cum, head_share) + 1)
    mask = np.zeros(len(pop), dtype=bool)
    mask[order[:n_head]] = True
    return mask


def unreachable_fraction(fit_df: pd.DataFrame, eval_df: pd.DataFrame, n_items: int) -> float:
    """평가 쌍 중 학습 청취자가 0명인 아티스트의 비율(어떤 방법도 맞힐 수 없는 몫)."""
    pop = item_popularity(fit_df, n_items)
    return float((pop[eval_df["i"].to_numpy()] == 0).mean())


def per_user_scores(recs: dict[int, np.ndarray], truth: dict[int, set[int]], k: int) -> pd.DataFrame:
    """사용자별 Recall@k, NDCG@k. bootstrap을 위해 사용자 단위로 남긴다."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for u, rel in truth.items():
        top = list(recs.get(u, []))[:k]
        hits = np.array([i in rel for i in top], dtype=float)
        hits = np.pad(hits, (0, k - len(hits)))
        dcg = float((hits * discounts).sum())
        idcg = float(discounts[: min(len(rel), k)].sum())
        rows.append((u, hits.sum() / len(rel), dcg / idcg if idcg > 0 else 0.0))
    return pd.DataFrame(rows, columns=["u", "recall", "ndcg"]).set_index("u")


def evaluate(recs: dict[int, np.ndarray], eval_df: pd.DataFrame, fit_df: pd.DataFrame,
             n_items: int, k: int = 10, restrict_reachable: bool = False) -> tuple[dict, pd.DataFrame]:
    """요약 지표 dict와 사용자별 점수 DataFrame을 반환한다.

    restrict_reachable=True면 학습 청취자 1명 이상인 아티스트만 정답으로 둔다.
    """
    pop = item_popularity(fit_df, n_items)
    if restrict_reachable:
        eval_df = eval_df[pop[eval_df["i"].to_numpy()] > 0]
    truth = truth_sets(eval_df)
    per_user = per_user_scores(recs, truth, k)

    rec_items = np.concatenate([np.asarray(recs[u])[:k] for u in truth if u in recs] or [np.array([], int)])
    head = head_items(pop)
    summary = {
        f"recall@{k}": float(per_user["recall"].mean()),
        f"ndcg@{k}": float(per_user["ndcg"].mean()),
        "coverage": float(len(np.unique(rec_items)) / n_items),
        "avg_popularity": float(pop[rec_items].mean()) if len(rec_items) else float("nan"),
        "tail_ratio": float((~head[rec_items]).mean()) if len(rec_items) else float("nan"),
        "n_eval_users": int(len(truth)),
    }
    return summary, per_user


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """같은 사용자 집합에서 mean(a - b)의 bootstrap 신뢰구간. 구간이 0을 포함하면 차이 없음."""
    users = a.index.intersection(b.index)
    d = (a.loc[users] - b.loc[users]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}
