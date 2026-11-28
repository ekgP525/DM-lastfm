"""H3: MinHash-LSH (b, r) 탐색. seed 하나의 validation에서 조합별로 아래를 재서 results/lsh_sweep*.csv에 저장한다.

- 이웃 recall@k: 정확한 Item-CF top-k 이웃 쌍 중 LSH 후보 쌍에 들어간 비율 (전체 쌍 기준 micro 평균)
  기본 k = n_neighbors(val에서 고른 2000, 계획서 기준). 보조로 k = 50도 함께 잰다.
  후보에 들어간 정확한 top-k 이웃은 근사 top-k에도 반드시 들어가므로(후보는 전체의 부분집합),
  이 값은 "근사 이웃이 정확한 이웃을 얼마나 되찾는가"와 같다(동점 처리 차이 제외).
- 후보 쌍 수, 공통 청취자가 있는 전체 쌍 대비 비율
- 시간: LSH(시그니처 + banding), 후보 코사인 + top-k, 합계. 정확한 Item-CF fit 시간과 비교
- 근사 Item-CF의 Recall@10, NDCG@10 등 (정확한 Item-CF와 같은 평가)

예)
  python scripts/lsh_sweep.py --fixed '{"n_neighbors": 2000}' --b 10,20,50,100,200 --r 1,2,3,5

선택 기준(계획서): 이웃 recall@2000 ≥ 0.9인 조합 중 후보 쌍 수가 가장 적은 (b, r) 1개. test 데이터는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lastfm_rec.data import load_user_artists, preprocess
from lastfm_rec.metrics import evaluate
from lastfm_rec.models import ItemCF, MinHashLSH
from lastfm_rec.split import split_per_user


def neighbor_keys(S, n_items):
    """top-k 이웃 행렬 S의 (i, j) 쌍을 i < j 순서의 정수 키로. 이웃 관계는 방향이 있어 행 기준으로 센다."""
    S = S.tocoo()
    lo, hi = np.minimum(S.row, S.col), np.maximum(S.row, S.col)
    return lo.astype(np.int64) * n_items + hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", default="10,20,50,100,200", help="band 수 후보")
    ap.add_argument("--r", default="1,2,3,5", help="band당 행 수 후보")
    ap.add_argument("--fixed", default='{"n_neighbors": 2000}', help="Item-CF 파라미터 JSON")
    ap.add_argument("--aux_k", type=int, default=50, help="보조 이웃 recall의 k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    fixed = json.loads(args.fixed)
    main_k = fixed.get("n_neighbors", 50)
    enc, _ = preprocess(load_user_artists())
    split = split_per_user(enc.interactions, seed=args.seed)
    fit_df, eval_df = split.fit_set("val"), split.eval_set("val")
    users = np.sort(eval_df["u"].unique())
    n_items = enc.n_items

    # 정답: 정확한 Item-CF
    t0 = time.perf_counter()
    exact = ItemCF(**fixed).fit(fit_df, enc.n_users, n_items)
    t_exact = time.perf_counter() - t0
    exact_aux = ItemCF(**{**fixed, "n_neighbors": args.aux_k}).fit(fit_df, enc.n_users, n_items)
    truth_main = neighbor_keys(exact.S, n_items)          # 이웃 관계 하나당 키 하나(양방향이면 2번 셈)
    truth_aux = neighbor_keys(exact_aux.S, n_items)
    B = sp.csr_matrix((np.ones(len(fit_df)), (fit_df["u"], fit_df["i"])), shape=(enc.n_users, n_items))
    B.data[:] = 1.0
    n_nonzero_pairs = int(sp.triu(B.T @ B, k=1).nnz)     # 공통 청취자가 있는 쌍 = 코사인 > 0인 쌍

    summ, _ = evaluate(exact.recommend(users, k=args.k), eval_df, fit_df, n_items, k=args.k)
    rows = [{"method": "exact", "b": None, "r": None, "n_hash": None, "threshold": None,
             "n_candidates": n_nonzero_pairs, "cand_ratio": 1.0,
             f"nbr_recall@{main_k}": 1.0, f"nbr_recall@{args.aux_k}": 1.0,
             "lsh_sec": 0.0, "cos_sec": t_exact, "fit_sec": t_exact, **summ}]
    print(f"exact: pairs={n_nonzero_pairs:,} fit={t_exact:.2f}s recall@{args.k}={summ[f'recall@{args.k}']:.4f}")

    path = ROOT / "results" / f"lsh_sweep{args.tag}.csv"
    path.parent.mkdir(exist_ok=True)
    combos = list(itertools.product([int(x) for x in args.b.split(",")], [int(x) for x in args.r.split(",")]))
    for n, (b, r) in enumerate(combos, 1):
        t0 = time.perf_counter()
        pairs = MinHashLSH(b=b, r=r, seed=args.seed).candidates(fit_df, n_items)
        t_lsh = time.perf_counter() - t0
        t0 = time.perf_counter()
        model = ItemCF(**fixed, neighbor_fn=lambda f, n, p=pairs: p).fit(fit_df, enc.n_users, n_items)
        t_cos = time.perf_counter() - t0
        cand = pairs[:, 0] * n_items + pairs[:, 1]
        summ, _ = evaluate(model.recommend(users, k=args.k), eval_df, fit_df, n_items, k=args.k)
        rows.append({"method": "lsh", "b": b, "r": r, "n_hash": b * r, "threshold": (1 / b) ** (1 / r),
                     "n_candidates": len(pairs), "cand_ratio": len(pairs) / n_nonzero_pairs,
                     f"nbr_recall@{main_k}": float(np.isin(truth_main, cand).mean()),
                     f"nbr_recall@{args.aux_k}": float(np.isin(truth_aux, cand).mean()),
                     "lsh_sec": t_lsh, "cos_sec": t_cos, "fit_sec": t_lsh + t_cos, **summ})
        pd.DataFrame(rows).to_csv(path, index=False)   # 중간에 멈춰도 결과가 남도록 매번 저장
        last = rows[-1]
        print(f"[{n}/{len(combos)}] b={b} r={r} cand={len(pairs):,} "
              f"nbr@{main_k}={last[f'nbr_recall@{main_k}']:.3f} nbr@{args.aux_k}={last[f'nbr_recall@{args.aux_k}']:.3f} "
              f"recall@{args.k}={summ[f'recall@{args.k}']:.4f} ({t_lsh + t_cos:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    ok = df[(df["method"] == "lsh") & (df[f"nbr_recall@{main_k}"] >= 0.9)]
    if len(ok):
        best = ok.sort_values("n_candidates").iloc[0]
        print(f"\n선택(이웃 recall@{main_k} ≥ 0.9 중 후보 최소): b={int(best.b)}, r={int(best.r)}, "
              f"후보 {int(best.n_candidates):,} ({best.cand_ratio:.1%}), recall@{args.k}={best[f'recall@{args.k}']:.4f}")
    else:
        print(f"\n이웃 recall@{main_k} ≥ 0.9인 조합 없음 → 그대로 보고")
    print(f"saved {path}")


if __name__ == "__main__":
    main()