"""하이퍼파라미터 그리드 탐색: seed 하나의 validation에서 조합별 지표를 results/tune_*.csv로 저장한다.

예)
  python scripts/tune.py --method mf --grid '{"n_factors": [16, 32, 64], "reg": [0.01, 0.05]}'

계획서 기준: seed 0의 validation NDCG@10(all)이 최대인 조합을 고르고 나머지 seed에 고정한다.
test 데이터는 쓰지 않는다.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lastfm_rec.data import load_user_artists, preprocess  # noqa: E402
from lastfm_rec.metrics import evaluate  # noqa: E402
from lastfm_rec.models import REGISTRY  # noqa: E402
from lastfm_rec.split import split_per_user  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--grid", required=True, help="파라미터 이름 → 후보 리스트 JSON")
    ap.add_argument("--fixed", default="{}", help="모든 조합에 공통으로 넣을 파라미터 JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    grid, fixed = json.loads(args.grid), json.loads(args.fixed)
    enc, _ = preprocess(load_user_artists())
    sp = split_per_user(enc.interactions, seed=args.seed)
    fit_df, eval_df = sp.fit_set("val"), sp.eval_set("val")
    users = np.sort(eval_df["u"].unique())

    path = ROOT / "results" / f"tune_{args.method}{args.tag}.csv"
    path.parent.mkdir(exist_ok=True)
    rows = []
    combos = list(itertools.product(*grid.values()))
    for n, values in enumerate(combos, 1):
        params = {**fixed, **dict(zip(grid, values))}
        model = REGISTRY[args.method](args.seed, **params)
        t0 = time.perf_counter()
        model.fit(fit_df, enc.n_users, enc.n_items)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        recs = model.recommend(users, k=args.k)
        t_rec = time.perf_counter() - t0  # 평가 사용자 전체 추천 시간
        summ, _ = evaluate(recs, eval_df, fit_df, enc.n_items, k=args.k)
        rows.append({**params, **summ, "fit_sec": t_fit, "rec_sec": t_rec})
        # 중간에 멈춰도 결과가 남도록 매번 저장
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{n}/{len(combos)}] {params} ndcg@{args.k}={summ[f'ndcg@{args.k}']:.4f} "
              f"recall@{args.k}={summ[f'recall@{args.k}']:.4f} ({t_fit:.1f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(f"ndcg@{args.k}", ascending=False)
    print("\ntop 5 by ndcg:\n" + df.head(5).to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
