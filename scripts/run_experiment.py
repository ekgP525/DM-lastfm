"""한 방법을 seed 여러 개로 돌려 지표를 results/에 저장한다.

예)
  python scripts/run_experiment.py --method popularity --stage val
  python scripts/run_experiment.py --method rwr --stage test --params '{"restart": 0.3, "beta": 0.5}'

--stage val : train으로 학습, validation으로 평가 (하이퍼파라미터 선택)
--stage test: train+val로 학습, test로 평가 (하이퍼파라미터 확정 후 한 번)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lastfm_rec.data import load_user_artists, preprocess  # noqa: E402
from lastfm_rec.metrics import evaluate  # noqa: E402
from lastfm_rec.models import REGISTRY  # noqa: E402
from lastfm_rec.split import split_per_user  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--stage", default="val", choices=["val", "test"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--params", default="{}", help="모델 하이퍼파라미터 JSON")
    ap.add_argument("--tag", default="", help="결과 파일 이름에 붙일 꼬리표")
    args = ap.parse_args()

    params = json.loads(args.params)
    enc, _ = preprocess(load_user_artists())
    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        sp = split_per_user(enc.interactions, seed=seed)
        fit_df, eval_df = sp.fit_set(args.stage), sp.eval_set(args.stage)
        model = REGISTRY[args.method](seed, **params)

        t0 = time.perf_counter()
        model.fit(fit_df, enc.n_users, enc.n_items)
        t_fit = time.perf_counter() - t0

        users = np.sort(eval_df["u"].unique())
        t0 = time.perf_counter()
        recs = model.recommend(users, k=args.k)
        t_rec = time.perf_counter() - t0

        row = {"seed": seed, "fit_sec": t_fit, "rec_sec_per_user": t_rec / len(users)}
        for scope, reach in [("all", False), ("reachable", True)]:
            summ, per_user = evaluate(recs, eval_df, fit_df, enc.n_items, k=args.k, restrict_reachable=reach)
            row[scope] = summ
            per_user.to_csv(ROOT / "results" / f"per_user_{args.method}{args.tag}_{args.stage}_s{seed}_{scope}.csv")
        runs.append(row)
        print(json.dumps(row, ensure_ascii=False))

    agg = {}
    for scope in ["all", "reachable"]:
        keys = runs[0][scope].keys()
        agg[scope] = {k: {"mean": float(np.mean([r[scope][k] for r in runs])),
                          "std": float(np.std([r[scope][k] for r in runs]))} for k in keys}
    out = {"method": args.method, "stage": args.stage, "params": params, "k": args.k,
           "env": {"python": platform.python_version(), "numpy": np.__version__,
                   "machine": platform.machine(), "system": platform.system()},
           "runs": runs, "summary": agg}
    path = ROOT / "results" / f"metrics_{args.method}{args.tag}_{args.stage}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved {path}")


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
