"""방법 간 차이의 paired bootstrap 표: results/per_user_*_{stage}_s{seed}_{scope}.csv
→ results/tables/bootstrap_{stage}_{scope}.{md,csv}, bootstrap_{stage}_{scope}_detail.csv.

예)
  python scripts/bootstrap_table.py --stage test --runs Popularity=popularity MF=mf_final RWR=rwr_final
  python scripts/bootstrap_table.py --stage test --runs Popularity=popularity MF=mf_final RWR=rwr_final \
      --pairs MF:Popularity RWR:MF

seed마다 같은 사용자 집합에서 mean(A − B)의 95% 신뢰구간을 구한다(난수 seed = 분할 seed).
판정: 모든 seed의 구간이 0보다 크면 A > B, 모두 0보다 작으면 A < B, 모두 0을 포함하면 차이 없음, 그 밖은 혼재.
--pairs를 생략하면 --runs의 모든 쌍(적은 순서대로 A, B)을 비교한다.
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lastfm_rec.tables import bootstrap_markdown, bootstrap_pairs  # noqa: E402

RESULTS = ROOT / "results"


def load_per_user(stem: str, stage: str, scope: str) -> dict[int, pd.DataFrame]:
    pat = re.compile(rf"per_user_{re.escape(stem)}_{stage}_s(\d+)_{scope}\.csv$")
    out = {}
    for p in RESULTS.glob(f"per_user_{stem}_{stage}_s*_{scope}.csv"):
        m = pat.search(p.name)
        if m:
            out[int(m.group(1))] = pd.read_csv(p, index_col="u")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="test", choices=["val", "test"])
    ap.add_argument("--scope", default="all", choices=["all", "reachable"])
    ap.add_argument("--runs", nargs="+", required=True, help="이름=결과 파일 이름 (예: MF=mf_final)")
    ap.add_argument("--pairs", nargs="*", help="A:B (이름 기준). 생략하면 모든 쌍")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    per_user = {}
    for spec in args.runs:
        label, stem = spec.split("=", 1) if "=" in spec else (spec, spec)
        data = load_per_user(stem, args.stage, args.scope)
        if not data:
            print(f"[skip] {label}: per_user_{stem}_{args.stage}_s*_{args.scope}.csv 없음", file=sys.stderr)
            continue
        per_user[label] = data

    if args.pairs:
        pairs = [tuple(p.split(":", 1)) for p in args.pairs]
        missing = {x for pair in pairs for x in pair} - set(per_user)
        if missing:
            print(f"[skip] 결과가 없는 이름이 들어간 쌍 제외: {sorted(missing)}", file=sys.stderr)
            pairs = [p for p in pairs if not set(p) & missing]
    else:
        pairs = list(itertools.combinations(per_user, 2))
    if not pairs:
        raise SystemExit("비교할 쌍이 없음")

    detail, summary, issues = bootstrap_pairs(per_user, pairs, n_boot=args.n_boot)
    for issue in issues:
        print(f"[warn] {issue}", file=sys.stderr)

    out_dir = RESULTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"bootstrap_{args.stage}_{args.scope}"
    md = (f"# Paired bootstrap ({args.stage}, {args.scope}, 사용자 단위, {args.n_boot}회, 95% CI)\n\n"
          + bootstrap_markdown(summary)
          + "\n\n- Δ mean: seed별 mean(A − B)의 평균"
          + "\n- seed별 95% CI 범위: seed마다 구한 구간 중 가장 낮은 하한 ~ 가장 높은 상한(하나의 신뢰구간이 아님)."
          + " seed별 값은 *_detail.csv"
          + f"\n- 쌍 {len(pairs)}개 × 지표 2개를 각각 검정함(다중 비교 보정 없음)\n")
    Path(f"{base}.md").write_text(md)
    summary.to_csv(f"{base}.csv", index=False)
    detail.to_csv(f"{base}_detail.csv", index=False)
    print(md)
    print(f"saved {base}.md\nsaved {base}.csv\nsaved {base}_detail.csv")


if __name__ == "__main__":
    main()
