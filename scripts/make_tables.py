"""방법 간 비교 표: results/metrics_*_{stage}.json → results/tables/compare_{stage}.{md,csv}.

예)
  python scripts/make_tables.py --stage val
  python scripts/make_tables.py --stage val --runs Random=random Popularity=popularity MF=mf_final

--runs의 각 항목은 "표에 쓸 이름=결과 파일 이름"(metrics_<이름>_<stage>.json의 <이름>).
생략하면 해당 stage의 결과 파일을 모두 쓴다. 없는 파일은 경고하고 건너뛴다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lastfm_rec.tables import check_consistency, compare_table, to_markdown  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val", choices=["val", "test"])
    ap.add_argument("--runs", nargs="*", help="이름=결과 파일 이름 (예: MF=mf_final)")
    ap.add_argument("--decimals", type=int, default=4)
    args = ap.parse_args()

    if args.runs:
        specs = [r.split("=", 1) if "=" in r else (r, r) for r in args.runs]
    else:
        suffix = f"_{args.stage}.json"
        specs = [(p.name[len("metrics_"):-len(suffix)],) * 2
                 for p in sorted(RESULTS.glob(f"metrics_*{suffix}"))]

    results = {}
    for label, stem in specs:
        path = RESULTS / f"metrics_{stem}_{args.stage}.json"
        if not path.exists():
            print(f"[skip] {label}: {path.name} 없음", file=sys.stderr)
            continue
        results[label] = json.loads(path.read_text())
    if not results:
        raise SystemExit("표에 넣을 결과가 없음")

    for issue in check_consistency(results):
        print(f"[warn] {issue}", file=sys.stderr)

    k = next(iter(results.values()))["k"]
    out_dir = RESULTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# 방법 간 비교 ({args.stage}, K={k}, seed별 평균 ± 표준편차)\n"]
    frames = []
    for scope in ["all", "reachable"]:
        df = compare_table(results, scope)
        md += [f"## {scope}\n", to_markdown(df, k, args.decimals), ""]
        frames.append(df.assign(scope=scope))
    md.append("파라미터: " + "; ".join(f"{label} `{json.dumps(m['params'])}`" for label, m in results.items()))

    md_path, csv_path = out_dir / f"compare_{args.stage}.md", out_dir / f"compare_{args.stage}.csv"
    md_path.write_text("\n".join(md) + "\n")
    import pandas as pd
    pd.concat(frames).to_csv(csv_path)
    print("\n".join(md))
    print(f"\nsaved {md_path}\nsaved {csv_path}")


if __name__ == "__main__":
    main()
