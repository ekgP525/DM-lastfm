"""run_experiment.py 결과(metrics_*.json)를 방법 간 비교 표로 만든다."""
from __future__ import annotations

import numpy as np
import pandas as pd

# (열 이름, summary 키 템플릿, 높을수록 좋은 정확도 지표인지)
METRICS = [
    ("Recall@{k}", "recall@{k}", True),
    ("NDCG@{k}", "ndcg@{k}", True),
    ("Coverage", "coverage", False),
    ("Tail ratio", "tail_ratio", False),
    ("Avg popularity", "avg_popularity", False),
]


def check_consistency(results: dict[str, dict]) -> list[str]:
    """K, stage, seed 집합, 평가 사용자 수가 방법마다 같은지 확인하고 어긋난 항목을 문장으로 돌려준다."""
    issues = []
    for field, get in [
        ("k", lambda m: m["k"]),
        ("stage", lambda m: m["stage"]),
        ("seeds", lambda m: tuple(r["seed"] for r in m["runs"])),
        # seed 개수 차이는 위에서 잡으므로 여기서는 사용자 수 값만 비교한다
        ("n_eval_users(all)", lambda m: frozenset(r["all"]["n_eval_users"] for r in m["runs"])),
    ]:
        values = {label: get(m) for label, m in results.items()}
        if len(set(values.values())) > 1:
            issues.append(f"{field}가 방법마다 다름: {values}")
    return issues


def compare_table(results: dict[str, dict], scope: str) -> pd.DataFrame:
    """행 = 방법, 열 = 지표별 mean/std와 seed당 학습 시간·사용자당 추천 시간(ms)."""
    rows = []
    for label, m in results.items():
        k, s = m["k"], m["summary"][scope]
        row = {"method": label}
        for col, key, _ in METRICS:
            row[f"{col.format(k=k)} mean"] = s[key.format(k=k)]["mean"]
            row[f"{col.format(k=k)} std"] = s[key.format(k=k)]["std"]
        row["Fit sec"] = float(np.mean([r["fit_sec"] for r in m["runs"]]))
        row["Rec ms/user"] = 1000 * float(np.mean([r["rec_sec_per_user"] for r in m["runs"]]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def to_markdown(df: pd.DataFrame, k: int, decimals: int = 4) -> str:
    """mean ± std 형식. 정확도 지표(Recall, NDCG)는 방법 중 최댓값을 굵게 표시한다."""
    cols = [col.format(k=k) for col, _, _ in METRICS]
    best = {col.format(k=k) for col, _, acc in METRICS if acc}
    header = ["Method", *cols, "Fit sec", "Rec ms/user"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for method, r in df.iterrows():
        cells = [str(method)]
        for c in cols:
            d = 1 if c == "Avg popularity" else decimals
            cell = f"{r[f'{c} mean']:.{d}f} ± {r[f'{c} std']:.{d}f}"
            if c in best and np.isclose(r[f"{c} mean"], df[f"{c} mean"].max()):
                cell = f"**{cell}**"
            cells.append(cell)
        cells += [f"{r['Fit sec']:.1f}", f"{r['Rec ms/user']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict(ci_lows: list[float], ci_highs: list[float], a: str, b: str) -> str:
    """seed별 신뢰구간을 하나의 판정으로 묶는다. 모든 seed가 같은 방향이어야 차이로 본다."""
    if all(lo > 0 for lo in ci_lows):
        return f"{a} > {b}"
    if all(hi < 0 for hi in ci_highs):
        return f"{a} < {b}"
    if all(lo <= 0 <= hi for lo, hi in zip(ci_lows, ci_highs)):
        return "차이 없음"
    return "혼재"


def bootstrap_pairs(per_user: dict[str, dict[int, pd.DataFrame]], pairs: list[tuple[str, str]],
                    metrics: tuple[str, ...] = ("ndcg", "recall"), n_boot: int = 2000
                    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """per_user[방법][seed] = 사용자별 점수(index u). seed마다 paired bootstrap(난수 seed = 분할 seed).

    반환: (seed별 상세, 쌍·지표별 요약, 경고 문장들)
    """
    from .metrics import paired_bootstrap

    rows, issues = [], []
    for a, b in pairs:
        seeds = sorted(set(per_user[a]) & set(per_user[b]))
        if set(per_user[a]) != set(per_user[b]):
            issues.append(f"{a} vs {b}: seed 집합이 다름 → 공통 seed {seeds}만 사용")
        for s in seeds:
            ua, ub = per_user[a][s], per_user[b][s]
            if set(ua.index) != set(ub.index):
                issues.append(f"{a} vs {b} seed {s}: 평가 사용자 집합이 다름 → 공통 사용자만 사용")
            for m in metrics:
                r = paired_bootstrap(ua[m], ub[m], n_boot=n_boot, seed=s)
                rows.append({"a": a, "b": b, "metric": m, "seed": s, **r})
    detail = pd.DataFrame(rows)

    summary = []
    for (a, b, m), g in detail.groupby(["a", "b", "metric"], sort=False):
        summary.append({
            "a": a, "b": b, "metric": m, "diff_mean": g["diff"].mean(),
            "ci_low_min": g["ci_low"].min(), "ci_high_max": g["ci_high"].max(),
            "n_significant": int(g["significant"].sum()), "n_seeds": len(g),
            "verdict": verdict(list(g["ci_low"]), list(g["ci_high"]), a, b),
        })
    return detail, pd.DataFrame(summary), issues


def bootstrap_markdown(summary: pd.DataFrame, decimals: int = 4) -> str:
    header = ["Comparison (A − B)", "Metric", "Δ mean", "seed별 95% CI 범위", "유의한 seed", "판정"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for _, r in summary.iterrows():
        d = decimals
        lines.append("| " + " | ".join([
            f"{r['a']} − {r['b']}", {"ndcg": "NDCG", "recall": "Recall"}.get(r["metric"], r["metric"]),
            f"{r['diff_mean']:+.{d}f}", f"[{r['ci_low_min']:+.{d}f}, {r['ci_high_max']:+.{d}f}]",
            f"{r['n_significant']}/{r['n_seeds']}", r["verdict"],
        ]) + " |")
    return "\n".join(lines)
