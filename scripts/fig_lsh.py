"""H3 그림 3: MinHash-LSH (b, r)별 후보 쌍 비율 대비 이웃 recall과 Recall@10 → results/figures/fig3_lsh.{pdf,png}.

입력은 scripts/lsh_sweep.py가 만든 results/lsh_sweep{tag}.csv (seed 0 validation).
(a) 이웃 recall: 실선 = 정확한 top-k(k = n_neighbors, 기본 기준), 점선 = top-50(보조 분석)
    회색 대각선 y = x = 공통 청취자가 있는 쌍에서 후보를 무작위로 뽑았을 때의 기대 recall
(b) 근사 Item-CF의 Recall@10. 회색 점선 = 정확한 Item-CF
선은 r별로 하나, 선 위의 점은 b = 10 → 200 (왼쪽에서 오른쪽으로 후보가 늘어난다).

예)
  python scripts/fig_lsh.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK = "#52514e"
REF = "#8a8984"                                           # 기준선
RAMP = ["#84b9f3", "#338ced", "#1b64be", "#0e3d7e"]       # r이 커질수록 진하게
MARKERS = ["o", "s", "^", "D"]
LABEL_OFFSETS = [(5, -7), (0, 9), (5, -7), (5, -7)]   # (b) 선 끝 라벨 위치 (points), 겹치지 않게 r별로


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(True, color="#ecebe7", linewidth=0.6)
    ax.set_axisbelow(True)


def k_of(col):
    return int(re.search(r"\d+", col).group())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "results" / f"lsh_sweep{args.tag}.csv")
    main_col, aux_col = sorted((c for c in df.columns if c.startswith("nbr_recall@")), key=lambda c: -k_of(c))[:2]
    main_k, aux_k = k_of(main_col), k_of(aux_col)
    rec_col = f"recall@{args.k}"
    exact = df[df["method"] == "exact"].iloc[0]
    lsh = df[df["method"] == "lsh"].copy()
    lsh["pct"] = lsh["cand_ratio"] * 100
    rs = sorted(lsh["r"].astype(int).unique())

    ok = lsh[lsh[main_col] >= 0.9]                         # 선택 기준: 이웃 recall ≥ 0.9 중 후보 최소
    best = ok.sort_values("n_candidates").iloc[0] if len(ok) else None

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    line = dict(lw=2, ms=5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    for n, r in enumerate(rs):
        d = lsh[lsh["r"] == r].sort_values("pct")
        color, mk = RAMP[n % len(RAMP)], MARKERS[n % len(MARKERS)]
        ax_a.plot(d["pct"], d[main_col], color=color, marker=mk, **line)
        ax_a.plot(d["pct"], d[aux_col], color=color, marker=mk, lw=1.2, ls=(0, (3, 2)), ms=3.5,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=2)
        ax_b.plot(d["pct"], d[rec_col], color=color, marker=mk, **line)
        last = d.iloc[-1]                                  # 선 끝 직접 라벨 (b가 가장 큰 점)
        dx, dy = LABEL_OFFSETS[n % len(LABEL_OFFSETS)]
        ax_b.annotate(f"r={r}", (last["pct"], last[rec_col]), xytext=(dx, dy), textcoords="offset points",
                      fontsize=7.5, color=INK, va="center", ha="left" if dx else "center")

    # (a) 대각선(무작위 후보 기준), 0.9 기준선, 정확한 Item-CF, 선택된 조합
    ax_a.plot([0, 100], [0, 1], color=REF, lw=0.8, zorder=1)
    ax_a.axhline(0.9, color=REF, lw=0.8, ls=":", zorder=1)
    ax_a.text(1, 0.91, "0.9 (selection rule)", fontsize=7, color=INK, va="bottom")
    ax_a.scatter([100], [1.0], marker="*", s=70, color=REF, zorder=4)
    ax_a.annotate("exact", (100, 1.0), xytext=(-5, 4), textcoords="offset points",
                  fontsize=7.5, color=INK, ha="right", va="bottom")
    if best is not None:
        ax_a.annotate(f"chosen (b={int(best.b)}, r={int(best.r)})\n{int(best.n_candidates):,} pairs",
                      (best.pct, best[main_col]), xytext=(-6, -58), textcoords="offset points", ha="right",
                      fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    style(ax_a, "(a) Neighbor recall", "Candidate pairs (% of pairs with a shared listener)", "Neighbor recall")
    ax_a.set_xlim(0, 105)
    ax_a.set_ylim(0, 1.08)

    # (b) 정확한 Item-CF 기준선
    ax_b.axhline(exact[rec_col], color=REF, lw=1, ls=(0, (3, 2)), zorder=1)
    ax_b.text(1, exact[rec_col] + 0.003, f"exact Item-CF {exact[rec_col]:.3f}", fontsize=7, color=INK, va="bottom")
    style(ax_b, f"(b) Approximate Item-CF Recall@{args.k}",
          "Candidate pairs (% of pairs with a shared listener)", f"Recall@{args.k}")
    ax_b.set_xlim(0, 112)
    ax_b.set_ylim(0, max(lsh[rec_col].max(), exact[rec_col]) * 1.15)

    r_handles = [Line2D([], [], color=RAMP[n % len(RAMP)], lw=2, marker=MARKERS[n % len(MARKERS)], ms=5,
                        markeredgecolor="white", label=f"r = {r}") for n, r in enumerate(rs)]
    style_handles = [Line2D([], [], color=INK, lw=2, label=f"exact top-{main_k} (main)"),
                     Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 2)), label=f"exact top-{aux_k} (supplementary)"),
                     Line2D([], [], color=REF, lw=0.8, label="y = x (random candidates)")]
    ax_a.legend(handles=style_handles, fontsize=6.5, frameon=False, loc="lower right")
    ax_b.legend(handles=r_handles, fontsize=7, frameon=False, loc="lower right")
    fig.text(0.5, -0.03, "Points on each line: b = 10, 20, 50, 100, 200 (left to right). Seed 0, validation split.",
             ha="center", fontsize=7, color=INK)

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig3_lsh{args.tag}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out_dir / f'fig3_lsh{args.tag}.pdf'}")


if __name__ == "__main__":
    main()