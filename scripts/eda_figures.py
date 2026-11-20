"""보고서 데이터 장용 EDA 그림과 수치 → results/figures/fig_data.{pdf,png}, results/eda_stats.json.

전처리(사용자 필터) 후 전체 데이터 기준의 기술 통계다. 분할 전 데이터이므로 모델 선택에는 쓰지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lastfm_rec.data import encode_side_info, load_friends, load_tags, load_user_artists, preprocess  # noqa: E402
from lastfm_rec.metrics import head_items, item_popularity  # noqa: E402

COLOR = "#2a78d6"
ACCENT = "#eb6834"
INK = "#52514e"


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(float))
    n = len(x)
    coef = 2 * np.arange(1, n + 1, dtype=float) - n - 1
    return float((coef * x).sum() / (n * x.sum()))


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
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


def main() -> None:
    ua = load_user_artists()
    enc, _ = preprocess(ua)
    inter = enc.interactions
    friends, _, _ = encode_side_info(enc, load_friends(), load_tags())

    pop = item_popularity(inter, enc.n_items)
    head = head_items(pop)
    per_user = inter.groupby("u").size().to_numpy()
    plays = np.rint(np.expm1(inter["w"].to_numpy()))
    deg = np.bincount(np.concatenate([friends["u"], friends["v"]]), minlength=enc.n_users)

    stats = {
        "n_head_items": int(head.sum()),
        "head_share_of_pairs": float(pop[head].sum() / pop.sum()),
        "popularity_gini": gini(pop),
        "popularity_max": int(pop.max()),
        "popularity_median": float(np.median(pop)),
        "items_per_user": {"mean": float(per_user.mean()), "median": float(np.median(per_user)),
                           "min": int(per_user.min()), "max": int(per_user.max()),
                           "share_at_max": float((per_user == per_user.max()).mean())},
        "plays": {"median": float(np.median(plays)), "mean": float(plays.mean()), "max": int(plays.max())},
        "friends_per_user": {"mean": float(deg.mean()), "median": float(np.median(deg)),
                             "max": int(deg.max()), "share_zero": float((deg == 0).mean())},
        "density": float(len(inter) / (enc.n_users * enc.n_items)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    ax = axes[0, 0]
    ranked = np.sort(pop)[::-1]
    rank = np.arange(1, len(ranked) + 1)
    ax.loglog(rank, ranked, color=COLOR, linewidth=2)
    ax.axvline(stats["n_head_items"], color=ACCENT, linewidth=1, linestyle="--")
    ax.text(stats["n_head_items"] * 1.3, ranked[0] * 0.6,
            f"head: top {stats['n_head_items']} artists\n= {stats['head_share_of_pairs']:.0%} of pairs",
            fontsize=8, color=INK, va="top")
    style(ax, "(a) Artist popularity (long tail)", "Artist rank", "Listeners")

    ax = axes[0, 1]
    # 97%가 상한 50에 몰려 있어 로그 축으로 나머지 분포도 보이게 한다
    ax.hist(per_user, bins=np.arange(0.5, per_user.max() + 1.5), color=COLOR, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.text(per_user.max() - 2, ax.get_ylim()[1] * 0.5,
            f"{stats['items_per_user']['share_at_max']:.0%} of users\nat cap {per_user.max()}",
            fontsize=8, color=INK, ha="right", va="top")
    style(ax, "(b) Artists per user (log scale)", "Artists listened", "Users")

    ax = axes[1, 0]
    ax.hist(np.log10(plays), bins=50, color=COLOR, edgecolor="white", linewidth=0.4)
    ax.axvline(np.log10(stats["plays"]["median"]), color=ACCENT, linewidth=1, linestyle="--")
    ax.text(0.97, 0.95, f"median {stats['plays']['median']:.0f} plays\nmax {stats['plays']['max']:,}",
            transform=ax.transAxes, fontsize=8, color=INK, ha="right", va="top")
    style(ax, "(c) Play counts per user-artist pair", "log10(plays)", "Pairs")

    ax = axes[1, 1]
    ax.hist(deg, bins=np.arange(-0.5, min(deg.max(), 120) + 1.5, 2), color=COLOR, edgecolor="white", linewidth=0.4)
    style(ax, "(d) Friends per user", "Friends", "Users")

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_data.{ext}", dpi=200)
    (ROOT / "results" / "eda_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
