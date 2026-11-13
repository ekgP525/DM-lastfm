"""보고서 데이터 장용 기초 통계: 필터 전후 규모, 사용자당 최대 아티스트 수, 롱테일, 맞힐 수 없는 테스트 쌍 비율."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lastfm_rec.data import load_tags, load_user_artists, preprocess  # noqa: E402
from lastfm_rec.metrics import head_items, item_popularity, unreachable_fraction  # noqa: E402
from lastfm_rec.split import split_per_user  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    enc, stats = preprocess(load_user_artists())
    pop_all = item_popularity(enc.interactions, enc.n_items)
    stats["items_with_1_listener"] = float((pop_all == 1).mean())
    stats["head_items_share_of_catalog"] = float(head_items(pop_all).mean())
    stats["tags_dropped_bad_year"] = load_tags().attrs["dropped_bad_year"]

    unreach = []
    for s in SEEDS:
        sp = split_per_user(enc.interactions, seed=s)
        unreach.append(unreachable_fraction(sp.fit_set("test"), sp.test, enc.n_items))
    stats["test_pairs_unreachable"] = {"mean": float(np.mean(unreach)), "std": float(np.std(unreach))}

    out = ROOT / "results" / "data_stats.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
