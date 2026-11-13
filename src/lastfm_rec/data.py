"""HetRec 2011 Last.fm 2K 로딩과 전처리."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "hetrec2011-lastfm-2k"


def _read(path: Path) -> pd.DataFrame:
    # 탭 구분, latin-1 (일부 아티스트·태그 이름에 비 UTF-8 문자 포함)
    return pd.read_csv(path, sep="\t", encoding="latin-1")


def load_user_artists(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = _read(raw_dir / "user_artists.dat")
    return df.rename(columns={"userID": "user", "artistID": "item", "weight": "plays"})


def load_friends(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = _read(raw_dir / "user_friends.dat")
    return df.rename(columns={"userID": "user", "friendID": "friend"})


def load_tags(raw_dir: Path = RAW_DIR, min_year: int = 2005, max_year: int = 2011) -> pd.DataFrame:
    """user_taggedartists.dat. 연도가 비정상적인 행은 제거하고 개수를 attrs에 남긴다."""
    df = _read(raw_dir / "user_taggedartists.dat")
    df = df.rename(columns={"userID": "user", "artistID": "item", "tagID": "tag"})
    bad = ~df["year"].between(min_year, max_year)
    out = df.loc[~bad, ["user", "item", "tag"]].reset_index(drop=True)
    out.attrs["dropped_bad_year"] = int(bad.sum())
    return out


def load_artist_names(raw_dir: Path = RAW_DIR) -> dict[int, str]:
    df = _read(raw_dir / "artists.dat")
    return dict(zip(df["id"], df["name"]))


@dataclass
class Encoded:
    """원본 ID ↔ 0부터 시작하는 연속 인덱스."""
    interactions: pd.DataFrame  # columns: u, i, w (w = log1p(plays))
    user_ids: np.ndarray        # index -> 원본 userID
    item_ids: np.ndarray        # index -> 원본 artistID

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def encode_users(self, s: pd.Series) -> pd.Series:
        return s.map(pd.Series(np.arange(self.n_users), index=self.user_ids))

    def encode_items(self, s: pd.Series) -> pd.Series:
        return s.map(pd.Series(np.arange(self.n_items), index=self.item_ids))


def encode_side_info(enc: "Encoded", friends: pd.DataFrame, tags: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """친구(u, v)와 태그(u, i, t)를 인코딩된 인덱스로 바꾼다. 필터로 빠진 사용자·아티스트 행은 버린다."""
    f = pd.DataFrame({"u": enc.encode_users(friends["user"]), "v": enc.encode_users(friends["friend"])})
    f = f.dropna().astype(int)
    f = f[f["u"] < f["v"]].drop_duplicates()  # 양방향 중복 제거

    tag_ids = np.sort(tags["tag"].unique())
    t = pd.DataFrame({
        "u": enc.encode_users(tags["user"]),
        "i": enc.encode_items(tags["item"]),
        "t": tags["tag"].map(pd.Series(np.arange(len(tag_ids)), index=tag_ids)),
    }).dropna().astype(int)
    return f.reset_index(drop=True), t.reset_index(drop=True), len(tag_ids)


def preprocess(ua: pd.DataFrame, min_items_per_user: int = 5) -> tuple[Encoded, dict]:
    """log(1 + plays) 변환, 아티스트 수가 적은 사용자 제거, 인덱스 인코딩.

    반환값 stats는 보고서 데이터 장의 필터 전후 표에 그대로 쓴다.
    """
    stats = {
        "before": {"users": ua["user"].nunique(), "items": ua["item"].nunique(), "pairs": len(ua)},
        "max_items_per_user": int(ua.groupby("user").size().max()),
    }
    counts = ua.groupby("user")["item"].transform("size")
    ua = ua[counts >= min_items_per_user]

    user_ids = np.sort(ua["user"].unique())
    item_ids = np.sort(ua["item"].unique())
    u = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i = pd.Series(np.arange(len(item_ids)), index=item_ids)
    inter = pd.DataFrame({
        "u": u.loc[ua["user"]].to_numpy(),
        "i": i.loc[ua["item"]].to_numpy(),
        "w": np.log1p(ua["plays"].to_numpy(dtype=float)),
    })
    stats["after"] = {"users": len(user_ids), "items": len(item_ids), "pairs": len(inter)}
    return Encoded(inter, user_ids, item_ids), stats
