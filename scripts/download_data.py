"""HetRec 2011 Last.fm 2K를 data/raw/에 내려받고 압축을 푼다.

원본은 비상업적 라이선스라 repo에 올리지 않는다. 접근일은 data/raw/ACCESS_DATE.txt에 기록된다.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = ["user_artists.dat", "user_friends.dat", "user_taggedartists.dat", "artists.dat", "tags.dat"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "hetrec2011-lastfm-2k.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    out = RAW / "hetrec2011-lastfm-2k"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    missing = [f for f in EXPECTED if not (out / f).exists()]
    if missing:
        raise SystemExit(f"압축 안에 없는 파일: {missing}")
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (RAW / "ACCESS_DATE.txt").write_text(f"{dt.date.today().isoformat()}\nsha256 {sha}\n{URL}\n")
    print(f"ok: {out}\nsha256 {sha}")


if __name__ == "__main__":
    main()
