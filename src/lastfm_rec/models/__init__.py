from .base import Recommender
from .baselines import Popularity, RandomRec
from .stubs import MF, P3, RWR, ItemCF, MinHashLSH

REGISTRY = {
    "popularity": lambda seed, **kw: Popularity(),
    "random": lambda seed, **kw: RandomRec(seed=seed),
    "itemcf": lambda seed, **kw: ItemCF(**kw),
    # MinHash-LSH 근사 이웃 Item-CF.
    "itemcf_lsh": lambda seed, b=20, r=5, **kw: ItemCF(neighbor_fn=MinHashLSH(b=b, r=r, seed=seed).candidates, **kw),
    "mf": lambda seed, **kw: MF(seed=seed, **kw),
    "rwr": lambda seed, **kw: RWR(**kw),
    "p3": lambda seed, **kw: P3(**kw),
}

__all__ = ["Recommender", "Popularity", "RandomRec", "ItemCF", "MinHashLSH", "MF", "RWR", "P3",
           "REGISTRY"]
