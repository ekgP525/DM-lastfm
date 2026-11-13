from .base import Recommender
from .baselines import Popularity, RandomRec
from .stubs import MF, RWR, ItemCF, MinHashLSH

REGISTRY = {
    "popularity": lambda seed, **kw: Popularity(),
    "random": lambda seed, **kw: RandomRec(seed=seed),
    "itemcf": lambda seed, **kw: ItemCF(**kw),
    "mf": lambda seed, **kw: MF(seed=seed, **kw),
    "rwr": lambda seed, **kw: RWR(**kw),
}

__all__ = ["Recommender", "Popularity", "RandomRec", "ItemCF", "MinHashLSH", "MF", "RWR", "REGISTRY"]
