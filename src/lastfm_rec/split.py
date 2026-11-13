"""사용자별 train / validation / test 분할."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    seed: int

    def fit_set(self, stage: str) -> pd.DataFrame:
        """stage='val': train만으로 학습(하이퍼파라미터 선택).
        stage='test': train + val로 학습(최종 평가). 하이퍼파라미터는 val 단계에서 고정한다."""
        if stage == "val":
            return self.train
        if stage == "test":
            return pd.concat([self.train, self.val], ignore_index=True)
        raise ValueError(stage)

    def eval_set(self, stage: str) -> pd.DataFrame:
        return {"val": self.val, "test": self.test}[stage]


def split_per_user(inter: pd.DataFrame, seed: int, test_frac: float = 0.2,
                   val_frac: float = 0.1) -> Split:
    """사용자마다 들은 아티스트의 test_frac을 test로, 나머지의 val_frac을 validation으로 뗀다.

    각 사용자에게 train이 최소 1개 남도록 test·val 개수를 제한한다.
    """
    rng = np.random.default_rng(seed)
    df = inter.copy()
    df["_r"] = rng.random(len(df))
    df["_rank"] = df.groupby("u")["_r"].rank(method="first").astype(int) - 1
    n = df.groupby("u")["i"].transform("size")

    n_test = np.maximum(1, np.round(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.round((n - n_test) * val_frac)).astype(int)
    n_val = np.minimum(n_val, n - n_test - 1).clip(lower=0)

    is_test = df["_rank"] < n_test
    is_val = (~is_test) & (df["_rank"] < n_test + n_val)
    cols = ["u", "i", "w"]
    return Split(
        train=df.loc[~is_test & ~is_val, cols].reset_index(drop=True),
        val=df.loc[is_val, cols].reset_index(drop=True),
        test=df.loc[is_test, cols].reset_index(drop=True),
        seed=seed,
    )
