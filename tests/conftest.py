import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def toy():
    """롱테일이 있는 작은 합성 데이터: 사용자 60명, 아티스트 200개."""
    rng = np.random.default_rng(42)
    n_users, n_items = 60, 200
    p = 1.0 / np.arange(1, n_items + 1) ** 1.1
    p /= p.sum()
    rows = []
    for u in range(n_users):
        k = rng.integers(5, 40)
        for i in rng.choice(n_items, size=k, replace=False, p=p):
            rows.append((u, int(i), float(np.log1p(rng.integers(1, 500)))))
    inter = pd.DataFrame(rows, columns=["u", "i", "w"])
    return inter, n_users, n_items
