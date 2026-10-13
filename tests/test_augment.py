import numpy as np
import pandas as pd

from src.augment import HORIZON, sibling_labels


def test_sibling_labels_stay_inside_fold():
    s = pd.Series(np.arange(20, dtype=float) ** 2)  # 단조 증가
    rows = pd.RangeIndex(5, 15)
    lab = sibling_labels(s, rows)
    assert list(lab.index) == list(rows)
    assert lab.loc[5 : 14 - HORIZON].eq(1).all()  # D+3이 fold 안이면 상승
    assert lab.loc[14 - HORIZON + 1 :].isna().all()  # D+3이 fold 밖이면 제외


def test_sibling_labels_direction():
    s = pd.Series([5.0, 4, 3, 2, 1, 9, 9, 9])
    lab = sibling_labels(s, pd.RangeIndex(0, 8))
    assert lab.iloc[0] == 0 and lab.iloc[2] == 1  # 5→2 하락, 3→9 상승
