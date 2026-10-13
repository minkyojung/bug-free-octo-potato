import numpy as np
import pandas as pd
import pytest

from src.data import N_TRAIN, feature_columns
from src.features import FAMILIES, build, build_from_full, trading_change, trading_days


@pytest.fixture(scope="module")
def full(data):
    train, test = data
    cols = feature_columns(train)
    return pd.concat([train[cols], test[cols]], ignore_index=True)


@pytest.mark.parametrize("family", list(FAMILIES))
# 모든 요일(특히 주말)과 train/test 경계를 지나는 지점: 주말 값을 미래에서 끌어오는 버그(bfill 등)도 잡는다
@pytest.mark.parametrize("cut", [100, *range(N_TRAIN - 7, N_TRAIN + 7)])
def test_no_future_dependence(full, family, cut):
    """cut 이후 행을 무작위로 바꿔도 cut 이하 행의 피처는 변하지 않아야 한다."""
    before = build_from_full(full, [family])
    scrambled = full.copy()
    rng = np.random.default_rng(cut)
    numeric = [c for c in full.columns if c != "Weekday"]
    scrambled.loc[cut + 1 :, numeric] = rng.random((len(full) - cut - 1, len(numeric)))
    after = build_from_full(scrambled, [family])
    pd.testing.assert_frame_equal(before.iloc[: cut + 1], after.iloc[: cut + 1])


def test_trading_change_carries_last_trading_day_over_weekend():
    # 월~금 거래, 토·일은 금요일 값 유지
    s = pd.Series([10.0, 11, 13, 12, 15, 15, 15, 16])  # 월 화 수 목 금 토 일 월
    traded = trading_days(s.to_frame(), [0])
    np.testing.assert_allclose(trading_change(s, 1, traded).to_numpy(), [np.nan, 1, 2, -1, 3, 3, 3, 1])
    np.testing.assert_allclose(trading_change(s, 3, traded).to_numpy(), [np.nan, np.nan, np.nan, 2, 4, 4, 4, 3])


def test_trading_day_with_unchanged_value_counts_as_zero_change():
    # 화요일에 a는 그대로지만 같은 시장의 b가 움직였으므로 거래일 -> a의 변화는 0
    df = pd.DataFrame({"a": [10.0, 10, 12, 12], "b": [5.0, 6, 7, 7]})  # 월 화 수 목(휴장)
    traded = trading_days(df, ["a", "b"])
    assert traded.tolist() == [True, True, True, False]
    np.testing.assert_allclose(trading_change(df["a"], 1, traded).to_numpy(), [np.nan, 0, 2, 2])


def test_build_rejects_missing_values(data):
    train, test = data
    broken = test.copy()
    broken.loc[5, "FI_Dow"] = np.nan
    with pytest.raises(ValueError):
        build(train, broken, ["us_change"])


def test_build_splits_rows_and_keeps_columns(data):
    train, test = data
    X_train, X_test = build(train, test, list(FAMILIES))
    assert len(X_train) == len(train) and len(X_test) == len(test)
    assert list(X_train.columns) == list(X_test.columns)
    # test 첫 행도 train 마지막 행들을 이용해 정상 계산됨 (warm-up NaN 없음)
    assert X_test.isna().sum().sum() == 0


def test_weekday_onehot_has_all_days(data):
    train, test = data
    X_train, _ = build(train, test, ["weekday"])
    assert list(X_train.columns) == [f"wd_{d}" for d in range(1, 8)]
    assert (X_train.sum(axis=1) == 1).all()
