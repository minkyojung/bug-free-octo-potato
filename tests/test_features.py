import numpy as np
import pandas as pd
import pytest

from src.data import N_TRAIN, feature_columns
from src.features import (
    FAMILIES,
    FOOD_WDDEV_COLUMNS,
    GENERAL_TRAFFIC_COLUMNS,
    US_CHANGE_COLUMNS,
    US_CORE_COLUMNS,
    US_VOLNORM_COLUMNS,
    build,
    build_from_full,
    trading_change,
    trading_days,
)


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
    families = [f for f in FAMILIES if f != "us_core"]  # us_core는 us_change+us_trend_vol의 교체용 (같은 이름)
    X_train, X_test = build(train, test, families)
    assert len(X_train) == len(train) and len(X_test) == len(test)
    assert list(X_train.columns) == list(X_test.columns)
    # test 첫 행도 train 마지막 행들을 이용해 정상 계산됨 (warm-up NaN 없음)
    assert X_test.isna().sum().sum() == 0


def test_weekday_onehot_has_all_days(data):
    train, test = data
    X_train, _ = build(train, test, ["weekday"])
    assert list(X_train.columns) == [f"wd_{d}" for d in range(1, 8)]
    assert (X_train.sum(axis=1) == 1).all()


def test_kr_holiday_marks_weekday_fx_freeze(data):
    train, test = data
    X_train, _ = build(train, test, ["kr_holiday"])
    holiday = X_train["kr_holiday"]
    assert holiday.isin([0, 1]).all()
    assert (holiday[train["Weekday"].isin([1, 7])] == 0).all()  # 주말은 휴장일로 세지 않음
    assert 5 <= holiday.sum() <= 60  # 2.5년치 한국 공휴일 수의 상식적 범위
    assert (X_train["kr_holiday_prev"].iloc[1:].to_numpy() == holiday.iloc[:-1].to_numpy()).all()


def test_us_core_is_subset_of_full_us_features(data):
    train, test = data
    core, _ = build(train, test, ["us_core"])
    full, _ = build(train, test, ["us_change", "us_trend_vol"])
    pd.testing.assert_frame_equal(core, full[core.columns])


def test_weekday_profile_uses_matching_weekdays():
    # 요일별로 다른 상수값(요일 패턴)만 있는 시계열 -> 결과는 정확히 "D+3 요일 값 - D 요일 값"
    days = np.arange(70)
    pattern = np.array([5.0, 1, 1, 1, 2, 8, 9])  # 요일 위치 0..6의 값
    df = pd.DataFrame({"Weekday": days % 7 + 1, "SH1_쇼핑1_식품": pattern[days % 7]})
    df["SH2_쇼핑2_식품_a"] = df["SH1_쇼핑1_식품"]
    out = FAMILIES["weekday_profile"](df)
    t = 40
    expected = pattern[(t + 3) % 7] - pattern[t % 7]
    assert out.loc[t, "SH1_쇼핑1_식품_wdprof"] == pytest.approx(expected)
    assert out.iloc[:28].isna().all().all()  # 4주치가 모이기 전에는 계산하지 않음


def test_monotone_target_columns_exist(data):
    """단조 제약 대상 이름이 실제 피처 이름과 어긋나면 제약이 조용히 0이 되므로 이름을 고정 확인한다."""
    train, test = data
    X_change, _ = build(train, test, ["us_change"])
    assert list(X_change.columns) == US_CHANGE_COLUMNS
    X_core, _ = build(train, test, ["us_core"])
    assert list(X_core.columns) == US_CORE_COLUMNS


def test_screening_target_columns_exist(data):
    """G2·G3·G4가 겨냥하는 컬럼 이름이 실제 피처 이름과 일치해야 한다 (어긋나면 조용히 무효가 됨)."""
    train, test = data
    X2, _ = build(train, test, ["weekday", "traffic", "weekday_dev"])
    assert set(FOOD_WDDEV_COLUMNS) <= set(X2.columns)
    assert set(GENERAL_TRAFFIC_COLUMNS) <= set(X2.columns) and len(GENERAL_TRAFFIC_COLUMNS) == 12
    X1, _ = build(train, test, ["us_volnorm"])
    assert list(X1.columns) == US_VOLNORM_COLUMNS


def test_food_decomposition_recovers_pure_weekly_pattern():
    # 요일 패턴만 있는 시계열이면 주기 분해 예측 = 정확히 F[t+3] - F[t]
    days = np.arange(80)
    pattern = np.array([5.0, 1, 1, 1, 2, 8, 9])
    df = pd.DataFrame({"SH1_쇼핑1_식품": pattern[days % 7] + 0.0})
    out = FAMILIES["food_l1"](df)
    t = 50
    assert out.loc[t, "F_decomp_chg3"] == pytest.approx(pattern[(t + 3) % 7] - pattern[t % 7])
    assert out.loc[t, "F_lag7_minus_now"] == pytest.approx(0.0)
