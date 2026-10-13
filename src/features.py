"""과거 방향 파생변수.

train과 test를 Time_step 순으로 이어 붙인 뒤, 각 행은 그 행과 이전 행의 설명변수만으로 계산한다.
타깃은 쓰지 않는다. fit이 필요한 변환(결측 대체, 스케일링, 상수 컬럼 제거)은 여기서 하지 않고
모델 쪽에서 학습 fold로만 맞춘다 (src/models.py).
"""

from typing import Callable

import pandas as pd

from src.data import feature_columns

US_INDICES = ["FI_Sp500", "FI_Dow", "FI_Nasdaq"]
# 주말·휴장일에 값이 멈추는 시장 변수 -> 거래일 기준 변화. 거래일 달력은 시장 그룹 단위로 정한다.
MARKET_GROUPS = {
    "FX": ["FI_FX_USDKRW", "FI_FX_JPYKRW", "FI_FX_CNYKRW"],
    "OIL": ["FI_WTI", "FI_Brent", "FI_Dubai", "FI_Gas"],
    "COMMODITY": ["FI_Gold", "FI_Guri", "FI_Nikel", "FI_Oksusu"],
}
# 매일 바뀌거나(국내 유가) 평일에도 드물게 바뀌는(금리) 변수 -> 달력일 기준 변화
CALENDAR_SERIES = ["FI_Gasoline", "FI_Diesel", "FI_CD91", "FI_Corp3yr"]
TRAFFIC_GROUPS = ["KE", "SH1", "SH2"]
FOOD_HEALTH = [
    "SH1_쇼핑1_식품", "SH1_쇼핑1_생활건강",
    "SH2_쇼핑2_면세점_식품건강", "SH2_쇼핑2_생활건강_건강관리용품", "SH2_쇼핑2_생활건강_건강측정용품",
]  # fmt: skip


def trading_days(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """같은 시장 그룹의 변수 중 하나라도 값이 바뀐 날을 거래일로 본다.

    변수 하나만 보면, 실제 거래일인데 값이 우연히 같았던 날을 휴장일로 오인한다
    (예: USDKRW는 다른 원화 환율이 움직인 평일에도 값이 같은 날이 43일 있다).
    """
    return df[cols].diff().ne(0).any(axis=1)


def trading_change(s: pd.Series, k: int, traded: pd.Series) -> pd.Series:
    """직전 k 거래일 대비 변화. traded는 거래일 여부(bool).

    휴장일(주말 등)에는 마지막 거래일의 변화를 이어 쓴다.
    예: 토·일 행에는 금요일의 변화가 들어간다 (단순 1일 차분이면 0이 되어 정보가 사라짐).
    """
    return s[traded].diff(k).reindex(s.index).ffill()


def _group(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df[[c for c in df.columns if c.startswith(prefix + "_")]]


def _us_level(df: pd.DataFrame) -> pd.Series:
    return df[US_INDICES].mean(axis=1)


def _weekday_deviation(s: pd.Series, weeks: int = 4) -> pd.Series:
    """오늘 값 - 지난 weeks주 같은 요일 값의 평균."""
    return s - pd.concat([s.shift(7 * w) for w in range(1, weeks + 1)], axis=1).mean(axis=1, skipna=False)


# ---- 피처 묶음 (입력: 원본 설명변수, 시간순) -------------------------------------------


def raw(df: pd.DataFrame) -> pd.DataFrame:
    """원본 설명변수 그대로 (베이스라인과 같은 입력)."""
    return df.copy()


def weekday(df: pd.DataFrame) -> pd.DataFrame:
    """요일 원-핫 (1=일 ... 7=토)."""
    codes = pd.Categorical(df["Weekday"], categories=range(1, 8))
    return pd.get_dummies(codes, prefix="wd").astype(float).set_index(df.index)


def us_change(df: pd.DataFrame) -> pd.DataFrame:
    """미국 3대 지수 각각의 1·3 거래일 변화."""
    traded = trading_days(df, US_INDICES)
    return pd.DataFrame({f"{c}_tchg{k}": trading_change(df[c], k, traded) for c in US_INDICES for k in (1, 3)})


def us_composite(df: pd.DataFrame) -> pd.DataFrame:
    """미국 3대 지수 평균의 1·3·5 거래일 변화 (개별 지수보다 잡음이 적음)."""
    level, traded = _us_level(df), trading_days(df, US_INDICES)
    return pd.DataFrame({f"US_tchg{k}": trading_change(level, k, traded) for k in (1, 3, 5)})


def us_trend_vol(df: pd.DataFrame) -> pd.DataFrame:
    """미국 지수 평균의 중기 추세(10·20 거래일), 변동성, 20 거래일 고점 대비 하락폭."""
    level = _us_level(df)
    traded = level[trading_days(df, US_INDICES)]
    daily = traded.diff()
    out = pd.DataFrame(
        {
            "US_tchg10": traded.diff(10),
            "US_tchg20": traded.diff(20),
            "US_vol10": daily.rolling(10).std(),
            "US_vol20": daily.rolling(20).std(),
            "US_drawdown20": traded - traded.rolling(20).max(),
        }
    )
    return out.reindex(df.index).ffill()


def macro_change(df: pd.DataFrame) -> pd.DataFrame:
    """환율·유가·원자재(거래일 1·3일)와 국내 유가·금리(달력 1·7일) 변화."""
    out = {
        f"{c}_tchg{k}": trading_change(df[c], k, trading_days(df, cols))
        for cols in MARKET_GROUPS.values()
        for c in cols
        for k in (1, 3)
    }
    out |= {f"{c}_chg{k}": df[c].diff(k) for c in CALENDAR_SERIES for k in (1, 7)}
    return pd.DataFrame(out)


def traffic(df: pd.DataFrame) -> pd.DataFrame:
    """포털·쇼핑 전체 트래픽 흐름: 그룹 내 변수들의 1·3·7일 변화의 중앙값.

    중앙값을 쓰는 이유: 특정 검색어 하나가 급증해도(예: 코로나) 그룹 전체 지표가 끌려가지 않게.
    """
    return pd.DataFrame(
        {f"{p}_med_chg{k}": _group(df, p).diff(k).median(axis=1) for p in TRAFFIC_GROUPS for k in (1, 3, 7)}
    )


def weekday_dev(df: pd.DataFrame) -> pd.DataFrame:
    """같은 요일 평소 대비 이상도 (오늘 - 지난 4주 같은 요일 평균)."""
    series = {f"{p}_med": _group(df, p).median(axis=1) for p in TRAFFIC_GROUPS}
    series["SH2_식품_med"] = _group(df, "SH2_쇼핑2_식품").median(axis=1)
    series |= {c: df[c] for c in FOOD_HEALTH}
    return pd.DataFrame({f"{name}_wddev": _weekday_deviation(s) for name, s in series.items()})


def food_health(df: pd.DataFrame) -> pd.DataFrame:
    """식품·건강 관련 변수(SH2 식품 11개 + 관련 5개)의 1·3·7일 변화."""
    cols = list(_group(df, "SH2_쇼핑2_식품").columns) + FOOD_HEALTH
    return pd.DataFrame({f"{c}_chg{k}": df[c].diff(k) for c in cols for k in (1, 3, 7)})


# ---- 덩어리 4.5 가설 묶음 (docs/EXPERIMENTS.md #7) ----------------------------------------


def us_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """미국 지수 평균의 1거래일 변화 × 요일 (요일마다 정답 기간에 반영되는 정도가 다름)."""
    change = trading_change(_us_level(df), 1, trading_days(df, US_INDICES))
    return weekday(df).mul(change, axis=0).add_prefix("US_tchg1_x_")


US_CHANGE_COLUMNS = [f"{c}_tchg{k}" for c in US_INDICES for k in (1, 3)]  # us_change의 출력 컬럼
US_CORE_COLUMNS = ["FI_Sp500_tchg1", "FI_Sp500_tchg3", "US_tchg20", "US_vol20", "US_drawdown20"]


def us_core(df: pd.DataFrame) -> pd.DataFrame:
    """us_change + us_trend_vol의 축소판 (S&P 1·3일 변화, 20일 추세·변동성·고점 대비)."""
    return pd.concat([us_change(df), us_trend_vol(df)], axis=1)[US_CORE_COLUMNS]


def kr_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """한국 휴장일 추론: 평일(월~금)인데 원화 환율 3종이 모두 정지한 날. 오늘과 어제만 본다 (과거 방향)."""
    weekday_row = df["Weekday"].between(2, 6)
    holiday = (weekday_row & ~trading_days(df, MARKET_GROUPS["FX"])).astype(float)
    return pd.DataFrame({"kr_holiday": holiday, "kr_holiday_prev": holiday.shift(1)})


def monthu_interact(df: pd.DataFrame) -> pd.DataFrame:
    """월·목요일 표시 × 트래픽 지표 3개 (틀리는 날이 월·목에 몰려 있음)."""
    signals = traffic(df)[["KE_med_chg3", "SH1_med_chg1"]].join(weekday_dev(df)[["SH2_med_wddev"]])
    out = {}
    for day, name in [(2, "mon"), (5, "thu")]:
        flag = (df["Weekday"] == day).astype(float)
        for col in signals:
            out[f"{name}_x_{col}"] = signals[col] * flag
    return pd.DataFrame(out)


# ---- 덩어리 4.7 후보 묶음 (docs/EXPERIMENTS.md #10) ----------------------------------------


def us_momentum60(df: pd.DataFrame) -> pd.DataFrame:
    """B2: 미국 지수 평균의 60거래일 변화 (시기별 기저율을 잡는다는 가설)."""
    level = _us_level(df)
    return pd.DataFrame({"US_tchg60": trading_change(level, 60, trading_days(df, US_INDICES))})


def weekday_profile(df: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """B1: 최근 요일 패턴. 지난 weeks주 동안 'D+3과 같은 요일 평균 - D와 같은 요일 평균'.

    t+3-7w (w>=1)은 t-4 이하라 과거 값만 쓴다. 요일 패턴이 시기마다 바뀌면 이 값이 따라 바뀐다.
    """
    proxies = {"SH2_식품_med": _group(df, "SH2_쇼핑2_식품").median(axis=1), "SH1_쇼핑1_식품": df["SH1_쇼핑1_식품"]}
    out = {}
    for name, p in proxies.items():
        target_day = pd.concat([p.shift(7 * w - 3) for w in range(1, weeks + 1)], axis=1).mean(axis=1, skipna=False)
        today = pd.concat([p.shift(7 * w) for w in range(1, weeks + 1)], axis=1).mean(axis=1, skipna=False)
        out[f"{name}_wdprof"] = target_day - today
    return pd.DataFrame(out)


# ---- 덩어리 4.8 후보 묶음 (docs/EXPERIMENTS.md #13) ----------------------------------------

FOOD_L1 = "SH1_쇼핑1_식품"
FOOD_WDDEV_COLUMNS = ["SH1_쇼핑1_식품_wddev", "SH2_식품_med_wddev"]  # G3 평균 회귀 제약 대상 (weekday_dev 출력)
GENERAL_TRAFFIC_COLUMNS = [f"{p}_med_chg{k}" for p in TRAFFIC_GROUPS for k in (1, 3, 7)] + [
    f"{p}_med_wddev" for p in TRAFFIC_GROUPS
]  # G2에서 빼는 일반 트래픽 입력 (traffic 9개 + weekday_dev 중 3개)
US_VOLNORM_COLUMNS = [f"{c}_tchg{k}_vn" for c in US_INDICES for k in (1, 3)]


def food_l1(df: pd.DataFrame) -> pd.DataFrame:
    """G1: 식품 L1(F)의 최근 2주 궤적 + 주기 분해 예측.

    Target_2는 사실상 'F가 3일 뒤 오를까'다 (#12 전제 확인 1). 주기 분해 예측은
    'F[t+3] ≈ 최근 7일 평균 + D+3 요일의 주간 편차(지난 4주 평균)'에서 오늘 값을 뺀 값이다.
    주간 편차는 t+3-7w (w>=1), 즉 t-4 이하 날짜의 것만 쓴다.
    """
    f = df[FOOD_L1]
    out = {f"F_lag{k}_minus_now": f.shift(k) - f for k in range(1, 15)}
    level = f.rolling(7).mean()
    weekly_dev = f - level
    season_target = pd.concat([weekly_dev.shift(7 * w - 3) for w in range(1, 5)], axis=1).mean(axis=1, skipna=False)
    out["F_decomp_chg3"] = level + season_target - f
    return pd.DataFrame(out)


def us_volnorm(df: pd.DataFrame) -> pd.DataFrame:
    """G4: 미국 지수 1·3거래일 변화 ÷ 최근 20거래일 변동성 (같은 변화도 잔잔한 시기엔 더 큰 신호)."""
    traded = trading_days(df, US_INDICES)
    out = {}
    for c in US_INDICES:
        s = df[c][traded]
        vol = s.diff().rolling(20).std()
        vol = vol.where(vol > 0)
        for k in (1, 3):
            out[f"{c}_tchg{k}_vn"] = (s.diff(k) / vol).reindex(df.index).ffill()
    return pd.DataFrame(out)


FAMILIES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    f.__name__: f
    for f in (
        raw, weekday, us_change, us_composite, us_trend_vol, macro_change, traffic, weekday_dev, food_health,
        us_weekday, us_core, kr_holiday, monthu_interact, us_momentum60, weekday_profile, food_l1, us_volnorm,
    )
}  # fmt: skip


def build_from_full(full: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """시간순으로 이어 붙인 원본 설명변수(full)에서 피처 묶음들을 만든다."""
    parts = [FAMILIES[name](full) for name in families]
    X = pd.concat(parts, axis=1)
    if X.columns.duplicated().any():
        raise ValueError(f"피처 이름 중복: {X.columns[X.columns.duplicated()].tolist()}")
    return X


def build(train: pd.DataFrame, test: pd.DataFrame, families: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(X_train, X_test). train과 test는 load_data()가 시간 연속성을 검사한 상태여야 한다."""
    cols = feature_columns(train)
    full = pd.concat([train[cols], test[cols]], ignore_index=True)
    if full.isna().any().any():  # 거래일 변화의 ffill이 결측을 오래된 값으로 덮어쓰지 않도록
        raise ValueError("설명변수에 결측이 있습니다. 피처 계산 전에 결측 처리 방식을 정해야 합니다.")
    X = build_from_full(full, families)
    return X.iloc[: len(train)].reset_index(drop=True), X.iloc[len(train) :].reset_index(drop=True)
