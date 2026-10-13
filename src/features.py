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


FAMILIES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    f.__name__: f
    for f in (raw, weekday, us_change, us_composite, us_trend_vol, macro_change, traffic, weekday_dev, food_health)
}


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
