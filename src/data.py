"""제공 데이터 로딩과 무결성 검사."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ENCODING = "cp949"

KEY = "Time_step"
TARGETS = ["Target_1", "Target_2"]
N_TRAIN, N_PUBLIC, N_PRIVATE = 924, 100, 200


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name, encoding=ENCODING)


def _check_time_axis(df: pd.DataFrame, name: str, start: int, n: int) -> None:
    expected = np.arange(start, start + n)
    if len(df) != n or not np.array_equal(df[KEY].to_numpy(), expected):
        raise ValueError(f"{name}: Time_step이 {start}~{start + n - 1} 연속 오름차순이 아닙니다.")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test)를 반환한다.

    test는 Private 평가용 200행(Time_step 925~1124)이며 Target 열은 제거한다.
    Public 평가용 100행은 test의 앞 100행과 같음을 확인했으므로 따로 반환하지 않는다.
    """
    train = _read("train.csv")
    private = _read("test_private.csv")
    public = _read("test_public.csv")

    if not (list(train.columns) == list(private.columns) == list(public.columns)):
        raise ValueError("train/test_public/test_private의 컬럼 구성이 다릅니다.")
    _check_time_axis(train, "train.csv", 1, N_TRAIN)
    _check_time_axis(private, "test_private.csv", N_TRAIN + 1, N_PRIVATE)
    _check_time_axis(public, "test_public.csv", N_TRAIN + 1, N_PUBLIC)

    for t in TARGETS:
        if not train[t].isin([0, 1]).all():
            raise ValueError(f"train.csv: {t}에 0/1 이외의 값이 있습니다.")
    if private[TARGETS].notna().any().any():
        raise ValueError("test_private.csv에 Target 값이 들어 있습니다.")

    features = feature_columns(train)
    if not np.allclose(
        public[features].to_numpy(float),
        private[features].head(N_PUBLIC).to_numpy(float),
        equal_nan=True,
    ):
        raise ValueError("test_public.csv가 test_private.csv의 앞 100행과 다릅니다.")

    train[TARGETS] = train[TARGETS].astype(int)
    return train, private.drop(columns=TARGETS)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Time_step과 Target을 제외한 설명변수 컬럼 (원본 순서 유지)."""
    return [c for c in df.columns if c != KEY and c not in TARGETS]
