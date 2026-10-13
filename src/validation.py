"""시계열 교차검증 하네스 (자체 채점기).

D+3 라벨은 인접한 날끼리 기간이 겹친다 (t일 라벨은 t~t+3일 구간을 본다).
학습 구간 끝과 검증 구간 시작 사이를 GAP일 비워, 학습 라벨이 검증 기간의 정보를 담지 않게 한다.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.data import TARGETS

N_SPLITS = 5
FOLD_SIZE = 100  # Public 평가 기간과 같은 길이
GAP = 3
HOLDOUT_SIZE = 200  # Private 평가 기간과 같은 길이

# (학습 X, 학습 y, 예측할 X) -> 1일 확률
FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]
Split = tuple[np.ndarray, np.ndarray]


def cv_splits(n: int) -> list[Split]:
    """확장 윈도우 5-fold. 각 fold는 직전까지의 데이터로 학습하고 다음 100일을 검증한다."""
    return list(TimeSeriesSplit(n_splits=N_SPLITS, test_size=FOLD_SIZE, gap=GAP).split(np.arange(n)))


def holdout_split(n: int) -> list[Split]:
    """마지막 200일 검증 (Private 길이 모사)."""
    valid = np.arange(n - HOLDOUT_SIZE, n)
    train = np.arange(0, n - HOLDOUT_SIZE - GAP)
    return [(train, valid)]


def evaluate(fit_predict: FitPredict, X: pd.DataFrame, y: pd.DataFrame, splits: list[Split]) -> pd.DataFrame:
    """타깃 × fold별 AUC 표를 반환한다. X, y는 시간 오름차순이어야 한다."""
    rows = []
    for target in TARGETS:
        for fold, (tr, va) in enumerate(splits, start=1):
            pred = np.asarray(fit_predict(X.iloc[tr], y[target].iloc[tr], X.iloc[va]), dtype=float)
            if pred.shape != (len(va),) or not np.isfinite(pred).all():
                raise ValueError(f"{target} fold {fold}: 예측값 형태가 잘못되었거나 유한하지 않습니다.")
            rows.append({"target": target, "fold": fold, "auc": roc_auc_score(y[target].iloc[va], pred)})
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """타깃별 평균·표준편차·최소·최대와, 두 타깃 평균(정량 평가 기준)을 반환한다."""
    summary = scores.groupby("target")["auc"].agg(["mean", "std", "min", "max"])
    summary.loc["평균(T1,T2)"] = [summary["mean"].mean(), np.nan, np.nan, np.nan]
    return summary
