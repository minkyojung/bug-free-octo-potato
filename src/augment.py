"""덩어리 4.6 지렛대: 형제 정답 증강과 최근 구간 학습 (docs/EXPERIMENTS.md #8).

형제 정답 증강: 학습 fold 안에서 형제 변수 s의 라벨 "s[t+3] > s[t]"를 만들어 추가 학습 예제로 쓴다.
피처는 원래 타깃과 공유한다. t+3이 학습 fold 밖이면 그 예제는 버린다 (fold 경계 밖 정보 사용 금지).
"""

import numpy as np
import pandas as pd

from src.features import FOOD_HEALTH
from src.models import lgbm, logistic, rank_average
from src.validation import FitPredict

HORIZON = 3


def sibling_columns(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c.startswith("SH2_쇼핑2_식품_")] + FOOD_HEALTH


def sibling_labels(series: pd.Series, rows: pd.Index) -> pd.Series:
    """rows(학습 fold의 행 번호) 안에서 D+3 상승 라벨. D+3이 rows 밖이면 NaN."""
    inside = (rows + HORIZON).isin(rows)  # 학습 fold 안의 행만 미래로 쓴다
    future = series.reindex(rows + HORIZON).to_numpy()
    label = np.where(inside, (future > series.loc[rows].to_numpy()).astype(float), np.nan)
    return pd.Series(label, index=rows)


def make_augmented(train: pd.DataFrame, target_weight: float) -> FitPredict:
    """logistic + lgbm 순위 평균. 학습 시 형제 라벨 예제를 덧붙이고 원래 타깃 예제에 target_weight를 준다."""
    siblings = train[sibling_columns(train)]

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        rows = X_train.index
        parts_X, parts_y, parts_w = [X_train], [y_train.to_numpy(float)], [np.full(len(rows), target_weight)]
        for col in siblings:
            lab = sibling_labels(siblings[col], rows).dropna()
            parts_X.append(X_train.loc[lab.index])
            parts_y.append(lab.to_numpy())
            parts_w.append(np.ones(len(lab)))
        X_aug = pd.concat(parts_X, ignore_index=True)
        y_aug = pd.Series(np.concatenate(parts_y))
        w_aug = np.concatenate(parts_w)
        return rank_average([m(X_aug, y_aug, X_pred, sample_weight=w_aug) for m in (logistic, lgbm)])

    return fit_predict


def make_recent(window: int) -> FitPredict:
    """학습 fold의 최근 window일만으로 logistic + lgbm 순위 평균을 학습한다."""

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        X_r, y_r = X_train.iloc[-window:], y_train.iloc[-window:]
        return rank_average([m(X_r, y_r, X_pred) for m in (logistic, lgbm)])

    return fit_predict
