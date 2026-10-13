"""측정용(probe) 모델.

덩어리 3에서는 모델 설정을 고정하고 입력 피처만 바꿔 비교한다.
fit이 필요한 전처리(상수 컬럼 제거, 결측 대체, 스케일링)는 모두 학습 fold로만 맞춘다.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.validation import FitPredict

LOGISTIC_C = 0.1
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def _non_constant(X_train: pd.DataFrame) -> pd.Index:
    return X_train.columns[X_train.nunique(dropna=False) > 1]


def logistic(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
    keep = _non_constant(X_train)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=LOGISTIC_C, max_iter=5000)
    )
    return model.fit(X_train[keep], y_train).predict_proba(X_pred[keep])[:, 1]


def _positional(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM은 일부 문자가 든 컬럼명을 거부하므로 위치 기반 이름(f0, f1, ...)으로 바꾼다."""
    return X.set_axis([f"f{i}" for i in range(X.shape[1])], axis=1)


def lgbm(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
    keep = _non_constant(X_train)
    model = LGBMClassifier(**LGBM_PARAMS).fit(_positional(X_train[keep]), y_train)  # 결측은 LightGBM이 직접 처리
    return model.predict_proba(_positional(X_pred[keep]))[:, 1]


PROBES: dict[str, FitPredict] = {"logistic": logistic, "lgbm": lgbm}


def probe_ensemble(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
    """측정 모델들의 순위 평균 (0~1). AUC는 순위만 보므로, 확률 크기가 다른 모델을 공평하게 섞는다."""
    ranks = [rankdata(fp(X_train, y_train, X_pred)) for fp in PROBES.values()]
    return np.mean(ranks, axis=0) / len(X_pred)
