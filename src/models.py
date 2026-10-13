"""모델 정의.

모든 모델은 FitPredict(학습 X, 학습 y, 예측할 X -> 1일 확률) 형태다.
fit이 필요한 전처리(상수 컬럼 제거, 결측 대체, 스케일링)는 모두 학습 fold로만 맞춘다.
"""

from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.validation import FitPredict

SEED = 42
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
    random_state=SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=1,
)
FOREST_PARAMS = dict(n_estimators=500, max_depth=6, min_samples_leaf=10, max_features="sqrt", random_state=SEED, n_jobs=1)


def _non_constant(X_train: pd.DataFrame) -> pd.Index:
    return X_train.columns[X_train.nunique(dropna=False) > 1]


def _positional(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM은 일부 문자가 든 컬럼명을 거부하므로 위치 기반 이름(f0, f1, ...)으로 바꾼다."""
    return X.set_axis([f"f{i}" for i in range(X.shape[1])], axis=1)


def make_logistic(C: float = LOGISTIC_C) -> FitPredict:
    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame, sample_weight=None) -> np.ndarray:
        keep = _non_constant(X_train)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        model.fit(X_train[keep], y_train, logisticregression__sample_weight=sample_weight)
        return model.predict_proba(X_pred[keep])[:, 1]

    return fit_predict


def _tree(build_model: Callable[[], object], impute: bool) -> FitPredict:
    """트리 모델 공통 래퍼. impute=False면 결측을 모델이 직접 처리한다 (LightGBM, XGBoost)."""

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame, sample_weight=None) -> np.ndarray:
        keep = _non_constant(X_train)
        model = build_model()
        fit_kwargs = {"sample_weight": sample_weight}
        if impute:
            model = make_pipeline(SimpleImputer(strategy="median"), model)
            fit_kwargs = {f"{model.steps[-1][0]}__sample_weight": sample_weight}
        model.fit(_positional(X_train[keep]), y_train, **fit_kwargs)
        return model.predict_proba(_positional(X_pred[keep]))[:, 1]

    return fit_predict


def make_lgbm(**overrides) -> FitPredict:
    params = LGBM_PARAMS | overrides
    return _tree(lambda: LGBMClassifier(**params), impute=False)


def make_xgb(**overrides) -> FitPredict:
    params = XGB_PARAMS | overrides
    return _tree(lambda: XGBClassifier(**params), impute=False)


def make_rf(**overrides) -> FitPredict:
    params = FOREST_PARAMS | overrides
    return _tree(lambda: RandomForestClassifier(**params), impute=True)


def make_et(**overrides) -> FitPredict:
    params = FOREST_PARAMS | overrides
    return _tree(lambda: ExtraTreesClassifier(**params), impute=True)


# 모델 이름 -> 설정을 받아 FitPredict를 만드는 함수
CANDIDATES: dict[str, Callable[..., FitPredict]] = {
    "logistic": make_logistic,
    "lgbm": make_lgbm,
    "xgb": make_xgb,
    "rf": make_rf,
    "et": make_et,
}
SEEDED = {"lgbm", "xgb", "rf", "et"}  # random_state를 받는 모델


def rank_average(preds: list[np.ndarray]) -> np.ndarray:
    """여러 예측의 순위 평균 (0~1). AUC는 순위만 보므로, 확률 크기가 다른 모델을 공평하게 섞는다."""
    return np.mean([rankdata(p) for p in preds], axis=0) / len(preds[0])


def make_ensemble(members: list[FitPredict]) -> FitPredict:
    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        return rank_average([fp(X_train, y_train, X_pred) for fp in members])

    return fit_predict


def make_seed_average(name: str, params: dict, seeds: list[int]) -> FitPredict:
    """같은 모델을 여러 seed로 학습해 확률을 평균한다 (seed에 따른 흔들림 감소)."""
    members = [CANDIDATES[name](**(params | {"random_state": s})) for s in seeds]

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        return np.mean([fp(X_train, y_train, X_pred) for fp in members], axis=0)

    return fit_predict


# 덩어리 3의 측정 모델
logistic, lgbm = make_logistic(), make_lgbm()
PROBES: dict[str, FitPredict] = {"logistic": logistic, "lgbm": lgbm}
probe_ensemble = make_ensemble(list(PROBES.values()))
