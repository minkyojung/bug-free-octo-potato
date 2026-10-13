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


def make_logistic(C: float = LOGISTIC_C, clip: bool = False) -> FitPredict:
    """clip=True면 예측할 입력을 학습 데이터의 최소~최대로 자른다 (범위 밖 선형 외삽 방지)."""

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame, sample_weight=None) -> np.ndarray:
        keep = _non_constant(X_train)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        model.fit(X_train[keep], y_train, logisticregression__sample_weight=sample_weight)
        X_in = X_pred[keep]
        if clip:
            X_in = X_in.clip(lower=X_train[keep].min(), upper=X_train[keep].max(), axis=1)
        return model.predict_proba(X_in)[:, 1]

    return fit_predict


def _tree(build_model: Callable[[pd.Index], object], impute: bool) -> FitPredict:
    """트리 모델 공통 래퍼. build_model은 실제로 쓰는 컬럼을 받는다.
    impute=False면 결측을 모델이 직접 처리한다 (LightGBM, XGBoost)."""

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame, sample_weight=None) -> np.ndarray:
        keep = _non_constant(X_train)
        model = build_model(keep)
        fit_kwargs = {"sample_weight": sample_weight}
        if impute:
            model = make_pipeline(SimpleImputer(strategy="median"), model)
            fit_kwargs = {f"{model.steps[-1][0]}__sample_weight": sample_weight}
        model.fit(_positional(X_train[keep]), y_train, **fit_kwargs)
        return model.predict_proba(_positional(X_pred[keep]))[:, 1]

    return fit_predict


def make_lgbm(monotone_increasing: tuple[str, ...] = (), monotone_decreasing: tuple[str, ...] = (), **overrides) -> FitPredict:
    """단조 제약: increasing 컬럼은 값이 커지면 예측이 줄지 않고, decreasing 컬럼은 늘지 않는다."""
    params = LGBM_PARAMS | overrides
    if set(monotone_increasing) & set(monotone_decreasing):
        raise ValueError("같은 컬럼에 증가·감소 제약을 동시에 걸 수 없습니다.")

    def build(columns: pd.Index) -> LGBMClassifier:
        if not monotone_increasing and not monotone_decreasing:
            return LGBMClassifier(**params)
        constraints = [int(c in monotone_increasing) - int(c in monotone_decreasing) for c in columns]
        return LGBMClassifier(**params, monotone_constraints=constraints)

    return _tree(build, impute=False)


def make_pairwise_logistic(C: float = LOGISTIC_C, clip: bool = False) -> FitPredict:
    """쌍 비교 로지스틱: (양성 − 음성) 입력 차이로 '어느 날이 더 오를까'를 학습한다 (AUC의 표준 대리 손실).

    정규화 강도는 일반 로지스틱과 같도록 C에 (표본 가중치 합 / 쌍 가중치 합)을 곱한다
    (가중치가 없으면 표본 수 / 쌍 수). sample_weight가 있으면 쌍의 가중치는 두 날 가중치의 곱이다.
    """

    def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame, sample_weight=None) -> np.ndarray:
        keep = _non_constant(X_train)
        prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_train[keep])
        Z = prep.transform(X_train[keep])
        y = np.asarray(y_train)
        pos, neg = Z[y == 1], Z[y == 0]
        diffs = (pos[:, None, :] - neg[None, :, :]).reshape(-1, Z.shape[1])
        X_pairs = np.vstack([diffs, -diffs])
        y_pairs = np.r_[np.ones(len(diffs)), np.zeros(len(diffs))]
        w = np.ones(len(Z)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w_pairs = np.tile(np.outer(w[y == 1], w[y == 0]).ravel(), 2)
        C_pair = C * w.sum() / w_pairs.sum()
        model = LogisticRegression(C=C_pair, fit_intercept=False, max_iter=5000).fit(
            X_pairs, y_pairs, sample_weight=None if sample_weight is None else w_pairs
        )
        X_in = X_pred[keep]
        if clip:
            X_in = X_in.clip(lower=X_train[keep].min(), upper=X_train[keep].max(), axis=1)
        return 1 / (1 + np.exp(-(prep.transform(X_in) @ model.coef_[0])))

    return fit_predict


def make_xgb(**overrides) -> FitPredict:
    params = XGB_PARAMS | overrides
    return _tree(lambda columns: XGBClassifier(**params), impute=False)


def make_rf(**overrides) -> FitPredict:
    params = FOREST_PARAMS | overrides
    return _tree(lambda columns: RandomForestClassifier(**params), impute=True)


def make_et(**overrides) -> FitPredict:
    params = FOREST_PARAMS | overrides
    return _tree(lambda columns: ExtraTreesClassifier(**params), impute=True)


def on_columns(columns: list[str], fit_predict: FitPredict) -> FitPredict:
    """지정한 컬럼만 쓰는 모델로 감싼다 (피처셋 앙상블용)."""

    def wrapped(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        return fit_predict(X_train[columns], y_train, X_pred[columns])

    return wrapped


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
