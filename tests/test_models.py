import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from src.models import make_lgbm, make_logistic, make_pairwise_logistic

rng = np.random.default_rng(0)
N = 400
X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
y = pd.Series((X["a"] + 0.5 * rng.normal(size=N) > 0).astype(int))


def test_clip_makes_out_of_range_input_equal_to_boundary():
    """A1 메커니즘: 학습 범위 밖 입력은 경계값과 같은 예측을 받는다."""
    fp = make_logistic(clip=True)
    edge = pd.DataFrame({"a": [X["a"].max()], "b": [0.0]})
    far = pd.DataFrame({"a": [X["a"].max() * 10], "b": [0.0]})
    assert fp(X, y, far)[0] == fp(X, y, edge)[0]
    assert make_logistic(clip=False)(X, y, far)[0] > fp(X, y, far)[0]  # 자르지 않으면 더 극단


def test_monotone_lgbm_never_decreases_along_constrained_feature():
    """A2 메커니즘: 제약 변수만 키우면 예측이 줄지 않는다 (잡음이 큰 데이터에서도)."""
    noisy_y = pd.Series(((np.sin(3 * X["a"]) + rng.normal(size=N)) > 0).astype(int))  # 비단조 관계
    fp = make_lgbm(monotone_increasing=("a",))
    grid = pd.DataFrame({"a": np.linspace(-3, 3, 200), "b": 0.0})
    pred = fp(X, noisy_y, grid)
    assert (np.diff(pred) >= -1e-12).all()


def test_monotone_decreasing_never_increases():
    """G3 메커니즘: 감소 제약 변수만 키우면 예측이 늘지 않는다."""
    noisy_y = pd.Series(((np.sin(3 * X["a"]) + rng.normal(size=N)) > 0).astype(int))
    fp = make_lgbm(monotone_decreasing=("a",))
    grid = pd.DataFrame({"a": np.linspace(-3, 3, 200), "b": 0.0})
    assert (np.diff(fp(X, noisy_y, grid)) <= 1e-12).all()


def test_pairwise_logistic_ranks_like_signal():
    """G5: 쌍 비교 로지스틱이 신호 방향을 제대로 배우고, 일반 로지스틱과 비슷한 순위를 낸다."""
    X_te = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y_te = (X_te["a"] + 0.5 * rng.normal(size=300) > 0).astype(int)
    pw = make_pairwise_logistic()(X, y, X_te)
    assert roc_auc_score(y_te, pw) > 0.85
    assert np.corrcoef(pw, make_logistic()(X, y, X_te))[0, 1] > 0.95
    assert ((pw > 0) & (pw < 1)).all()
