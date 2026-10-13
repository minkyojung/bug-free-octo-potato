import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eda import auc_scorer, shift_null_max, stratified_auc_scorer

rng = np.random.default_rng(0)
N = 300
X = np.column_stack([rng.normal(size=N), rng.integers(0, 4, N)])  # 2번째 열은 동점이 많음
y = (X[:, 0] + rng.normal(size=N) > 0).astype(int)
groups = rng.integers(0, 3, N)


def test_auc_matches_sklearn_with_ties():
    expected = [roc_auc_score(y, X[:, j]) for j in range(X.shape[1])]
    assert auc_scorer(X)(y) == pytest.approx(expected)


def test_stratified_auc_pools_within_group_pairs():
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        pairs = y[m].sum() * (m.sum() - y[m].sum())
        num += roc_auc_score(y[m], X[m, 0]) * pairs
        den += pairs
    assert stratified_auc_scorer(X, groups)(y)[0] == pytest.approx(num / den)


def test_stratified_auc_skips_single_class_group():
    g = np.where(np.arange(N) < 10, 9, 0)  # 그룹 9를 한 클래스로 만듦
    yy = y.copy()
    yy[:10] = 1
    expected = roc_auc_score(yy[10:], X[10:, 0])
    assert stratified_auc_scorer(X, g)(yy)[0] == pytest.approx(expected)


def test_shift_null_shape_and_range():
    null = shift_null_max(auc_scorer(X), y, n_shifts=50, step=7)
    assert null.shape == (50,)
    assert ((null >= 0) & (null <= 0.5)).all()
