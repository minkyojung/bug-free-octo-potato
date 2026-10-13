import numpy as np
import pandas as pd
import pytest

from src.validation import FOLD_SIZE, GAP, HOLDOUT_SIZE, N_SPLITS, cv_splits, evaluate, holdout_split, summarize

N = 924


@pytest.mark.parametrize("splits", [cv_splits(N), holdout_split(N)])
def test_splits_keep_time_order_and_gap(splits):
    for tr, va in splits:
        assert tr[0] == 0 and np.array_equal(tr, np.arange(tr[0], tr[-1] + 1))  # 처음부터 연속
        assert np.array_equal(va, np.arange(va[0], va[-1] + 1))  # 연속 구간
        assert va[0] - tr[-1] - 1 == GAP  # 학습 끝과 검증 시작 사이 정확히 GAP일


def test_cv_layout():
    splits = cv_splits(N)
    assert len(splits) == N_SPLITS
    assert all(len(va) == FOLD_SIZE for _, va in splits)
    assert splits[-1][1][-1] == N - 1  # 마지막 fold가 train 끝까지 검증


def test_holdout_layout():
    (tr, va), = holdout_split(N)
    assert len(va) == HOLDOUT_SIZE and va[-1] == N - 1


def test_evaluate_passes_only_past_rows():
    X = pd.DataFrame({"x": np.arange(N, dtype=float)})
    y = pd.DataFrame({"Target_1": np.arange(N) % 2, "Target_2": (np.arange(N) // 2) % 2})

    def fit_predict(X_tr, y_tr, X_va):
        assert X_tr.index.max() + GAP < X_va.index.min()
        return X_va["x"].to_numpy() % 2  # Target_1과 완전히 일치하는 예측

    scores = evaluate(fit_predict, X, y, cv_splits(N))
    assert len(scores) == 2 * N_SPLITS
    assert (scores.query("target == 'Target_1'")["auc"] == 1.0).all()
    summary = summarize(scores)
    assert summary.loc["평균(T1,T2)", "mean"] == pytest.approx(summary.loc[["Target_1", "Target_2"], "mean"].mean())


def test_evaluate_rejects_bad_predictions():
    X = pd.DataFrame({"x": np.zeros(N)})
    y = pd.DataFrame({"Target_1": np.arange(N) % 2, "Target_2": np.arange(N) % 2})
    with pytest.raises(ValueError):
        evaluate(lambda X_tr, y_tr, X_va: np.full(len(X_va), np.nan), X, y, cv_splits(N))
