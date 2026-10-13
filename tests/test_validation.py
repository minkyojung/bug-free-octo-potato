import numpy as np
import pandas as pd
import pytest

from src.validation import (
    FOLD_SIZE,
    GAP,
    HOLDOUT_SIZE,
    N_SPLITS,
    beats,
    cv_splits,
    cv_splits_fine,
    evaluate,
    holdout_split,
    private_mimic_splits,
    summarize,
)

N = 924


@pytest.mark.parametrize("splits", [cv_splits(N), cv_splits_fine(N), private_mimic_splits(N), holdout_split(N)])
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


@pytest.mark.parametrize(
    "new, expected",
    [
        ([0.65, 0.65, 0.65, 0.65, 0.55], True),  # 평균 상승, 4/5 개선
        ([0.61, 0.61, 0.61, 0.50, 0.50], False),  # 3/5 개선
        ([0.70, 0.59, 0.59, 0.59, 0.59], False),  # 1/5 개선 (평균은 상승)
        ([0.601, 0.601, 0.601, 0.601, 0.40], False),  # 4/5 개선이지만 평균 하락
    ],
)
def test_beats_rule(new, expected):
    assert beats(new, [0.6] * 5) is expected


def test_fine_cv_covers_same_period_as_cv5():
    fine, coarse = cv_splits_fine(N), cv_splits(N)
    assert len(fine) == 10 and all(len(va) == 50 for _, va in fine)
    assert fine[0][1][0] == coarse[0][1][0] and fine[-1][1][-1] == coarse[-1][1][-1]


def test_private_mimic_layout():
    splits = private_mimic_splits(N)
    assert [va[0] + 1 for _, va in splits] == [425, 525, 625, 725]  # Time_step 기준 시작일
    assert all(len(va) == HOLDOUT_SIZE for _, va in splits)
    last_tr, last_va = splits[-1]
    (ho_tr, ho_va), = holdout_split(N)
    assert np.array_equal(last_tr, ho_tr) and np.array_equal(last_va, ho_va)  # 마지막 창 = holdout200


def test_backlog_decision_rules():
    """사전 등록 규칙(#10)이 코드와 일치하는지: B는 세 조건 모두, A는 A 시작점 대비 −0.002 이내."""
    from src.backlog import Scores, decide

    base = Scores(cv=np.full(5, 0.60), pm=np.full(4, 0.60), regime=0.01, monthu=0.68)
    up = Scores(cv=np.array([0.62, 0.62, 0.62, 0.62, 0.59]), pm=np.full(4, 0.61), regime=0.02, monthu=0.69)
    assert decide("B", up, base, base, "regime")[0]
    assert not decide("B", Scores(up.cv, up.pm, regime=0.0, monthu=0.69), base, base, "regime")[0]  # 메커니즘 실패
    assert not decide("B", Scores(up.cv, np.full(4, 0.59), 0.02, 0.69), base, base, "regime")[0]  # pm4 하락
    slight_drop = Scores(cv=np.full(5, 0.5985), pm=np.full(4, 0.5985), regime=0.0, monthu=0.0)
    assert decide("A", slight_drop, base, base, None)[0]  # −0.0015: 허용
    big_drop = Scores(cv=np.full(5, 0.597), pm=np.full(4, 0.60), regime=0.0, monthu=0.0)
    assert not decide("A", big_drop, base, base, None)[0]  # −0.003: 거부
    # A는 직전 단계(current)가 아니라 A 시작 직전 상태(a_base)와 비교한다
    cur, new = (Scores(np.full(5, v), np.full(4, v), 0.0, 0.0) for v in (0.63, 0.62))
    assert decide("A", new, cur, base, None)[0]  # a_base=0.60 기준: 통과
    assert not decide("A", new, base, cur, None)[0]  # a_base=0.63 기준: 거부
