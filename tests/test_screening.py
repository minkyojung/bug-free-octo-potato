import numpy as np
import pandas as pd
import pytest

from src.screening import (
    HALFLIFE_DAYS,
    Config2,
    Scores2,
    decay_weights,
    decide2,
    make_model2,
    null_importance_columns,
)

rng = np.random.default_rng(0)


def test_decay_weights_halve_every_halflife():
    w = decay_weights(pd.RangeIndex(100, 100 + 2 * HALFLIFE_DAYS + 1))
    assert w[-1] == 1.0
    assert w[-1 - HALFLIFE_DAYS] == pytest.approx(0.5)
    assert w[0] == pytest.approx(0.25)


def test_null_importance_keeps_signal_and_drops_noise():
    n = 600
    X = pd.DataFrame({"signal": rng.normal(size=n), **{f"noise{i}": rng.normal(size=n) for i in range(8)}})
    y = pd.Series((X["signal"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    kept = null_importance_columns(X, y)
    assert "signal" in kept
    assert len(kept) >= 3  # 최소 보장


def test_v2_config_reproduces_backlog_model_exactly(data):
    """새 모델 빌더가 v2 설정에서는 이전 빌더와 같은 예측을 내야 한다 (출발점 일치)."""
    from src.backlog import FINAL_V2, make_model
    from src.features import build

    train, test = data
    for target, cfg in FINAL_V2.items():
        X, X_test = build(train, test, list(cfg.families))
        old = make_model(cfg)(X.iloc[:500], train[target].iloc[:500], X.iloc[503:600])
        new = make_model2(Config2(**cfg.__dict__))(X.iloc[:500], train[target].iloc[:500], X.iloc[503:600])
        np.testing.assert_array_equal(old, new)


def test_screening_rules():
    base = Scores2(cv=np.full(5, 0.60), pm=np.full(4, 0.60), monthu=0.68, highvol=0.60)
    up = Scores2(cv=np.array([0.62, 0.62, 0.62, 0.62, 0.59]), pm=np.full(4, 0.61), monthu=0.69, highvol=0.61)
    assert decide2("B", up, base, None, None)[0]
    assert decide2("B", up, base, None, "monthu")[0]
    assert not decide2("B", Scores2(up.cv, up.pm, monthu=0.67, highvol=0.61), base, None, "monthu")[0]
    assert not decide2("B", Scores2(up.cv, np.full(4, 0.59), 0.69, 0.61), base, None, None)[0]
    cur, new = (Scores2(np.full(5, v), np.full(4, v), 0.0, 0.0) for v in (0.63, 0.62))
    assert decide2("A", new, cur, base, None)[0]  # A 시작점(0.60) 기준
    assert not decide2("A", new, base, cur, None)[0]  # A 시작점이 0.63이면 거부
