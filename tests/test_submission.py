import numpy as np
import pytest

from src.submission import build_submission, validate_submission


@pytest.fixture
def private_sub(data):
    _, test = data
    rng = np.random.default_rng(0)
    return build_submission(test, {"Target_1": rng.random(len(test)), "Target_2": rng.random(len(test))})


def test_valid_submissions_pass(private_sub):
    validate_submission(private_sub, "private")
    validate_submission(private_sub.head(100).reset_index(drop=True), "public")


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda s: s.head(199),  # 행 수
        lambda s: s.iloc[::-1].reset_index(drop=True),  # 순서
        lambda s: s.rename(columns={"pred_1": "pred1"}),  # 컬럼명
        lambda s: s[["Time_step", "pred_2", "pred_1"]],  # 컬럼 순서
        lambda s: s.assign(pred_1=np.nan),  # 결측
        lambda s: s.assign(pred_2=1.5),  # 범위 초과
        lambda s: s.assign(pred_1=-0.1),  # 범위 미만
        lambda s: s.assign(pred_1=1),  # 정수형
    ],
)
def test_invalid_submissions_fail(private_sub, corrupt):
    with pytest.raises(ValueError):
        validate_submission(corrupt(private_sub.copy()), "private")


def test_public_size_checked(private_sub):
    with pytest.raises(ValueError):
        validate_submission(private_sub, "public")
