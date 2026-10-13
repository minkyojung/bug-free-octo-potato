"""제출 파일 생성과 형식 검사."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import KEY, N_PUBLIC, RAW_DIR, ROOT, TARGETS

SUBMISSION_DIR = ROOT / "outputs" / "submissions"
PRED_COLUMNS = {"Target_1": "pred_1", "Target_2": "pred_2"}
SUBMITTER = "반_이름"  # 파일명 표기. 제출 전에 직접 수정 (예: "1반_홍길동")


def build_submission(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Private(200행) 제출 프레임을 만든다. preds는 {타깃명: 확률 배열}."""
    sub = pd.DataFrame({KEY: test[KEY].to_numpy()})
    for target in TARGETS:
        sub[PRED_COLUMNS[target]] = np.asarray(preds[target], dtype=float)
    return sub


def validate_submission(sub: pd.DataFrame, kind: str) -> None:
    """sample_submission_<kind>.csv와 비교해 형식이 맞지 않으면 ValueError를 낸다."""
    sample = pd.read_csv(RAW_DIR / f"sample_submission_{kind}.csv")
    if list(sub.columns) != list(sample.columns):
        raise ValueError(f"{kind}: 컬럼이 {list(sample.columns)}이어야 합니다. 현재 {list(sub.columns)}")
    if len(sub) != len(sample) or not np.array_equal(sub[KEY].to_numpy(), sample[KEY].to_numpy()):
        raise ValueError(f"{kind}: 행 수 또는 Time_step 순서가 양식과 다릅니다.")
    for col in PRED_COLUMNS.values():
        if not pd.api.types.is_float_dtype(sub[col]):
            raise ValueError(f"{kind}: {col}은 실수형이어야 합니다.")
        values = sub[col].to_numpy()
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError(f"{kind}: {col}은 NaN 없이 0 이상 1 이하여야 합니다.")


def write_submissions(sub_private: pd.DataFrame, run_name: str, public_round: int) -> tuple[Path, Path]:
    """검사를 통과한 Public(앞 100행)·Private 파일을 outputs/submissions/<run_name>/에 쓴다."""
    sub_public = sub_private.head(N_PUBLIC).reset_index(drop=True)
    validate_submission(sub_public, "public")
    validate_submission(sub_private, "private")

    out_dir = SUBMISSION_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    public_path = out_dir / f"public_{SUBMITTER}_{public_round}.csv"
    private_path = out_dir / f"private_{SUBMITTER}.csv"
    sub_public.to_csv(public_path, index=False)
    sub_private.to_csv(private_path, index=False)
    return public_path, private_path
