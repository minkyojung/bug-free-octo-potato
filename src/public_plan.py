"""Public 제출 계획 (docs/EXPERIMENTS.md #15).

회차별 Public 파일을 만들고, Public 결과를 받은 뒤 타깃별로 점수가 높았던 회차의 후보를 조합해
최종 Private 파일을 만든다. 리더보드는 KOSPI(auc_1)와 건강식품(auc_2)을 따로 채점하므로
한 번 제출로 두 문제의 후보를 각각 시험할 수 있다.

실행:
  python -m src.public_plan round 1        # 1회차 Public 파일
  python -m src.public_plan round 2        # 2회차 Public 파일
  python -m src.public_plan final 1 2      # 최종 Private: KOSPI는 1회차 후보, 건강식품은 2회차 후보
"""

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data import N_PUBLIC, TARGETS, load_data
from src.features import build
from src.screening import FINAL_V3, Config2, make_model2
from src.submission import PRED_COLUMNS, SUBMISSION_DIR, SUBMITTER, build_submission, validate_submission

ROUNDS = {
    1: {"Target_1": FINAL_V3["Target_1"], "Target_2": FINAL_V3["Target_2"]},  # K-a, H-a
    2: {
        "Target_1": replace(FINAL_V3["Target_1"], decay=True),  # K-b: + 최근 비중
        "Target_2": Config2(("weekday", "traffic", "weekday_dev"), clip=True),  # H-b: v2 건강식품
    },
}
# 검산: 이미 확정해 둔 제출 파일과 예측이 정확히 같아야 하는 (회차, 타깃)
CROSS_CHECK = {(1, "Target_1"): "features_v3", (1, "Target_2"): "features_v3", (2, "Target_2"): "features_v2"}
PLAN_DIR = SUBMISSION_DIR / "public_plan"
FINAL_DIR = SUBMISSION_DIR / "final"


def predict(round_no: int, target: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """해당 회차 후보를 train 전체로 학습해 test 200일을 예측하고, 검산 대상이면 기존 파일과 대조한다."""
    cfg = ROUNDS[round_no][target]
    X, X_test = build(train, test, list(cfg.families))
    pred = make_model2(cfg)(X, train[target], X_test)
    run = CROSS_CHECK.get((round_no, target))
    if run is not None:
        recorded = pd.read_csv(SUBMISSION_DIR / run / f"private_{SUBMITTER}.csv", float_precision="round_trip")
        if not np.array_equal(recorded[PRED_COLUMNS[target]].to_numpy(), pred):
            raise RuntimeError(f"검산 실패: {round_no}회차 {target} 예측이 {run} 제출 파일과 다릅니다.")
    return pred


def make_round(round_no: int) -> None:
    train, test = load_data()
    sub = build_submission(test, {t: predict(round_no, t, train, test) for t in TARGETS})
    public = sub.head(N_PUBLIC).reset_index(drop=True)
    validate_submission(public, "public")
    validate_submission(sub, "private")
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAN_DIR / f"public_{SUBMITTER}_{round_no}.csv"
    public.to_csv(path, index=False)
    print(f"[{round_no}회차 Public 파일] {path.relative_to(SUBMISSION_DIR.parent.parent)}")


def make_final(round_t1: int, round_t2: int) -> None:
    """최종 Private 파일: KOSPI는 round_t1회차 후보, 건강식품은 round_t2회차 후보로 만든다."""
    train, test = load_data()
    preds = {"Target_1": predict(round_t1, "Target_1", train, test), "Target_2": predict(round_t2, "Target_2", train, test)}
    sub = build_submission(test, preds)
    validate_submission(sub, "private")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    path = FINAL_DIR / f"private_{SUBMITTER}.csv"
    sub.to_csv(path, index=False)
    print(f"[최종 Private 파일] {path.relative_to(SUBMISSION_DIR.parent.parent)} (KOSPI: {round_t1}회차 후보, 건강식품: {round_t2}회차 후보)")


if __name__ == "__main__":
    args = sys.argv[1:]
    rounds = [str(r) for r in ROUNDS]
    if len(args) == 2 and args[0] == "round" and args[1] in rounds:
        make_round(int(args[1]))
    elif len(args) == 3 and args[0] == "final" and args[1] in rounds and args[2] in rounds:
        make_final(int(args[1]), int(args[2]))
    else:
        sys.exit("사용법: python -m src.public_plan round {1|2}  또는  python -m src.public_plan final {KOSPI 회차} {건강식품 회차}")
