"""덩어리 3: 피처 묶음 사다리 실험.

측정용 모델 2개(src/models.py)를 고정하고, 핵심 묶음에서 시작해 묶음을 하나씩 더하며 cv5로 비교한다.
사다리 순서와 채택 규칙은 결과를 보기 전에 정했다.

채택 규칙: 두 측정 모델 AUC의 fold별 평균을 점수로 삼는다. 직전 채택 상태보다
  (1) 5개 fold 평균이 오르고, (2) 5개 fold 중 4개 이상에서 오르면 채택한다 (validation.beats).
  기각된 묶음은 이후 단계에 포함하지 않는다. 마지막 단계 raw는 "원래값을 더하는 게 나은가"의 검증이다.

실행:
  python -m src.experiments Target_1   # 사다리
  python -m src.experiments Target_2   # 사다리
  python -m src.experiments final      # 선택된 묶음으로 최종 채점 + 제출 파일
"""

import sys

import pandas as pd

from src.data import ROOT, TARGETS, load_data
from src.features import build
from src.models import PROBES, probe_ensemble
from src.submission import build_submission, write_submissions
from src.validation import beats, cv_splits, evaluate, holdout_split, summarize

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
REFERENCE = ["raw"]  # 베이스라인과 같은 입력
LADDERS = {
    "Target_1": {"core": ["us_change"], "rungs": ["us_composite", "us_trend_vol", "macro_change", "raw"]},
    "Target_2": {"core": ["weekday"], "rungs": ["traffic", "weekday_dev", "food_health", "raw"]},
}
# 사다리 결과로 확정한 묶음 (docs/EXPERIMENTS.md #2, #3)
SELECTED = {
    "Target_1": ["us_change", "us_trend_vol"],
    "Target_2": ["weekday", "traffic", "weekday_dev"],
}
FINAL_RUN = "features_v1"
PUBLIC_ROUND = 2


def score_folds(train: pd.DataFrame, test: pd.DataFrame, target: str, families: list[str]) -> pd.DataFrame:
    """fold별 측정 모델 AUC와 그 평균."""
    X, _ = build(train, test, families)
    splits = cv_splits(len(X))
    table = pd.DataFrame(
        {name: evaluate(fp, X, train[[target]], splits)["auc"].to_numpy() for name, fp in PROBES.items()},
        index=pd.RangeIndex(1, len(splits) + 1, name="fold"),
    )
    table["평균"] = table[list(PROBES)].mean(axis=1)
    return table


def _summary_row(step: str, families: list[str], scores: pd.DataFrame, previous: pd.DataFrame | None, decision: str) -> dict:
    return {
        "단계": step,
        "묶음": "+".join(families),
        **{name: scores[name].mean() for name in PROBES},
        "평균": scores["평균"].mean(),
        "fold별 평균": " ".join(f"{v:.3f}" for v in scores["평균"]),
        "개선 fold": "-" if previous is None else f"{int((scores['평균'] > previous['평균']).sum())}/{len(scores)}",
        "결정": decision,
    }


def run_ladder(target: str) -> list[str]:
    train, test = load_data()
    ladder = LADDERS[target]

    reference = score_folds(train, test, target, REFERENCE)
    current = list(ladder["core"])
    best = score_folds(train, test, target, current)
    rows = [
        _summary_row("기준: 원본 변수", REFERENCE, reference, None, "-"),
        _summary_row("핵심 묶음", current, best, reference, "시작점"),
    ]
    details = [reference.assign(단계="기준"), best.assign(단계="핵심")]

    for rung in ladder["rungs"]:
        candidate = current + [rung]
        scores = score_folds(train, test, target, candidate)
        adopted = beats(scores["평균"], best["평균"])
        rows.append(_summary_row(f"+ {rung}", candidate, scores, best, "채택" if adopted else "기각"))
        details.append(scores.assign(단계=f"+ {rung}"))
        if adopted:
            current, best = candidate, scores

    summary = pd.DataFrame(rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPERIMENT_DIR / f"ladder_{target}.csv", index=False)
    pd.concat(details).to_csv(EXPERIMENT_DIR / f"ladder_{target}_folds.csv")

    with pd.option_context("display.width", 250, "display.max_colwidth", 80, "display.float_format", "{:.3f}".format):
        print(f"[{target}] 사다리 결과\n{summary.to_string(index=False)}\n\n최종 묶음: {current}")
    return current


def run_final() -> None:
    """선택된 묶음 + 측정 모델 순위 평균을 cv5·holdout200으로 채점하고, train 전체로 재학습해 제출 파일을 만든다."""
    train, test = load_data()
    preds = {}
    for scheme, make_splits in [("cv5", cv_splits), ("holdout200", holdout_split)]:
        scores = []
        for target in TARGETS:
            X, _ = build(train, test, SELECTED[target])
            scores.append(evaluate(probe_ensemble, X, train[[target]], make_splits(len(X))))
        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(EXPERIMENT_DIR / f"{FINAL_RUN}_{scheme}.csv", index=False)
        print(f"[{scheme}]\n{summarize(scores).round(4).to_string()}\n")

    for target in TARGETS:
        X, X_test = build(train, test, SELECTED[target])
        preds[target] = probe_ensemble(X, train[target], X_test)
    public_path, private_path = write_submissions(build_submission(test, preds), FINAL_RUN, PUBLIC_ROUND)
    print(f"[제출 파일] {public_path.relative_to(ROOT)}, {private_path.relative_to(ROOT)}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command in TARGETS:
        run_ladder(command)
    elif command == "final":
        run_final()
    else:
        sys.exit(f"사용법: python -m src.experiments {{{'|'.join(TARGETS)}|final}}")
