"""덩어리 4.5: 사전 등록한 가설 사다리 (docs/EXPERIMENTS.md #7).

시작점은 features_v1(experiments.SELECTED), 모델은 logistic + lgbm 순위 평균.
채택 규칙(강화): cv5에서 validation.beats 를 통과하고, holdout200 평균도 올라야 채택한다.
단계는 "추가"(묶음을 더함) 또는 "교체"(지정한 묶음들을 대신함)다. 마지막으로 모델 비율(C-1)을 본다.

실행:
  python -m src.hypotheses Target_1
  python -m src.hypotheses Target_2
"""

import json
import sys

import numpy as np
import pandas as pd

from src.data import ROOT, TARGETS, load_data
from src.experiments import SELECTED
from src.features import build
from src.models import lgbm, logistic, rank_average
from src.validation import FitPredict, beats, cv_splits, evaluate, holdout_split

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
# (단계 이름, 묶음, 교체 대상 목록) — 교체 대상이 비어 있으면 추가
STEPS = {
    "Target_1": [
        ("A-1 요일 추가", "weekday", []),
        ("A-2 미국 변화×요일", "us_weekday", []),
        ("A-3 입력 축소", "us_core", ["us_change", "us_trend_vol"]),
    ],
    "Target_2": [
        ("B-1 휴장일", "kr_holiday", []),
        ("B-2 월·목 단서", "monthu_interact", []),
    ],
}
WEIGHTS = {"1:1": (1, 1), "2:1": (2, 1), "1:2": (1, 2)}  # logistic : lgbm


def make_weighted(w_logistic: float, w_lgbm: float) -> FitPredict:
    def fit_predict(X_train, y_train, X_pred) -> np.ndarray:
        preds = [logistic(X_train, y_train, X_pred), lgbm(X_train, y_train, X_pred)]
        return rank_average([p for p, w in zip(preds, (w_logistic, w_lgbm)) for _ in range(int(w))])

    return fit_predict


def score(train, test, target: str, families: list[str], model: FitPredict) -> tuple[np.ndarray, float]:
    """(cv5 fold별 AUC, holdout200 AUC)."""
    X, _ = build(train, test, families)
    y = train[[target]]
    cv = evaluate(model, X, y, cv_splits(len(X)))["auc"].to_numpy()
    ho = evaluate(model, X, y, holdout_split(len(X)))["auc"].iloc[0]
    return cv, ho


def adopt(cv: np.ndarray, ho: float, best_cv: np.ndarray, best_ho: float) -> bool:
    return beats(cv, best_cv) and ho > best_ho


def _row(step, families, weights, cv, ho, best_cv, decision) -> dict:
    return {
        "단계": step,
        "묶음": "+".join(families),
        "비율": weights,
        "cv5 평균": cv.mean(),
        "fold별": " ".join(f"{v:.3f}" for v in cv),
        "개선 fold": "-" if best_cv is None else f"{int((cv > best_cv).sum())}/5",
        "holdout200": ho,
        "결정": decision,
    }


def run(target: str) -> dict:
    train, test = load_data()
    current, weights = list(SELECTED[target]), "1:1"
    best_cv, best_ho = score(train, test, target, current, make_weighted(*WEIGHTS[weights]))
    rows = [_row("시작점 (features_v1)", current, weights, best_cv, best_ho, None, "-")]

    for step, family, replaces in STEPS[target]:
        candidate = [f for f in current if f not in replaces] + [family]
        cv, ho = score(train, test, target, candidate, make_weighted(*WEIGHTS[weights]))
        ok = adopt(cv, ho, best_cv, best_ho)
        rows.append(_row(step, candidate, weights, cv, ho, best_cv, "채택" if ok else "기각"))
        if ok:
            current, best_cv, best_ho = candidate, cv, ho

    for name, w in WEIGHTS.items():
        if name == weights:
            continue
        cv, ho = score(train, test, target, current, make_weighted(*w))
        ok = adopt(cv, ho, best_cv, best_ho)
        rows.append(_row(f"C-1 비율 {name}", current, name, cv, ho, best_cv, "채택" if ok else "기각"))
        if ok:
            weights, best_cv, best_ho = name, cv, ho

    summary = pd.DataFrame(rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPERIMENT_DIR / f"hypotheses_{target}.csv", index=False)
    result = {"target": target, "families": current, "weights": weights, "cv5": best_cv.mean(), "holdout200": best_ho}
    (EXPERIMENT_DIR / f"hypotheses_{target}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.float_format", "{:.4f}".format):
        print(f"[{target}] 가설 사다리\n{summary.to_string(index=False)}\n\n최종: {result}")
    return result


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command not in TARGETS:
        sys.exit(f"사용법: python -m src.hypotheses {{{'|'.join(TARGETS)}}}")
    run(command)
