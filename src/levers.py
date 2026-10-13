"""덩어리 4.6: 사전 등록한 구조적 지렛대 (docs/EXPERIMENTS.md #8).

저울: cv10(50일 × 10회) + holdout200. 채택: cv10 평균 상승 + 10개 중 7개 이상 개선 + holdout200 상승.

실행:
  python -m src.levers Target_1
  python -m src.levers Target_2
"""

import json
import sys

import numpy as np
import pandas as pd

from src.augment import make_augmented, make_recent, sibling_columns
from src.data import ROOT, TARGETS, load_data
from src.experiments import SELECTED
from src.features import build
from src.models import probe_ensemble
from src.validation import FitPredict, beats, cv_splits_fine, evaluate, holdout_split

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
MIN_WINS_FINE = 7
RECENT_WINDOW = 500


def levers(train: pd.DataFrame, target: str) -> list[tuple[str, FitPredict]]:
    if target == "Target_2":
        n_sib = len(sibling_columns(train))
        return [
            ("L-1a 형제 증강 (동일 가중)", make_augmented(train, target_weight=1.0)),
            (f"L-1b 형제 증강 (원래 정답 ×{n_sib})", make_augmented(train, target_weight=float(n_sib))),
        ]
    return [(f"L-3 최근 {RECENT_WINDOW}일 학습", make_recent(RECENT_WINDOW))]


def score(X: pd.DataFrame, y: pd.DataFrame, model: FitPredict) -> tuple[np.ndarray, float]:
    cv = evaluate(model, X, y, cv_splits_fine(len(X)))["auc"].to_numpy()
    ho = evaluate(model, X, y, holdout_split(len(X)))["auc"].iloc[0]
    return cv, ho


def _row(step, cv, ho, best_cv, decision) -> dict:
    return {
        "단계": step,
        "cv10 평균": cv.mean(),
        "fold별": " ".join(f"{v:.3f}" for v in cv),
        "개선 fold": "-" if best_cv is None else f"{int((cv > best_cv).sum())}/10",
        "holdout200": ho,
        "결정": decision,
    }


def run(target: str) -> dict:
    train, test = load_data()
    X, _ = build(train, test, SELECTED[target])
    y = train[[target]]

    best_cv, best_ho = score(X, y, probe_ensemble)
    rows = [_row("시작점 (features_v1, cv10으로 재측정)", best_cv, best_ho, None, "-")]
    chosen = "features_v1"
    for name, model in levers(train, target):
        cv, ho = score(X, y, model)
        ok = beats(cv, best_cv, min_wins=MIN_WINS_FINE) and ho > best_ho
        rows.append(_row(name, cv, ho, best_cv, "채택" if ok else "기각"))
        if ok:
            chosen, best_cv, best_ho = name, cv, ho

    summary = pd.DataFrame(rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPERIMENT_DIR / f"levers_{target}.csv", index=False)
    result = {"target": target, "chosen": chosen, "cv10": best_cv.mean(), "holdout200": best_ho}
    (EXPERIMENT_DIR / f"levers_{target}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with pd.option_context("display.width", 250, "display.max_colwidth", 80, "display.float_format", "{:.4f}".format):
        print(f"[{target}] 지렛대 결과\n{summary.to_string(index=False)}\n\n최종: {result}")
    return result


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command not in TARGETS:
        sys.exit(f"사용법: python -m src.levers {{{'|'.join(TARGETS)}}}")
    run(command)
