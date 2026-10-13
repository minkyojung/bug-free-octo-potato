"""덩어리 4: 모델 비교 → 설정 조정 → 앙상블 → 안정성 점검.

입력 피처는 덩어리 3에서 확정한 묶음(experiments.SELECTED)으로 고정한다.
아래 탐색 범위와 판정 규칙(validation.beats)은 결과를 보기 전에 정했다.

1. 비교: CANDIDATES의 기본 설정을 cv5로 비교한다.
2. 조정: 상위 TUNE_TOP개 모델만 GRIDS 범위에서 조정한다. 최고 설정이 기본 설정을 이길(beats) 때만 바꾼다.
3. 앙상블: 최고 단일 모델에서 시작해, 남은 모델 중 가장 좋은 하나를 순위 평균에 넣어 본다.
   이기면 채택하고 반복, 지면 멈춘다.
4. 안정성: 최종 조합을 seed 5개로 반복해 흔들림을 재고, seed 평균 버전을 최종 후보로 쓴다.
마지막으로 최종 후보가 덩어리 3 결과(측정 모델 순위 평균)를 이기지 못하면 덩어리 3 결과를 유지한다.

실행:
  python -m src.modeling Target_1
  python -m src.modeling Target_2
"""

import json
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import ROOT, TARGETS, load_data
from src.experiments import SELECTED
from src.features import build
from src.models import CANDIDATES, SEEDED, make_ensemble, make_seed_average, rank_average
from src.validation import FitPredict, Split, beats, cv_splits

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
TUNE_TOP = 2
GRIDS = {
    "logistic": {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "lgbm": {"num_leaves": [3, 7, 15], "n_estimators": [150, 300, 600]},
    "xgb": {"max_depth": [2, 3, 4], "n_estimators": [150, 300, 600]},
    "rf": {"max_depth": [4, 6, 10], "min_samples_leaf": [5, 10, 30]},
    "et": {"max_depth": [4, 6, 10], "min_samples_leaf": [5, 10, 30]},
}
SEEDS = [0, 1, 2, 3, 4]

Spec = tuple[str, dict]  # (모델 이름, 기본값에서 바꿀 설정)


def label(specs: list[Spec]) -> str:
    def one(name: str, params: dict) -> str:
        return name if not params else f"{name}({', '.join(f'{k}={v}' for k, v in params.items())})"

    return " + ".join(one(n, p) for n, p in specs)


def seed_averaged(specs: list[Spec]) -> FitPredict:
    """seed를 받는 모델은 SEEDS 평균으로 바꾼 순위 평균 앙상블."""
    members = [make_seed_average(n, p, SEEDS) if n in SEEDED else CANDIDATES[n](**p) for n, p in specs]
    return make_ensemble(members)


class FoldCache:
    """모델별 fold 예측을 한 번만 계산해 두고, 어떤 조합의 순위 평균이든 바로 채점한다."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, splits: list[Split]):
        self.X, self.y, self.splits = X, y, splits
        self._preds: dict[str, list[np.ndarray]] = {}

    def _fold_preds(self, fit_predict: FitPredict) -> list[np.ndarray]:
        preds = []
        for tr, va in self.splits:
            p = np.asarray(fit_predict(self.X.iloc[tr], self.y.iloc[tr], self.X.iloc[va]), dtype=float)
            if p.shape != (len(va),) or not np.isfinite(p).all():
                raise ValueError("예측값 형태가 잘못되었거나 유한하지 않습니다.")
            preds.append(p)
        return preds

    def preds(self, spec: Spec) -> list[np.ndarray]:
        key = json.dumps(spec, sort_keys=True)
        if key not in self._preds:
            name, params = spec
            self._preds[key] = self._fold_preds(CANDIDATES[name](**params))
        return self._preds[key]

    def _auc(self, fold_preds: list[np.ndarray]) -> np.ndarray:
        return np.array([roc_auc_score(self.y.iloc[va], p) for p, (_, va) in zip(fold_preds, self.splits)])

    def auc(self, specs: list[Spec]) -> np.ndarray:
        """조합(순위 평균)의 fold별 AUC."""
        members = [self.preds(s) for s in specs]
        return self._auc([rank_average(list(fold)) for fold in zip(*members)])

    def auc_of(self, fit_predict: FitPredict) -> np.ndarray:
        return self._auc(self._fold_preds(fit_predict))


def _row(step: str, what: str, scores: np.ndarray, compared: np.ndarray | None, decision: str) -> dict:
    return {
        "단계": step,
        "구성": what,
        "평균": scores.mean(),
        "fold별": " ".join(f"{v:.3f}" for v in scores),
        "개선 fold": "-" if compared is None else f"{int((scores > compared).sum())}/{len(scores)}",
        "결정": decision,
    }


def run_target(target: str) -> dict:
    train, test = load_data()
    X, _ = build(train, test, SELECTED[target])
    cache = FoldCache(X, train[target], cv_splits(len(X)))
    rows = []

    reference = cache.auc([("logistic", {}), ("lgbm", {})])
    rows.append(_row("기준", "덩어리 3 결과 (logistic + lgbm)", reference, None, "-"))

    # 1. 비교
    singles = {name: cache.auc([(name, {})]) for name in CANDIDATES}
    for name, scores in singles.items():
        rows.append(_row("1. 비교", name, scores, None, "-"))
    ranked = sorted(singles, key=lambda n: singles[n].mean(), reverse=True)

    # 2. 조정 (상위 TUNE_TOP개)
    chosen = {name: {} for name in CANDIDATES}
    for name in ranked[:TUNE_TOP]:
        grid = [dict(zip(GRIDS[name], values)) for values in product(*GRIDS[name].values())]
        best = max(grid, key=lambda p: cache.auc([(name, p)]).mean())
        best_scores = cache.auc([(name, best)])
        adopted = beats(best_scores, singles[name])
        if adopted:
            chosen[name] = best
        rows.append(_row("2. 조정", label([(name, best)]), best_scores, singles[name], "채택" if adopted else "기각 (기본 유지)"))

    # 3. 앙상블 (전진 선택)
    pool = [(name, chosen[name]) for name in CANDIDATES]
    current = [max(pool, key=lambda s: cache.auc([s]).mean())]
    current_scores = cache.auc(current)
    rows.append(_row("3. 앙상블", label(current), current_scores, None, "시작점 (최고 단일)"))
    while rest := [s for s in pool if s not in current]:
        add = max(rest, key=lambda s: cache.auc(current + [s]).mean())
        trial = cache.auc(current + [add])
        adopted = beats(trial, current_scores)
        rows.append(_row("3. 앙상블", label(current + [add]), trial, current_scores, "채택" if adopted else "기각 (중단)"))
        if not adopted:
            break
        current, current_scores = current + [add], trial

    # 4. 안정성: seed별 흔들림과 seed 평균
    seeded = [n for n, _ in current if n in SEEDED]
    if seeded:
        per_seed = np.array(
            [cache.auc([(n, p | {"random_state": s}) if n in SEEDED else (n, p) for n, p in current]) for s in SEEDS]
        )
        rows.append(_row("4. 안정성", f"seed {len(SEEDS)}개 각각 (평균 ± 표준편차 {per_seed.mean(1).std():.4f})", per_seed.mean(0), None, "-"))
    final_scores = cache.auc_of(seed_averaged(current))
    rows.append(_row("4. 안정성", f"seed 평균: {label(current)}", final_scores, current_scores, "-"))

    adopted = beats(final_scores, reference)
    rows.append(_row("최종 판정", "덩어리 4 후보 vs 덩어리 3 결과", final_scores, reference, "덩어리 4 채택" if adopted else "덩어리 3 유지"))

    summary = pd.DataFrame(rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPERIMENT_DIR / f"modeling_{target}.csv", index=False)
    result = {"target": target, "members": current if adopted else None}
    (EXPERIMENT_DIR / f"modeling_{target}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.float_format", "{:.4f}".format):
        print(f"[{target}] 모델링 결과\n{summary.to_string(index=False)}\n\n최종: {result}")
    return result


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command in TARGETS:
        run_target(command)
    else:
        sys.exit(f"사용법: python -m src.modeling {{{'|'.join(TARGETS)}}}")
