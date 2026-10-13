"""덩어리 4.8: 독립 시험 → 합치기 → 최종 확인 (docs/EXPERIMENTS.md #13).

출발점은 features_v2. 후보 정의, 순서, 판정 규칙은 사전 등록대로다.
1. 독립 시험: 각 후보를 v2에 하나씩 붙여 v2와 비교한다.
2. 합치기: 통과한 후보만 정한 순서로 하나씩 쌓으며 직전 상태와 비교한다 (A는 A 시작 직전 상태와 비교).
3. 최종 확인: 합친 결과를 다시 v2와 비교하고, 실패하면 통과 후보 중 pm4가 가장 높은 하나만 쓴다.

실행:
  python -m src.screening Target_1
  python -m src.screening Target_2
"""

import json
import sys
from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.backlog import FINAL_V2, _predict
from src.data import ROOT, TARGETS, load_data
from src.features import (
    FOOD_WDDEV_COLUMNS,
    GENERAL_TRAFFIC_COLUMNS,
    US_CHANGE_COLUMNS,
    US_CORE_COLUMNS,
    US_VOLNORM_COLUMNS,
    build,
)
from src.models import (
    LGBM_PARAMS,
    _non_constant,
    _positional,
    make_ensemble,
    make_lgbm,
    make_logistic,
    make_pairwise_logistic,
    on_columns,
    rank_average,
)
from src.validation import FitPredict, beats, cv_splits, private_mimic_splits

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
A_TOLERANCE = 0.002
HALFLIFE_DAYS = 365  # T10
N_NULL, NULL_PERCENTILE, MIN_KEEP, MIN_SHIFT = 20, 95, 3, 60  # N9
MON, THU = 2, 5


@dataclass(frozen=True)
class Config2:
    families: tuple[str, ...]
    clip: bool = False  # v2 A1
    monotone: bool = False  # v2 A2 (+ G4 컬럼에도 적용)
    hedge_core: bool = False  # v2 A3
    drop_general: bool = False  # G2
    mean_reversion: bool = False  # G3
    pairwise: bool = False  # G5
    null_select: bool = False  # N9
    decay: bool = False  # T10


def decay_weights(index: pd.Index) -> np.ndarray:
    """T10: 가장 최근 날의 가중치 1, HALFLIFE_DAYS일 전은 0.5. index는 시간 순서의 행 번호다."""
    age = index.max() - np.asarray(index)
    return 0.5 ** (age / HALFLIFE_DAYS)


def _gain(X: pd.DataFrame, y: np.ndarray, sample_weight) -> np.ndarray:
    model = LGBMClassifier(**LGBM_PARAMS, importance_type="gain").fit(_positional(X), y, sample_weight=sample_weight)
    return model.feature_importances_


def null_importance_columns(X_train: pd.DataFrame, y_train: pd.Series, sample_weight=None) -> list[str]:
    """N9: 정답을 순환 이동(자기상관 보존)시킨 20회의 중요도 95분위를 넘는 변수만 남긴다 (최소 MIN_KEEP개)."""
    keep = _non_constant(X_train)
    X, y = X_train[keep], np.asarray(y_train)
    true = _gain(X, y, sample_weight)
    shifts = np.random.default_rng(0).integers(MIN_SHIFT, len(y) - MIN_SHIFT, N_NULL)
    null = np.array([_gain(X, np.roll(y, s), sample_weight) for s in shifts])
    passed = true > np.percentile(null, NULL_PERCENTILE, axis=0)
    if passed.sum() < MIN_KEEP:
        passed = np.zeros(len(keep), dtype=bool)
        passed[np.argsort(-true)[:MIN_KEEP]] = True
    return list(keep[passed])


def make_model2(cfg: Config2) -> FitPredict:
    lr = (make_pairwise_logistic if cfg.pairwise else make_logistic)(clip=cfg.clip)
    gbm = make_lgbm(
        monotone_increasing=tuple(US_CHANGE_COLUMNS + US_VOLNORM_COLUMNS) if cfg.monotone else (),
        monotone_decreasing=tuple(FOOD_WDDEV_COLUMNS) if cfg.mean_reversion else (),
    )

    def weights(X: pd.DataFrame):
        return decay_weights(X.index) if cfg.decay else None

    def base(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        w = weights(X_train)
        return rank_average([m(X_train, y_train, X_pred, sample_weight=w) for m in (lr, gbm)])

    def main(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
        cols = list(X_train.columns)
        if cfg.drop_general:
            cols = [c for c in cols if c not in GENERAL_TRAFFIC_COLUMNS]
        if cfg.null_select:
            cols = null_importance_columns(X_train[cols], y_train, weights(X_train))
        return base(X_train[cols], y_train, X_pred[cols])

    if cfg.hedge_core:  # us_core 보험 모델은 변수 선택 없이 그대로 둔다
        return make_ensemble([main, on_columns(US_CORE_COLUMNS, base)])
    return main


@dataclass
class Scores2:
    cv: np.ndarray  # cv5 fold별 AUC
    pm: np.ndarray  # pm4 창별 200일 통째 AUC
    monthu: float  # cv5 fold별 월·목 요일 안 AUC 평균 (G1·G3 메커니즘)
    highvol: float  # cv5 fold별 '변동성 높은 날' AUC 평균 (G4 메커니즘)


def score2(cfg: Config2, train: pd.DataFrame, test: pd.DataFrame, target: str) -> Scores2:
    X, _ = build(train, test, list(cfg.families))
    y, weekday = train[target], train["Weekday"].to_numpy()
    vol = build(train, test, ["us_trend_vol"])[0]["US_vol20"].to_numpy()
    model = make_model2(cfg)

    def auc_if_possible(yy: np.ndarray, pp: np.ndarray) -> float | None:
        return roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else None

    split = cv_splits(len(X))
    preds = _predict(model, X, y, split)
    cv, monthu, highvol = [], [], []
    for p, (tr, va) in zip(preds, split):
        yy = y.iloc[va].to_numpy()
        cv.append(roc_auc_score(yy, p))
        for day in (MON, THU):
            m = weekday[va] == day
            if (a := auc_if_possible(yy[m], p[m])) is not None:
                monthu.append(a)
        m = vol[va] > np.nanmedian(vol[tr])
        if (a := auc_if_possible(yy[m], p[m])) is not None:
            highvol.append(a)

    pm_split = private_mimic_splits(len(X))
    pm = [roc_auc_score(y.iloc[va], p) for p, (_, va) in zip(_predict(model, X, y, pm_split), pm_split)]
    return Scores2(np.array(cv), np.array(pm), float(np.mean(monthu)), float(np.mean(highvol)))


@dataclass(frozen=True)
class Candidate:
    name: str
    track: str  # "B": 올라야 채택, "A": 안 지면 채택
    apply: Callable[[Config2], Config2]
    mechanism: str | None


CANDIDATES2 = {
    "G1": Candidate("G1 식품 L1 궤적·주기 분해", "B", lambda c: replace(c, families=c.families + ("food_l1",)), "monthu"),
    "G2": Candidate("G2 일반 트래픽 입력 제외", "A", lambda c: replace(c, drop_general=True), None),
    "G3": Candidate("G3 평균 회귀 방향 고정", "B", lambda c: replace(c, mean_reversion=True), "monthu"),
    "G4": Candidate("G4 변동성 대비 미국 변화", "B", lambda c: replace(c, families=c.families + ("us_volnorm",)), "highvol"),
    "G5": Candidate("G5 순위 손실 (쌍 비교 로지스틱)", "B", lambda c: replace(c, pairwise=True), None),
    "N9": Candidate("N9 가짜 정답 비교 변수 선택", "B", lambda c: replace(c, null_select=True), None),
    "T10": Candidate("T10 최근 비중 (반감기 365일)", "B", lambda c: replace(c, decay=True), None),
}
ORDER = {"Target_1": ["G4", "N9", "T10", "G5"], "Target_2": ["G1", "G3", "N9", "T10", "G5", "G2"]}
MECHANISM_NAMES = {"monthu": "월·목 요일 안 AUC", "highvol": "변동성 높은 날 AUC"}
REPORTED_MECHANISM = {"Target_1": "highvol", "Target_2": "monthu"}


def decide2(track: str, new: Scores2, ref: Scores2, a_base: Scores2 | None, mechanism: str | None) -> tuple[bool, str]:
    """사전 등록 규칙 (#13). (채택 여부, 이유)."""
    if track == "B":
        checks = {
            f"cv5 {int((new.cv > ref.cv).sum())}/5": beats(new.cv, ref.cv),
            "pm4 평균": new.pm.mean() > ref.pm.mean(),
        }
        if mechanism:
            checks[MECHANISM_NAMES[mechanism]] = getattr(new, mechanism) > getattr(ref, mechanism)
    else:
        checks = {
            "cv5 평균 −0.002 이내": new.cv.mean() >= a_base.cv.mean() - A_TOLERANCE,
            "pm4 평균 −0.002 이내": new.pm.mean() >= a_base.pm.mean() - A_TOLERANCE,
        }
    failed = [name for name, ok in checks.items() if not ok]
    return (not failed, "채택" if not failed else "기각: " + ", ".join(failed) + " 미충족")


def _check_start(target: str, start: Scores2) -> None:
    """검산: 시작점 cv5·pm4가 features_v2 기록과 정확히 같아야 한다."""
    for scheme, values in [("cv5", start.cv), ("pm4", start.pm)]:
        recorded = pd.read_csv(EXPERIMENT_DIR / f"features_v2_{scheme}.csv").query("target == @target")["auc"].to_numpy()
        if not np.allclose(recorded, values, atol=1e-12):
            raise RuntimeError(f"검산 실패 ({scheme}): 시작점 {values} != 기록 {recorded}")


def _describe(cfg: Config2, start: Config2) -> str:
    added = [f for f in cfg.families if f not in start.families]
    flags = [k for k in ("drop_general", "mean_reversion", "pairwise", "null_select", "decay") if getattr(cfg, k)]
    return "v2" + "".join(f" +{x}" for x in added + flags)


def _row(step: str, cfg: Config2, start: Config2, s: Scores2, ref: Scores2 | None, mechanism: str, decision: str) -> dict:
    return {
        "단계": step,
        "구성": _describe(cfg, start),
        "cv5 평균": s.cv.mean(),
        "cv5 fold별": " ".join(f"{v:.3f}" for v in s.cv),
        "개선 fold": "-" if ref is None else f"{int((s.cv > ref.cv).sum())}/5",
        "pm4 평균": s.pm.mean(),
        "pm4 창별": " ".join(f"{v:.3f}" for v in s.pm),
        MECHANISM_NAMES[mechanism]: getattr(s, mechanism),
        "결정": decision,
    }


def run(target: str) -> Config2:
    train, test = load_data()
    cache: dict[Config2, Scores2] = {}

    def scored(cfg: Config2) -> Scores2:
        if cfg not in cache:
            cache[cfg] = score2(cfg, train, test, target)
        return cache[cfg]

    mech = REPORTED_MECHANISM[target]
    start = Config2(**asdict(FINAL_V2[target]))
    s0 = scored(start)
    _check_start(target, s0)
    rows = [_row("시작점 (v2)", start, start, s0, None, mech, "-")]

    # 1. 독립 시험
    passed: dict[str, Scores2] = {}
    for key in ORDER[target]:
        cand = CANDIDATES2[key]
        cfg = cand.apply(start)
        s = scored(cfg)
        ok, why = decide2(cand.track, s, s0, s0, cand.mechanism)
        rows.append(_row(f"1. {cand.name}", cfg, start, s, s0, mech, why))
        if ok:
            passed[key] = s

    # 2. 합치기 (통과한 후보만, 정한 순서로)
    current_cfg, current, a_base, adopted = start, s0, None, []
    for key in [k for k in ORDER[target] if k in passed]:
        cand = CANDIDATES2[key]
        if cand.track == "A" and a_base is None:
            a_base = current
        cfg = cand.apply(current_cfg)
        s = scored(cfg)
        ok, why = decide2(cand.track, s, current, a_base, cand.mechanism)
        rows.append(_row(f"2. + {key}", cfg, start, s, current, mech, why))
        if ok:
            current_cfg, current = cfg, s
            adopted.append(key)

    # 3. 최종 확인
    final_cfg = current_cfg
    if final_cfg != start:
        if any(CANDIDATES2[k].track == "B" for k in adopted):
            ok = beats(current.cv, s0.cv) and current.pm.mean() > s0.pm.mean()
        else:
            ok = decide2("A", current, s0, s0, None)[0]
        rows.append(_row("3. 최종 확인 (vs v2)", final_cfg, start, current, s0, mech, "통과" if ok else "실패"))
        if not ok:
            best = max(passed, key=lambda k: passed[k].pm.mean())
            final_cfg = CANDIDATES2[best].apply(start)
            rows.append(_row(f"3. 대체: {best} 단독", final_cfg, start, passed[best], s0, mech, "채택"))

    summary = pd.DataFrame(rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPERIMENT_DIR / f"screening_{target}.csv", index=False)
    (EXPERIMENT_DIR / f"screening_{target}.json").write_text(json.dumps(asdict(final_cfg), ensure_ascii=False, indent=2))
    with pd.option_context("display.width", 300, "display.max_colwidth", 60, "display.float_format", "{:.4f}".format):
        print(f"[{target}] 독립 시험 → 합치기 → 최종 확인\n{summary.to_string(index=False)}\n\n최종 설정: {final_cfg}")
    return final_cfg


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command not in TARGETS:
        sys.exit(f"사용법: python -m src.screening {{{'|'.join(TARGETS)}}}")
    run(command)
