"""덩어리 4.7: 후보 백로그 실행 (docs/EXPERIMENTS.md #10).

B(구조, 올라야 채택)를 먼저 하고, A(견고성, 안 지면 채택)는 B 결과 위에서 진행한다.
저울은 cv5 + pm4(Private 흉내 200일 창 4개)다. 판정 규칙은 사전 등록대로 decide()에 있다.

실행:
  python -m src.backlog Target_1
  python -m src.backlog Target_2
  python -m src.backlog final      # 확정 설정(FINAL_V2)으로 채점 + 제출 파일
"""

import json
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import ROOT, TARGETS, load_data
from src.experiments import SELECTED
from src.features import US_CHANGE_COLUMNS, US_CORE_COLUMNS, build
from src.models import make_ensemble, make_lgbm, make_logistic, on_columns
from src.submission import build_submission, write_submissions
from src.validation import FitPredict, Split, beats, cv_splits, evaluate, holdout_split, private_mimic_splits, summarize

EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
A_TOLERANCE = 0.002
MON, THU = 2, 5
# #9 진단 1에서 소수 셋째 자리로 기록한 시작점 pm4 (검산용)
START_PM4 = {"Target_1": [0.620, 0.548, 0.605, 0.680], "Target_2": [0.952, 0.975, 0.977, 0.959]}


@dataclass(frozen=True)
class Config:
    families: tuple[str, ...]
    clip: bool = False  # A1
    monotone: bool = False  # A2
    hedge_core: bool = False  # A3


def make_model(cfg: Config) -> FitPredict:
    lr = make_logistic(clip=cfg.clip)
    gbm = make_lgbm(monotone_increasing=tuple(US_CHANGE_COLUMNS) if cfg.monotone else ())
    model = make_ensemble([lr, gbm])
    if cfg.hedge_core:
        model = make_ensemble([model, on_columns(US_CORE_COLUMNS, model)])
    return model


# (이름, 트랙, 설정 변경, B의 메커니즘 지표)
STEPS = {
    "Target_1": [
        ("B2 60거래일 모멘텀", "B", lambda c: replace(c, families=c.families + ("us_momentum60",)), "regime"),
        ("A1 입력 자르기", "A", lambda c: replace(c, clip=True), None),
        ("A2 단조 제약", "A", lambda c: replace(c, monotone=True), None),
        ("A3 피처셋 앙상블", "A", lambda c: replace(c, hedge_core=True), None),
    ],
    "Target_2": [
        ("B1 최근 요일 패턴", "B", lambda c: replace(c, families=c.families + ("weekday_profile",)), "monthu"),
        ("A1 입력 자르기", "A", lambda c: replace(c, clip=True), None),
    ],
}
MECHANISM_NAMES = {"regime": "pm4 시기 성분", "monthu": "월·목 요일 안 AUC"}


@dataclass
class Scores:
    cv: np.ndarray  # cv5 fold별 AUC
    pm: np.ndarray  # pm4 창별 200일 통째 AUC
    regime: float  # pm4 창별 (통째 AUC - 100일 반쪽 AUC 평균)의 평균
    monthu: float  # cv5 fold별 월·목 요일 안 AUC의 평균


def _predict(model: FitPredict, X: pd.DataFrame, y: pd.Series, splits: list[Split]) -> list[np.ndarray]:
    preds = []
    for tr, va in splits:
        p = np.asarray(model(X.iloc[tr], y.iloc[tr], X.iloc[va]), dtype=float)
        if p.shape != (len(va),) or not np.isfinite(p).all():
            raise ValueError("예측값 형태가 잘못되었거나 유한하지 않습니다.")
        preds.append(p)
    return preds


def score(cfg: Config, train: pd.DataFrame, test: pd.DataFrame, target: str) -> Scores:
    X, _ = build(train, test, list(cfg.families))
    y, weekday = train[target], train["Weekday"].to_numpy()
    model = make_model(cfg)

    cv_split = cv_splits(len(X))
    cv_pred = _predict(model, X, y, cv_split)
    cv = np.array([roc_auc_score(y.iloc[va], p) for p, (_, va) in zip(cv_pred, cv_split)])
    within = []
    for p, (_, va) in zip(cv_pred, cv_split):
        yy, wd = y.iloc[va].to_numpy(), weekday[va]
        for day in (MON, THU):
            m = wd == day
            if 0 < yy[m].sum() < m.sum():  # 두 클래스가 모두 있어야 AUC 계산 가능
                within.append(roc_auc_score(yy[m], p[m]))

    pm_split = private_mimic_splits(len(X))
    pm_pred = _predict(model, X, y, pm_split)
    pm, regime = [], []
    for p, (_, va) in zip(pm_pred, pm_split):
        yy, half = y.iloc[va].to_numpy(), len(va) // 2
        pooled = roc_auc_score(yy, p)
        halves = (roc_auc_score(yy[:half], p[:half]) + roc_auc_score(yy[half:], p[half:])) / 2
        pm.append(pooled)
        regime.append(pooled - halves)
    return Scores(cv=cv, pm=np.array(pm), regime=float(np.mean(regime)), monthu=float(np.mean(within)))


def decide(track: str, new: Scores, current: Scores, a_base: Scores, mechanism: str | None) -> tuple[bool, str]:
    """사전 등록한 판정 규칙. (채택 여부, 이유)."""
    if track == "B":
        checks = {
            f"cv5 {int((new.cv > current.cv).sum())}/5": beats(new.cv, current.cv),
            "pm4 평균": new.pm.mean() > current.pm.mean(),
            MECHANISM_NAMES[mechanism]: getattr(new, mechanism) > getattr(current, mechanism),
        }
    else:
        checks = {
            "cv5 평균 −0.002 이내": new.cv.mean() >= a_base.cv.mean() - A_TOLERANCE,
            "pm4 평균 −0.002 이내": new.pm.mean() >= a_base.pm.mean() - A_TOLERANCE,
        }
    failed = [name for name, ok in checks.items() if not ok]
    return (not failed, "채택" if not failed else "기각: " + ", ".join(failed) + " 미충족")


def _check_start(target: str, start: Scores) -> None:
    """검산: 시작점 cv5는 #4(features_v1) 기록과 정확히, pm4는 #9 진단 1 기록과 반올림 범위 안에서 같아야 한다."""
    if np.abs(start.pm - np.array(START_PM4[target])).max() > 5e-4:
        raise RuntimeError(f"검산 실패: 시작점 pm4 {start.pm} != 기록 {START_PM4[target]}")
    path = EXPERIMENT_DIR / "features_v1_cv5.csv"
    if not path.exists():
        print(f"(검산 건너뜀: {path.name} 없음 — python -m src.experiments final 먼저 실행)")
        return
    recorded = pd.read_csv(path).query("target == @target")["auc"].to_numpy()
    if not np.allclose(recorded, start.cv, atol=1e-12):
        raise RuntimeError(f"검산 실패: 시작점 cv5 {start.cv} != 기록 {recorded}")


def _row(step: str, cfg: Config, s: Scores, compared: Scores | None, mechanism: str, decision: str) -> dict:
    return {
        "단계": step,
        "구성": "+".join(cfg.families) + "".join(f" [{k}]" for k in ("clip", "monotone", "hedge_core") if getattr(cfg, k)),
        "cv5 평균": s.cv.mean(),
        "cv5 fold별": " ".join(f"{v:.3f}" for v in s.cv),
        "개선 fold": "-" if compared is None else f"{int((s.cv > compared.cv).sum())}/5",
        "pm4 평균": s.pm.mean(),
        "pm4 창별": " ".join(f"{v:.3f}" for v in s.pm),
        MECHANISM_NAMES[mechanism]: getattr(s, mechanism),
        "결정": decision,
    }


def run(target: str) -> Config:
    train, test = load_data()
    mechanism = next(m for *_, m in STEPS[target] if m)
    current_cfg = Config(tuple(SELECTED[target]))
    current = score(current_cfg, train, test, target)
    _check_start(target, current)
    rows = [_row("시작점 (features_v1)", current_cfg, current, None, mechanism, "-")]

    a_base = None
    for name, track, change, step_mechanism in STEPS[target]:
        if track == "A" and a_base is None:
            a_base = current  # A 시작 직전 상태
        cfg = change(current_cfg)
        new = score(cfg, train, test, target)
        ok, decision = decide(track, new, current, a_base, step_mechanism)
        rows.append(_row(name, cfg, new, current, mechanism, decision))
        if ok:
            current_cfg, current = cfg, new

    summary = pd.DataFrame(rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPERIMENT_DIR / f"backlog_{target}.csv", index=False)
    (EXPERIMENT_DIR / f"backlog_{target}.json").write_text(json.dumps(asdict(current_cfg), ensure_ascii=False, indent=2))
    with pd.option_context("display.width", 300, "display.max_colwidth", 90, "display.float_format", "{:.4f}".format):
        print(f"[{target}] 후보 백로그 결과\n{summary.to_string(index=False)}\n\n최종 설정: {current_cfg}")
    return current_cfg


# 백로그 실행 결과로 확정한 설정 (docs/EXPERIMENTS.md #10). run_final이 실험 결과 파일과 일치하는지 확인한다.
FINAL_V2 = {
    "Target_1": Config(("us_change", "us_trend_vol"), clip=True, monotone=True, hedge_core=True),
    "Target_2": Config(("weekday", "traffic", "weekday_dev"), clip=True),
}
FINAL_RUN = "features_v2"
PUBLIC_ROUND = 3


def run_final() -> None:
    """FINAL_V2를 cv5·holdout200·pm4로 채점하고, train 전체로 재학습해 제출 파일을 만든다."""
    for target in TARGETS:
        recorded = json.loads((EXPERIMENT_DIR / f"backlog_{target}.json").read_text())
        if recorded != json.loads(json.dumps(asdict(FINAL_V2[target]))):
            raise RuntimeError(f"{target}: FINAL_V2가 최근 백로그 결과와 다릅니다. 결과를 확인하고 FINAL_V2를 갱신하세요.")
    train, test = load_data()
    for scheme, make_splits in [("cv5", cv_splits), ("holdout200", holdout_split), ("pm4", private_mimic_splits)]:
        scores = []
        for target in TARGETS:
            cfg = FINAL_V2[target]
            X, _ = build(train, test, list(cfg.families))
            scores.append(evaluate(make_model(cfg), X, train[[target]], make_splits(len(X))))
        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(EXPERIMENT_DIR / f"{FINAL_RUN}_{scheme}.csv", index=False)
        print(f"[{scheme}]\n{summarize(scores).round(4).to_string()}\n")

    preds = {}
    for target in TARGETS:
        cfg = FINAL_V2[target]
        X, X_test = build(train, test, list(cfg.families))
        preds[target] = make_model(cfg)(X, train[target], X_test)
    public_path, private_path = write_submissions(build_submission(test, preds), FINAL_RUN, PUBLIC_ROUND)
    print(f"[제출 파일] {public_path.relative_to(ROOT)}, {private_path.relative_to(ROOT)}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command in TARGETS:
        run(command)
    elif command == "final":
        run_final()
    else:
        sys.exit(f"사용법: python -m src.backlog {{{'|'.join(TARGETS)}|final}}")
