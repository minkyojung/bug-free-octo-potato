"""강사 제공 베이스라인(reference/Mini-Project_Baseline.ipynb) 재구현.

노트북의 전처리 단계를 그대로 따른다.
1. 학습 구간에서 고유값 1개 이하 컬럼 제거 (KE_QR체크인)
2. 학습 구간 중앙값으로 결측 대체 (train 결측이 0개라 실제로는 변화 없음)
3. 범주형 원-핫 인코딩: 범주형 컬럼이 없어 생략
4. RandomForest, 타깃별 독립 학습

실행: python -m src.baseline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data import ROOT, TARGETS, feature_columns, load_data
from src.submission import build_submission, write_submissions
from src.validation import cv_splits, evaluate, holdout_split, summarize

RUN_NAME = "baseline"
EXPERIMENT_DIR = ROOT / "outputs" / "experiments"
RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> np.ndarray:
    keep = X_train.columns[X_train.nunique(dropna=False) > 1]
    medians = X_train[keep].median()
    model = RandomForestClassifier(**RF_PARAMS).fit(X_train[keep].fillna(medians), y_train)
    return model.predict_proba(X_pred[keep].fillna(medians))[:, 1]


def notebook_split_auc(train: pd.DataFrame) -> dict[str, float]:
    """노트북과 같은 70/30 분할(gap 없음)의 검증 AUC. 재현 확인용."""
    X, split = train[feature_columns(train)], int(len(train) * 0.7)
    return {
        t: roc_auc_score(train[t].iloc[split:], fit_predict(X.iloc[:split], train[t].iloc[:split], X.iloc[split:]))
        for t in TARGETS
    }


def main() -> None:
    train, test = load_data()
    X, y = train[feature_columns(train)], train[TARGETS]

    print("[재현] 노트북 70/30 분할 AUC:", {t: round(v, 4) for t, v in notebook_split_auc(train).items()})

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for scheme, splits in [("cv5", cv_splits(len(X))), ("holdout200", holdout_split(len(X)))]:
        scores = evaluate(fit_predict, X, y, splits)
        scores.to_csv(EXPERIMENT_DIR / f"{RUN_NAME}_{scheme}.csv", index=False)
        print(f"\n[{scheme}]\n{summarize(scores).round(4).to_string()}")

    # 최종 예측: train 전체로 재학습
    preds = {t: fit_predict(X, y[t], test[X.columns]) for t in TARGETS}
    public_path, private_path = write_submissions(build_submission(test, preds), RUN_NAME, public_round=1)
    print(f"\n[제출 파일] {public_path.relative_to(ROOT)}, {private_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
