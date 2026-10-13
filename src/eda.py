"""EDA용 단변량 AUC 통계.

변수 하나를 점수로 쓴 AUC를 수백 개 변수에 대해 한 번에 계산하고,
타깃을 시간축으로 순환 이동시킨 "우연 기준선"과 비교한다.
"""

from typing import Callable

import numpy as np
from scipy.stats import rankdata

Scorer = Callable[[np.ndarray], np.ndarray]  # 0/1 타깃 -> 열별 AUC


def _auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Mann-Whitney U 통계량(열별)과 n1*n0. 동점은 평균 순위."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    return y @ ranks - n1 * (n1 + 1) / 2, n1 * n0


def auc_scorer(X: np.ndarray) -> Scorer:
    """X의 각 열을 점수로 썼을 때의 AUC를 계산하는 함수를 반환한다 (순위는 한 번만 계산)."""
    ranks = rankdata(X, axis=0)

    def score(y: np.ndarray) -> np.ndarray:
        u, pairs = _auc_from_ranks(ranks, y)
        return u / pairs

    return score


def stratified_auc_scorer(X: np.ndarray, groups: np.ndarray) -> Scorer:
    """같은 그룹(예: 요일) 안의 쌍만 비교하는 AUC. 그룹 간 차이의 영향을 제거한다."""
    parts = [(groups == g, rankdata(X[groups == g], axis=0)) for g in np.unique(groups)]

    def score(y: np.ndarray) -> np.ndarray:
        u_total, pairs_total = 0.0, 0
        for mask, ranks in parts:
            u, pairs = _auc_from_ranks(ranks, y[mask])
            if pairs:  # 한쪽 클래스만 있는 그룹은 비교할 쌍이 없음
                u_total, pairs_total = u_total + u, pairs_total + pairs
        return u_total / pairs_total

    return score


def shift_null_max(
    score: Scorer, y: np.ndarray, n_shifts: int = 200, step: int = 1, min_shift: int = 60, seed: int = 0
) -> np.ndarray:
    """타깃을 순환 이동시켜 얻은 '우연히 나오는 최대 |AUC-0.5|'의 분포 (길이 n_shifts).

    순환 이동은 타깃의 자기상관을 보존하면서 변수와의 실제 관계만 끊는다.
    step=7이면 7일 배수로만 이동해 요일 정렬을 유지한다.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift // step + 1, (n - min_shift) // step, n_shifts) * step
    return np.array([np.abs(score(np.roll(y, s)) - 0.5).max() for s in shifts])
