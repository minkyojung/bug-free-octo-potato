"""보고서용 정적 그림 스타일."""

import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.data import ROOT

FIG_DIR = ROOT / "outputs" / "figures"

# 범주형 팔레트 앞 3칸 (색약 대비 검증된 순서). 타깃별 색은 모든 그림에서 고정.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
COLORS = {"Target_1": SERIES[0], "Target_2": SERIES[1]}
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SHADE = "#f0efec"  # 구간 강조 배경

KOREAN_FONTS = ["AppleGothic", "Malgun Gothic", "NanumGothic"]


def setup() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in available), None)
    if font is None:
        raise RuntimeError(f"한글 폰트가 없습니다. 다음 중 하나를 설치하세요: {KOREAN_FONTS}")
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "text.color": INK,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
