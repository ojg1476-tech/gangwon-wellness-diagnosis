"""그림 공통 설정 (한글 글꼴, 색)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

KOREAN_FONTS = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"]

# 진단 유형 색 (색각이상에도 구분되는 순서)
TYPE_COLORS = {
    "자원 미활용·통과형": "#C8553D",
    "계절 의존 리조트형": "#2D6A8E",
    "성수기 집중형": "#E0A030",
    "자원 대비 저소비형": "#7A5195",
    "당일치기형": "#3C9D6B",
    "체류·소비 거점형": "#8C8C8C",
}
ACCENT = "#2D6A8E"
WARN = "#C8553D"
MUTED = "#B8B8B8"


def setup():
    """설치된 한글 글꼴을 찾아 기본으로 지정한다.
    코랩에서는 먼저 `!apt-get -qq install fonts-nanum` 후 런타임을 다시 시작한다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            mpl.rcParams["font.family"] = name
            break
    mpl.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
    })
    return mpl.rcParams["font.family"]


def short(name: str) -> str:
    """'춘천시' -> '춘천'"""
    return name[:-1] if name.endswith(("시", "군")) else name


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=160)
    return path


__all__ = ["setup", "short", "save", "TYPE_COLORS", "ACCENT", "WARN", "MUTED", "plt"]
