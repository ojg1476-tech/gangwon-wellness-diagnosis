"""규칙 기반 진단(A~D 플래그, 유형, 처방)과 기준값 민감도 분석."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from . import config
from .config import Thresholds


def gangwon_spend_per_outsider(f: pd.DataFrame) -> float:
    """강원 전체 외지인 1명당 관광 지출(원) = 전체 지출 ÷ 전체 외지인."""
    return float(f["spend_domestic_k"].sum() * 1000 / f["outsider_visitors"].sum())


def spend_cutoff(f: pd.DataFrame, t: Thresholds = config.DEFAULT_THRESHOLDS, gangwon_spend: float | None = None) -> float:
    """C 기준을 원 단위로 (강원 평균 × 비율)."""
    g = gangwon_spend if gangwon_spend is not None else gangwon_spend_per_outsider(f)
    return g * t.spend_vs_gangwon


def flags(f: pd.DataFrame, t: Thresholds = config.DEFAULT_THRESHOLDS, gangwon_spend: float | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "A_stay_weak": f["lodging_share_pct"] < t.lodging_share_pct,
        "B_seasonal": f["seasonality_ratio"] >= t.seasonality_ratio,
        "C_low_spend": f["spend_per_outsider"] < spend_cutoff(f, t, gangwon_spend),
        "D_underused": f["resource_spend_ratio"] < t.resource_ratio,
    }, index=f.index)


def classify(row: pd.Series, x: pd.Series, t: Thresholds, gangwon_spend: float) -> str:
    """플래그 조합 -> 유형. 위에서부터 먼저 맞는 규칙을 쓴다.
    거점은 '문제 없음'이 아니라 근거(강원 평균 이상 지출, 자원 몫 이상)가 있을 때만 붙인다."""
    if row["D_underused"] and row["C_low_spend"]:
        return "자원 미활용·통과형"
    if row["B_seasonal"] and x["lodging_share_pct"] >= t.resort_lodging_pct:
        return "계절 의존 리조트형"
    if row["B_seasonal"]:
        return "성수기 집중형"
    if row["D_underused"]:
        return "자원 대비 저소비형"
    if row["A_stay_weak"]:
        return "당일치기형"
    if (x["spend_per_outsider"] >= gangwon_spend * t.hub_spend_vs_gangwon
            and x["resource_spend_ratio"] >= t.hub_resource_ratio):
        return "체류·소비 거점형"
    return "관찰 대상"


def diagnose(f: pd.DataFrame, t: Thresholds = config.DEFAULT_THRESHOLDS, gangwon_spend: float | None = None) -> pd.DataFrame:
    g = gangwon_spend if gangwon_spend is not None else gangwon_spend_per_outsider(f)
    fl = flags(f, t, g)
    out = f.join(fl)
    out["type"] = [classify(fl.loc[s], f.loc[s], t, g) for s in f.index]
    out["type_order"] = out["type"].map({k: i + 1 for i, k in enumerate(config.TYPE_ORDER)})
    out["prescription"] = out["type"].map(config.PRESCRIPTION)
    out["group"] = out["type"].map(config.GROUP_OF_TYPE)
    out["group_order"] = out["group"].map({k: i + 1 for i, k in enumerate(config.GROUP_ORDER)})
    out["group_prescription"] = out["group"].map(config.GROUP_PRESCRIPTION)
    return out.sort_values(["type_order", "outsider_visitors"], ascending=[True, False])


THRESHOLD_FIELDS = ["lodging_share_pct", "seasonality_ratio", "spend_vs_gangwon", "resource_ratio"]
METRIC_OF = {"lodging_share_pct": "lodging_share_pct", "seasonality_ratio": "seasonality_ratio",
             "spend_vs_gangwon": "spend_per_outsider", "resource_ratio": "resource_spend_ratio"}


def threshold_margins(f: pd.DataFrame, t: Thresholds = config.DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """기준값마다 바로 아래·위 시군과 그 간격(%). 간격이 클수록 기준이 조금 바뀌어도 결과가 안 변한다."""
    rows = []
    for name in THRESHOLD_FIELDS:
        v = f[METRIC_OF[name]]
        cut = spend_cutoff(f, t) if name == "spend_vs_gangwon" else getattr(t, name)
        below, above = v[v < cut], v[v >= cut]
        rows.append({
            "기준": name, "기준값": cut,
            "바로 아래": f"{below.idxmax()} {below.max():.3f}", "바로 위": f"{above.idxmin()} {above.min():.3f}",
            "가까운 쪽 간격(%)": min(np.log(cut / below.max()), np.log(above.min() / cut)) * 100,
        })
    return pd.DataFrame(rows)


def sensitivity(f: pd.DataFrame, step: float = 0.2, base: Thresholds = config.DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """기준값 4개를 각각 -step, 0, +step 만큼 바꾼 3^4=81가지 경우에서
    시군별로 기본 유형이 유지되는 비율과 D 플래그가 켜지는 비율을 계산한다."""
    base_type = diagnose(f, base)["type"]
    keep = pd.Series(0, index=f.index, dtype=float)
    d_on = pd.Series(0, index=f.index, dtype=float)
    n = 0
    for mult in itertools.product([1 - step, 1.0, 1 + step], repeat=len(THRESHOLD_FIELDS)):
        t = replace(base, **{k: getattr(base, k) * m for k, m in zip(THRESHOLD_FIELDS, mult)})
        d = diagnose(f, t)
        keep += (d["type"] == base_type.reindex(d.index)).reindex(f.index)
        d_on += d["D_underused"].reindex(f.index)
        n += 1
    return pd.DataFrame({"base_type": base_type.reindex(f.index), "type_kept_pct": keep / n * 100,
                         "D_on_pct": d_on / n * 100}).sort_values("type_kept_pct")
