"""규칙 기반 진단(A~D 플래그, 유형, 처방)과 기준값 민감도 분석."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from . import config
from .config import Thresholds


def flags(f: pd.DataFrame, t: Thresholds = config.DEFAULT_THRESHOLDS) -> pd.DataFrame:
    return pd.DataFrame({
        "A_stay_weak": f["lodging_share_pct"] < t.lodging_share_pct,
        "B_seasonal": f["seasonality_ratio"] >= t.seasonality_ratio,
        "C_low_spend": f["spend_per_outsider"] < t.spend_per_visitor,
        "D_underused": f["resource_spend_ratio"] < t.resource_ratio,
    }, index=f.index)


def classify(row: pd.Series, lodging_pct: float, t: Thresholds = config.DEFAULT_THRESHOLDS) -> str:
    """플래그 조합 -> 유형. 위에서부터 먼저 맞는 규칙을 쓴다."""
    if row["D_underused"] and row["C_low_spend"]:
        return "자원 미활용·통과형"
    if row["B_seasonal"] and lodging_pct >= t.resort_lodging_pct:
        return "계절 의존 리조트형"
    if row["B_seasonal"]:
        return "성수기 집중형"
    if row["D_underused"]:
        return "자원 대비 저소비형"
    if row["A_stay_weak"]:
        return "당일치기형"
    return "체류·소비 거점형"


def diagnose(f: pd.DataFrame, t: Thresholds = config.DEFAULT_THRESHOLDS) -> pd.DataFrame:
    fl = flags(f, t)
    out = f.join(fl)
    out["type"] = [classify(fl.loc[s], f.loc[s, "lodging_share_pct"], t) for s in f.index]
    out["type_order"] = out["type"].map({k: i + 1 for i, k in enumerate(config.TYPE_ORDER)})
    out["prescription"] = out["type"].map(config.PRESCRIPTION)
    out["group"] = out["type"].map(config.GROUP_OF_TYPE)
    out["group_order"] = out["group"].map({k: i + 1 for i, k in enumerate(config.GROUP_ORDER)})
    out["group_prescription"] = out["group"].map(config.GROUP_PRESCRIPTION)
    return out.sort_values(["type_order", "outsider_visitors"], ascending=[True, False])


def sensitivity(f: pd.DataFrame, step: float = 0.2, base: Thresholds = config.DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """기준값 4개를 각각 -step, 0, +step 만큼 바꾼 3^4=81가지 경우에서
    시군별로 기본 유형이 유지되는 비율과 D 플래그가 켜지는 비율을 계산한다."""
    base_type = diagnose(f, base)["type"]
    names = ["lodging_share_pct", "seasonality_ratio", "spend_per_visitor", "resource_ratio"]
    keep = pd.Series(0, index=f.index, dtype=float)
    d_on = pd.Series(0, index=f.index, dtype=float)
    n = 0
    for mult in itertools.product([1 - step, 1.0, 1 + step], repeat=len(names)):
        t = replace(base, **{k: getattr(base, k) * m for k, m in zip(names, mult)})
        d = diagnose(f, t)
        keep += (d["type"] == base_type.reindex(d.index)).reindex(f.index)
        d_on += d["D_underused"].reindex(f.index)
        n += 1
    return pd.DataFrame({"base_type": base_type.reindex(f.index), "type_kept_pct": keep / n * 100,
                         "D_on_pct": d_on / n * 100}).sort_values("type_kept_pct")
