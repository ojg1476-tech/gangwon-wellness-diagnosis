"""시·군 단위 지표 만들기 (가설 A~D와 외국인 지표)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _in_period(df: pd.DataFrame, period=config.PERIOD) -> pd.DataFrame:
    return df[(df["ym"] >= period[0]) & (df["ym"] <= period[1])]


def monthly_spending(dl: dict, period=config.PERIOD) -> pd.DataFrame:
    """시군 x 월 내국인 관광 총지출(천원)과 월평균 대비 비율."""
    sp = _in_period(dl["spending_domestic"], period)
    m = sp[sp["category"] == "관광총소비"][["sigun", "ym", "spend_k"]].copy()
    m["ratio_to_mean"] = m["spend_k"] / m.groupby("sigun")["spend_k"].transform("mean")
    return m.sort_values(["sigun", "ym"]).reset_index(drop=True)


def category_mix(dl: dict, who: str = "domestic", period=config.PERIOD) -> pd.DataFrame:
    """시군 x 업종 지출(천원) 피벗. 관광총소비 열 포함."""
    sp = _in_period(dl[f"spending_{who}"], period)
    return sp.pivot_table(index="sigun", columns="category", values="spend_k", aggfunc="sum").fillna(0)


def build_features(dl: dict, api: dict, period=config.PERIOD) -> pd.DataFrame:
    """시군별 지표 표 한 장."""
    vis = _in_period(dl["visitors_domestic"], period)
    v = vis.pivot_table(index="sigun", columns="visitor_type", values="visitors", aggfunc="sum")

    mix = category_mix(dl, "domestic", period)
    total = mix["관광총소비"]
    lodging = mix[[c for c in config.LODGING_CATEGORIES if c in mix]].sum(axis=1)

    m = monthly_spending(dl, period).pivot(index="sigun", columns="ym", values="spend_k")
    n_months = m.notna().sum(axis=1)

    f = pd.DataFrame(index=v.index)
    f["outsider_visitors"] = v["outsider"]
    f["local_visitors"] = v["local"]
    f["outsider_share_pct"] = v["outsider"] / v["total"] * 100
    f["spend_domestic_k"] = total
    f["spend_per_outsider"] = total * 1000 / v["outsider"]          # 원
    f["lodging_share_pct"] = lodging / total * 100
    f["lodging_per_outsider"] = lodging * 1000 / v["outsider"]       # 원
    f["seasonality_ratio"] = m.max(axis=1) / m.min(axis=1)
    f["low3_share_pct"] = m.apply(lambda r: r.nsmallest(3).sum(), axis=1) / m.sum(axis=1) * 100
    f["peak_month"] = m.idxmax(axis=1)
    f["n_months"] = n_months

    # 외국인: 방문자와 지출 기간이 겹치는 달만 사용
    fv = dl["visitors_foreign"]
    fs = dl["spending_foreign"]
    fs_tot = fs[fs["category"] == "관광총소비"]
    months = sorted(set(fv["ym"]) & set(fs_tot["ym"]) & set(range(period[0], period[1] + 1)))
    fvv = fv[fv["ym"].isin(months)].groupby("sigun")["visitors"].sum()
    fss = fs_tot[fs_tot["ym"].isin(months)].groupby("sigun")["spend_k"].sum()
    fmix = category_mix(dl, "foreign", period)
    f["foreign_visitors"] = fvv
    f["foreign_spend_per_visitor"] = fss * 1000 / fvv
    f["foreign_share_pct"] = fvv / fvv.sum() * 100
    f["outsider_share_of_gangwon_pct"] = f["outsider_visitors"] / f["outsider_visitors"].sum() * 100
    f["foreign_medical_k"] = fmix.get("의료관광", 0)
    f.attrs["foreign_months"] = (min(months), max(months), len(months))

    # 자원 (공공데이터포털 API)
    tour, wel = api["tour"], api["wellness"]
    f["wellness_sites"] = wel.groupby("sigun").size()
    f["forest_sites"] = tour[tour["lclsSystm2"] == config.FOREST_CODE].groupby("sigun").size()
    f["nature_sites"] = tour[tour["lclsSystm1"] == "NA"].groupby("sigun").size()
    f["stay_listed"] = tour[tour["contenttypeid"] == config.STAY_CONTENT_TYPE].groupby("sigun").size()
    f[["wellness_sites", "forest_sites", "nature_sites", "stay_listed"]] = (
        f[["wellness_sites", "forest_sites", "nature_sites", "stay_listed"]].fillna(0).astype(int))
    f["wellness_resources"] = f["wellness_sites"] + f["forest_sites"]
    res_share = f["wellness_resources"] / f["wellness_resources"].sum()
    spend_share = f["spend_domestic_k"] / f["spend_domestic_k"].sum()
    f["resource_share_pct"] = res_share * 100
    f["spend_share_pct"] = spend_share * 100
    f["resource_spend_ratio"] = spend_share / res_share

    f["wellness_cluster"] = f.index.isin(config.WELLNESS_CLUSTER)
    f.index.name = "sigun"
    return f


def gangwon_summary(f: pd.DataFrame) -> dict:
    """강원 전체 값 (지출 가중)."""
    lodging_k = (f["spend_domestic_k"] * f["lodging_share_pct"] / 100).sum()
    return {
        "외지인 1명당 지출(원)": f["spend_domestic_k"].sum() * 1000 / f["outsider_visitors"].sum(),
        "숙박 비중(%)": lodging_k / f["spend_domestic_k"].sum() * 100,
        "외국인 1명당 지출(원)": (f["foreign_spend_per_visitor"] * f["foreign_visitors"]).sum() / f["foreign_visitors"].sum(),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표 사이 직선거리(km)."""
    r = np.pi / 180
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def wellness_sites_with_coords(api: dict) -> pd.DataFrame:
    """웰니스 자원(웰니스 관광지 + 휴양림·수목원)의 좌표."""
    w, t = api["wellness"], api["tour"]
    a = pd.DataFrame({"title": w["title"], "sigun": w["sigun"], "kind": "웰니스 관광지",
                      "x": pd.to_numeric(w["mapX"], errors="coerce"), "y": pd.to_numeric(w["mapY"], errors="coerce")})
    f = t[t["lclsSystm2"] == config.FOREST_CODE]
    b = pd.DataFrame({"title": f["title"], "sigun": f["sigun"], "kind": "휴양림·수목원",
                      "x": pd.to_numeric(f["mapx"], errors="coerce"), "y": pd.to_numeric(f["mapy"], errors="coerce")})
    return pd.concat([a, b], ignore_index=True).dropna(subset=["x", "y"])


def medical_features(api: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """가설 E: 시군별 의료 인프라와 웰니스 자원에서 가장 가까운 종합병원까지 거리.
    반환: (시군 지표, 자원별 최근접 종합병원)"""
    h = api["hospitals"].copy()
    h["doctors"] = pd.to_numeric(h["drTotCnt"], errors="coerce").fillna(0)
    h["x"] = pd.to_numeric(h["XPos"], errors="coerce")
    h["y"] = pd.to_numeric(h["YPos"], errors="coerce")
    gen = h[h["clCdNm"].isin(config.GENERAL_HOSPITAL)].dropna(subset=["x", "y"])

    sites = wellness_sites_with_coords(api)
    dist = haversine_km(sites["y"].values[:, None], sites["x"].values[:, None],
                        gen["y"].values[None, :], gen["x"].values[None, :])
    sites["nearest_general_km"] = dist.min(axis=1)
    sites["nearest_general"] = gen["yadmNm"].values[dist.argmin(axis=1)]

    g = h.groupby("sigun")
    m = pd.DataFrame({
        "medical_facilities": g.size(),
        "doctors": g["doctors"].sum(),
        "general_hospitals": gen.groupby("sigun").size(),
        "korean_medicine": h[h["clCdNm"].isin(config.KOREAN_MEDICINE)].groupby("sigun").size(),
        "resource_to_general_km": sites.groupby("sigun")["nearest_general_km"].median(),
    })
    m[["medical_facilities", "doctors", "general_hospitals", "korean_medicine"]] = (
        m[["medical_facilities", "doctors", "general_hospitals", "korean_medicine"]].fillna(0).astype(int))
    m.index.name = "sigun"
    return m, sites
