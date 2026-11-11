"""원자료 -> 지표 -> 진단 -> 결과 파일.

실행:  python 2_코드/실행스크립트/run_pipeline.py
결과:  1_데이터/가공/ 아래 CSV
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "분석모듈"))

from scipy import stats  # noqa: E402

from gwdx import config, diagnose, features, load  # noqa: E402

# 대시보드·README에 싣는 관계 (Spearman)
RELATIONS = [
    ("res_vis", "wellness_resources", "outsider_visitors", "웰니스 자원 수 ↔ 외지인 방문"),
    ("res_spend", "wellness_resources", "spend_domestic_k", "웰니스 자원 수 ↔ 관광 지출"),
    ("stay_spend", "stay_listed", "spend_domestic_k", "등록 숙박 수 ↔ 관광 지출"),
    ("out_season", "outsider_share_pct", "seasonality_ratio", "외지인 비율 ↔ 계절 편차"),
    ("med_doctors", "doctors", "foreign_medical_k", "의사 수 ↔ 외국인 의료관광 지출"),
    ("med_distance", "resource_to_general_km", "foreign_medical_k", "자원→종합병원 거리 ↔ 외국인 의료관광 지출"),
]


def relations(f):
    rows = []
    for key, a, b, label in RELATIONS:
        r = stats.spearmanr(f[a], f[b])
        rows.append({"key": key, "관계": label, "rho": round(float(r.statistic), 2), "p값": round(float(r.pvalue), 4),
                     "유의(p<0.05)": "Y" if r.pvalue < 0.05 else "N", "시군수": len(f)})
    return pd.DataFrame(rows)

# 대시보드에 쓰는 한글 컬럼 이름
DASH_COLUMNS = {
    "outsider_visitors": ("외지인_만명", lambda s: (s / 1e4).round(0)),
    "spend_domestic_k": ("내국인지출_억원", lambda s: (s / 1e5).round(0)),
    "spend_per_outsider": ("외지인1명당지출_원", lambda s: s.round(0)),
    "lodging_share_pct": ("숙박비중_pct", lambda s: s.round(1)),
    "seasonality_ratio": ("지출계절편차_배", lambda s: s.round(2)),
    "resource_spend_ratio": ("자원대비지출_배", lambda s: s.round(2)),
    "wellness_resources": ("웰니스자원_곳", lambda s: s),
    "foreign_visitors": ("외국인_만명", lambda s: (s / 1e4).round(1)),
    "foreign_spend_per_visitor": ("외국인1명당지출_원", lambda s: s.round(0)),
    "foreign_medical_k": ("외국인의료관광_억원", lambda s: (s / 1e5).round(2)),
    "foreign_share_pct": ("외국인점유_pct", lambda s: s.round(1)),
    "outsider_share_of_gangwon_pct": ("외지인점유_pct", lambda s: s.round(1)),
    "general_hospitals": ("종합병원_곳", lambda s: s),
    "doctors": ("의사_명", lambda s: s),
    "korean_medicine": ("한방기관_곳", lambda s: s),
    "resource_to_general_km": ("자원종합병원거리_km", lambda s: s.round(1)),
}
FLAG_COLUMNS = {"A_stay_weak": "A_체류약함", "B_seasonal": "B_계절의존",
                "C_low_spend": "C_저객단가", "D_underused": "D_자원미활용"}


def to_dashboard(d, sens):
    out = d[[]].copy()
    for col, (name, fn) in DASH_COLUMNS.items():
        out[name] = fn(d[col])
    for col, name in FLAG_COLUMNS.items():
        out[name] = d[col].map({True: "Y", False: "N"})
    out["진단유형"] = d["type"]
    out["웰니스클러스터"] = d["wellness_cluster"].map({True: "Y", False: "N"})
    out["처방"] = d["prescription"]
    out["유형순서"] = d["type_order"]
    out["처방묶음"] = d["group"]
    out["묶음순서"] = d["group_order"]
    out["묶음처방"] = d["group_prescription"]
    out["유형유지_pct"] = sens["type_kept_pct"].reindex(d.index).round(0)
    out.index.name = "시군"
    return out.reset_index()


def main():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    dl = load.load_datalab()
    api = load.load_api()
    print(load.quality_report(dl).to_string(index=False))

    f = features.build_features(dl, api)
    med, sites = features.medical_features(api)
    attrs = dict(f.attrs)
    f = f.join(med)
    f.attrs.update(attrs)
    d = diagnose.diagnose(f)
    s = diagnose.sensitivity(f)
    m = features.monthly_spending(dl)

    f.to_csv(config.PROCESSED / "features.csv", encoding="utf-8-sig")
    sites.to_csv(config.PROCESSED / "wellness_sites_nearest_hospital.csv", index=False, encoding="utf-8-sig")
    d.to_csv(config.PROCESSED / "diagnosis.csv", encoding="utf-8-sig")
    s.to_csv(config.PROCESSED / "sensitivity.csv", encoding="utf-8-sig")
    margins = diagnose.threshold_margins(f)
    margins.to_csv(config.PROCESSED / "threshold_margins.csv", index=False, encoding="utf-8-sig")
    print("\n기준값과 가장 가까운 시군까지 간격:\n" + margins.round(3).to_string(index=False))

    # 대시보드용
    to_dashboard(d, s).to_csv(config.PROCESSED / "dashboard_diagnosis.csv", index=False, encoding="utf-8")
    md = m.assign(내국인지출_억원=(m["spend_k"] / 1e5).round(1), 월평균대비=m["ratio_to_mean"].round(3))
    md.rename(columns={"sigun": "시군", "ym": "기준년월"})[["시군", "기준년월", "내국인지출_억원", "월평균대비"]] \
        .to_csv(config.PROCESSED / "dashboard_monthly_spend.csv", index=False, encoding="utf-8")

    relations(f).to_csv(config.PROCESSED / "dashboard_relations.csv", index=False, encoding="utf-8")

    print("\n강원 요약:", {k: round(v, 1) for k, v in features.gangwon_summary(f).items()})
    print("외국인 사용 기간:", f.attrs["foreign_months"])
    print("\n유형별 시군:")
    for t, g in d.groupby("type", sort=False):
        print(f"  {t}: {', '.join(g.index)}")
    print(f"\n저장 위치: {config.PROCESSED}")


if __name__ == "__main__":
    main()
