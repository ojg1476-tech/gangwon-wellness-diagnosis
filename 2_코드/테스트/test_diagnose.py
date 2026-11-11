import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "분석모듈"))

from gwdx import config, diagnose, features, load  # noqa: E402
from gwdx.config import Thresholds  # noqa: E402


GANGWON = 20000.0  # 테스트용 강원 평균 1명당 지출(원)


def make_features(**overrides):
    base = dict(lodging_share_pct=8.0, seasonality_ratio=1.5, spend_per_outsider=21000.0,
                resource_spend_ratio=1.2, outsider_visitors=1e6)
    base.update(overrides)
    return pd.DataFrame([base], index=["테스트군"])


def run(**overrides):
    return diagnose.diagnose(make_features(**overrides), gangwon_spend=GANGWON)


@pytest.mark.parametrize("overrides, expected", [
    (dict(resource_spend_ratio=0.3, spend_per_outsider=9000.0), "자원 미활용·통과형"),
    (dict(seasonality_ratio=3.0, lodging_share_pct=20.0), "계절 의존 리조트형"),
    (dict(seasonality_ratio=3.0), "성수기 집중형"),
    (dict(resource_spend_ratio=0.4), "자원 대비 저소비형"),
    (dict(lodging_share_pct=2.0), "당일치기형"),
    (dict(), "체류·소비 거점형"),
    (dict(spend_per_outsider=17000.0), "관찰 대상"),      # 문제는 없지만 강원 평균 이하 -> 거점 아님
    (dict(resource_spend_ratio=0.8), "관찰 대상"),        # 자원 몫만큼 못 벌면 거점 아님
])
def test_classify_rules(overrides, expected):
    assert run(**overrides)["type"].iloc[0] == expected


def test_threshold_is_exact_not_rounded():
    # 2.45배는 2.5배 기준에 들지 않아야 한다 (반올림 후 비교하면 생기는 오류 방지)
    assert not run(seasonality_ratio=2.45)["B_seasonal"].iloc[0]


def test_spend_cutoff_is_relative_to_gangwon():
    # C 기준 = 강원 평균의 4분의 3
    assert run(spend_per_outsider=14999.0)["C_low_spend"].iloc[0]
    assert not run(spend_per_outsider=15001.0)["C_low_spend"].iloc[0]


def test_custom_thresholds():
    t = Thresholds(spend_vs_gangwon=1.2, resource_ratio=1.5)
    d = diagnose.diagnose(make_features(), t, gangwon_spend=GANGWON)
    assert d["type"].iloc[0] == "자원 미활용·통과형"


@pytest.fixture(scope="module")
def real():
    if not any(config.RAW_DATALAB.rglob("*.zip")):
        pytest.skip("원자료 없음")
    dl, api = load.load_datalab(), load.load_api()
    return dl, api, features.build_features(dl, api)


def test_raw_data_complete(real):
    dl, _, _ = real
    q = load.quality_report(dl)
    assert (q["시군 수"] == 18).all()
    assert (q["시군별 달 수(최소~최대)"] == "12~12").all()


def test_shares_sum_to_100(real):
    *_, f = real
    for col in ["resource_share_pct", "spend_share_pct", "foreign_share_pct"]:
        assert f[col].sum() == pytest.approx(100)


def test_category_total_matches_parts(real):
    dl, _, _ = real
    mix = features.category_mix(dl)
    parts = mix.drop(columns="관광총소비").sum(axis=1)
    assert (parts / mix["관광총소비"]).round(6).eq(1).all()


def test_medical_features(real):
    _, api, _ = real
    if "hospitals" not in api:
        pytest.skip("의료기관 자료 없음")
    med, sites = features.medical_features(api)
    assert len(med) == 18
    assert (sites["nearest_general_km"] >= 0).all()
    # 종합병원이 있는 시군은 자원에서 가장 가까운 종합병원이 대체로 가깝다
    assert med.loc[med["general_hospitals"] > 0, "resource_to_general_km"].median() < \
        med.loc[med["general_hospitals"] == 0, "resource_to_general_km"].median()


def test_thresholds_sit_in_gaps(real):
    # 모든 기준값이 가장 가까운 시군과 3% 이상 떨어진 빈 구간에 있어야 한다
    *_, f = real
    m = diagnose.threshold_margins(f)
    assert (m["가까운 쪽 간격(%)"] >= 3).all(), m.to_string()
