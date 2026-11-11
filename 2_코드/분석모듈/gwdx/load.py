"""원자료 불러오기.

한국관광 데이터랩에서 내려받은 파일(zip 또는 csv)을 그대로 넣으면
머리글(컬럼 이름)을 보고 종류를 알아내 정돈된(tidy) 표로 바꾼다.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from . import config

# 머리글 -> (데이터 종류, 표준 컬럼 이름)
SCHEMAS = {
    ("기준년월", "지자체", "방문자 구분", "방문자수"): (
        "visitors_domestic", ["ym", "region", "visitor_type", "visitors"]),
    ("날짜", "지역", "외국인 방문자수"): (
        "visitors_foreign", ["ym", "sigun", "visitors"]),
    ("기준년월", "지자체", "중분류", "소비금액"): (
        "spending_domestic", ["ym", "region", "category", "spend_k"]),
    ("기준년월", "지자체", "KTO업종중분류명", "소비액(천원)"): (
        "spending_foreign", ["ym", "region", "category", "spend_k"]),
}

VISITOR_TYPE = {"전체방문자(a+b)": "total", "현지인방문자(a)": "local", "외지인방문자(b)": "outsider"}


def _read_any(path: Path) -> list[pd.DataFrame]:
    """zip 안의 csv 또는 csv 파일 하나를 읽는다."""
    if path.suffix.lower() == ".zip":
        out = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    out.append(pd.read_csv(io.BytesIO(zf.read(name)), encoding="utf-8-sig"))
        return out
    return [pd.read_csv(path, encoding="utf-8-sig")]


def load_datalab(folder: Path = config.RAW_DATALAB) -> dict[str, pd.DataFrame]:
    """데이터랩 폴더의 모든 파일을 종류별로 합쳐 돌려준다."""
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k, _ in SCHEMAS.values()}
    for path in sorted(folder.rglob("*")):  # 종류별 하위 폴더까지
        if path.suffix.lower() not in (".zip", ".csv"):
            continue
        for df in _read_any(path):
            key = tuple(df.columns)
            if key not in SCHEMAS:
                raise ValueError(f"알 수 없는 형식: {path.name} {list(df.columns)}")
            kind, cols = SCHEMAS[key]
            df.columns = cols
            df["source_file"] = path.name
            parts[kind].append(df)

    out = {}
    for kind, frames in parts.items():
        if not frames:
            raise FileNotFoundError(f"{kind} 파일이 없습니다: {folder}")
        df = pd.concat(frames, ignore_index=True)
        df["ym"] = df["ym"].astype(int)
        if "region" in df:
            # '강원특별자치도 춘천시' -> 시도 / 시군
            split = df["region"].str.split(" ", n=1, expand=True)
            df["sido"], df["sigun"] = split[0], split[1]
        if kind == "visitors_domestic":
            df["visitor_type"] = df["visitor_type"].map(VISITOR_TYPE)
        out[kind] = df.drop_duplicates(subset=[c for c in df.columns if c != "source_file"])
    return out


def load_api(folder: Path = config.RAW_API) -> dict[str, pd.DataFrame]:
    """공공데이터포털 API로 받은 CSV(수집 스크립트 결과)를 읽는다."""
    read = lambda name: pd.read_csv(folder / name, dtype=str, keep_default_na=False)  # 'NA'(자연) 코드 보존
    ldong = read("gangwon_ldong.csv")
    names = dict(zip(ldong["code"], ldong["name"]))
    out = {"ldong": ldong}
    for key, name in [("wellness", "gangwon_wellness.csv"), ("stay", "gangwon_stay.csv"), ("tour", "gangwon_tour_all.csv")]:
        df = read(name)
        df["sigun"] = df["lDongSignguCd"].map(names)
        out[key] = df
    hosp = folder / "gangwon_hospitals.csv"
    if hosp.exists():  # 가설 E (의료 연계) - 심평원 병원기본목록
        h = read("gangwon_hospitals.csv")
        h["sigun"] = h["sgguCdNm"]
        out["hospitals"] = h
    return out


def quality_report(dl: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """데이터 종류별 기간, 시군 수, 시군별 달 수, 중복 여부를 한 표로 보여준다."""
    rows = []
    for kind, df in dl.items():
        months = df.groupby("sigun")["ym"].nunique()
        rows.append({
            "데이터": kind,
            "행 수": len(df),
            "시작": df["ym"].min(),
            "끝": df["ym"].max(),
            "시군 수": df["sigun"].nunique(),
            "시군별 달 수(최소~최대)": f"{months.min()}~{months.max()}",
            "시도": ",".join(sorted(df["sido"].dropna().unique())) if "sido" in df else "(지역명만 제공)",
        })
    return pd.DataFrame(rows)
