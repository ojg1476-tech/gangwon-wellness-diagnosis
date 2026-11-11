"""
강원 웰니스관광 수용태세 진단 - 공공데이터 수집 스크립트

데이터 출처 (공공데이터포털, 한국관광공사)
  - 웰니스관광정보 API       : B551011/WellnessTursmService
  - 국문 관광정보 서비스 API : B551011/KorService2

출력 파일
  gangwon_ldong.csv     강원 시군구 법정동 코드표 (강원 시도코드 = 51)
  gangwon_wellness.csv  강원 웰니스 관광지 목록
  gangwon_stay.csv      강원 숙박시설 목록 (contentTypeId=32)
  gangwon_tour_all.csv  강원 관광정보 전체 목록 (전 유형)

인증키
  환경변수 DATA_GO_KR_KEY 에 일반 인증키(Decoding)를 넣거나,
  실행 시 입력창에 입력한다. 키는 코드와 출력 파일에 저장되지 않는다.
"""

import csv
import json
import os
import time
import urllib.parse
import urllib.request
from getpass import getpass

WELLNESS = "https://apis.data.go.kr/B551011/WellnessTursmService"
KOR = "https://apis.data.go.kr/B551011/KorService2"
ROWS = 1000  # 페이지당 최대 건수


def call(base, op, key, **params):
    """API 1회 호출 -> (item 리스트, 전체 건수)"""
    q = {"serviceKey": key, "MobileOS": "ETC", "MobileApp": "GangwonWellness", "_type": "json"}
    q.update({k: v for k, v in params.items() if v is not None})
    url = f"{base}/{op}?{urllib.parse.urlencode(q)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 인증 오류 등은 XML 형식으로 반환됨
        raise RuntimeError(f"{op} 응답 오류:\n{text[:800]}")
    # 오류 응답은 {"resultCode": ..., "resultMsg": ...} 형태로 바로 오기도 함
    head = data.get("response", {}).get("header") or data
    if head.get("resultCode") not in ("0000", "00", None):
        raise RuntimeError(f"{op} 응답 오류: {head}")
    body = data.get("response", {}).get("body", {})
    items = body.get("items") or {}
    items = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(items, dict):
        items = [items]
    return items, int(body.get("totalCount") or 0)


def fetch_all(base, op, key, **params):
    """전체 페이지 수집"""
    out, page = [], 1
    while True:
        items, total = call(base, op, key, numOfRows=ROWS, pageNo=page, **params)
        out += items
        print(f"  {op} p.{page}: {len(out)}/{total}")
        if len(out) >= total or not items:
            return out
        page += 1
        time.sleep(0.3)


def save(rows, path):
    """dict 리스트를 CSV로 저장 (모든 필드 보존)"""
    if not rows:
        print(f"  {path}: 수집 결과 없음")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  저장: {path} ({len(rows)}건)")


def main():
    key = os.environ.get("DATA_GO_KR_KEY") or getpass("인증키(Decoding): ").strip()

    print("[1/4] 강원 법정동 코드")
    sido, _ = call(KOR, "ldongCode2", key, numOfRows=100, pageNo=1)
    gw = [s for s in sido if "강원" in json.dumps(s, ensure_ascii=False)]
    if not gw:
        raise RuntimeError(f"시도 목록에서 강원을 찾지 못함: {sido[:3]}")
    gw_code = gw[0].get("code") or gw[0].get("lDongRegnCd")
    print(f"  강원 시도코드 = {gw_code}")
    save(fetch_all(KOR, "ldongCode2", key, lDongRegnCd=gw_code), "gangwon_ldong.csv")

    print("[2/4] 웰니스 관광지")
    # 웰니스 API는 언어구분(langDivCd)이 필수
    save(fetch_all(WELLNESS, "areaBasedList", key, langDivCd="KOR", lDongRegnCd=gw_code),
         "gangwon_wellness.csv")

    print("[3/4] 숙박시설")
    save(fetch_all(KOR, "areaBasedList2", key, lDongRegnCd=gw_code, contentTypeId=32), "gangwon_stay.csv")

    print("[4/4] 관광정보 전체")
    save(fetch_all(KOR, "areaBasedList2", key, lDongRegnCd=gw_code), "gangwon_tour_all.csv")

    print("완료")


if __name__ == "__main__":
    main()
