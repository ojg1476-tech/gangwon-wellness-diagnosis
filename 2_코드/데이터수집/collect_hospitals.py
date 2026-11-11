"""
강원 의료기관 목록 수집 (가설 E: 의료 연계)

데이터 출처: 공공데이터포털, 건강보험심사평가원 병원정보서비스 / 병원기본목록(getHospBasisList)
출력 파일: gangwon_hospitals.csv  (강원 전체 의료기관: 이름, 종별, 시군구, 주소, 의사 수, 좌표 등)

인증키
  환경변수 DATA_GO_KR_KEY 에 일반 인증키(Decoding)를 넣거나 실행 시 입력한다.
  키는 코드와 출력 파일에 저장되지 않는다.

응답이 JSON이든 XML이든 모두 읽는다.
"""

import csv
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from getpass import getpass

# 서비스 버전에 따라 주소가 다를 수 있어 차례로 시도한다
BASES = [
    "https://apis.data.go.kr/B551182/hospInfoServicev2",
    "http://apis.data.go.kr/B551182/hospInfoServicev2",
    "https://apis.data.go.kr/B551182/hospInfoService1",
]
OP = "getHospBasisList"
GANGWON_SIDO = "320000"  # 심평원 시도코드: 강원
ROWS = 1000


def _parse(text):
    """JSON 또는 XML 응답 -> (결과코드, 메시지, item 리스트, 전체 건수)"""
    try:
        data = json.loads(text)
        head = data.get("response", {}).get("header") or data
        body = data.get("response", {}).get("body", {})
        items = body.get("items") or {}
        items = items.get("item", []) if isinstance(items, dict) else []
        if isinstance(items, dict):
            items = [items]
        return str(head.get("resultCode")), head.get("resultMsg"), items, int(body.get("totalCount") or 0)
    except json.JSONDecodeError:
        root = ET.fromstring(text)
        code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode")
        msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg")
        items = [{c.tag: (c.text or "") for c in it} for it in root.iter("item")]
        return str(code), msg, items, int(root.findtext(".//totalCount") or 0)


def call(base, key, **params):
    q = {"serviceKey": key, "_type": "json", **params}
    url = f"{base}/{OP}?{urllib.parse.urlencode(q)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    code, msg, items, total = _parse(text)
    if code not in ("00", "0000", "None"):
        raise RuntimeError(f"{OP} 응답 오류 [{code}] {msg}\n{text[:500]}")
    return items, total


def find_base(key):
    """작동하는 서비스 주소를 찾는다."""
    errors = []
    for base in BASES:
        try:
            items, total = call(base, key, pageNo=1, numOfRows=1, sidoCd=GANGWON_SIDO)
            if total:
                return base
            errors.append(f"{base}: 0건")
        except Exception as e:  # 다음 주소로
            errors.append(f"{base}: {e}")
    raise RuntimeError("작동하는 주소를 찾지 못함\n" + "\n".join(errors))


def main():
    key = os.environ.get("DATA_GO_KR_KEY") or getpass("인증키(Decoding): ").strip()
    base = find_base(key)
    print("서비스 주소:", base)

    rows, page = [], 1
    while True:
        items, total = call(base, key, pageNo=page, numOfRows=ROWS, sidoCd=GANGWON_SIDO)
        rows += items
        print(f"  p.{page}: {len(rows)}/{total}")
        if len(rows) >= total or not items:
            break
        page += 1
        time.sleep(0.3)

    sido_names = {r.get("sidoCdNm", "") for r in rows}
    print("시도 이름:", sido_names)
    if rows and not any("강원" in s for s in sido_names):
        raise RuntimeError("강원 자료가 아닙니다. 시도코드를 확인해야 합니다.")

    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open("gangwon_hospitals.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"저장: gangwon_hospitals.csv ({len(rows)}건)")


if __name__ == "__main__":
    main()
