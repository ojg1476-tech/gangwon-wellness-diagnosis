"""데이터 연결 지도: 데이터 -> 연결 -> 지표 -> 가설 -> 결론 을 한 장의 그림으로 그린다.

실행: python 2_코드/실행스크립트/draw_data_map.py  ->  4_결과물/데이터연결지도/data_map.svg
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[2] / "4_결과물" / "데이터연결지도" / "data_map.svg"

FONT = "'Noto Sans CJK KR','Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif"

COLUMNS = [
    ("① 데이터", "어디서 가져왔나", "#2D6A8E", "#EAF2F8"),
    ("② 연결", "어떻게 붙였나", "#6B7280", "#F3F4F6"),
    ("③ 지표", "무엇을 쟀나", "#6B7280", "#FFFFFF"),
    ("④ 가설", "무엇을 확인했나", "#B7791F", "#FFF6E5"),
    ("⑤ 결론", "그래서 무엇을 말하나", "#C8553D", "#FCEDE9"),
]

# (id, 열, 제목, 설명 줄들)
NODES = [
    ("vis", 0, "방문자", ["한국관광 데이터랩 · 이동통신", "내국인·외국인, 월·시군"]),
    ("spend", 0, "관광 지출", ["한국관광 데이터랩 · 신용카드", "업종 22개, 월·시군"]),
    ("res", 0, "웰니스 자원", ["관광공사 API", "웰니스 15곳 + 휴양림 94곳, 좌표"]),
    ("stay", 0, "숙박", ["관광공사 API", "등록 숙박 531곳"]),
    ("hosp", 0, "의료기관", ["심평원 API", "1,992곳 (종합병원 16), 좌표"]),

    ("k_sigun", 1, "시·군으로 합치기", ["18개 시·군 이름이 공통 열쇠"]),
    ("k_month", 1, "기간 맞추기", ["2025.11–2026.10, 월별", "외국인은 겹치는 11개월"]),
    ("k_geo", 1, "좌표로 거리 재기", ["자원마다 가장 가까운", "종합병원까지 km"]),

    ("m_lodg", 2, "숙박 비중", ["숙박 지출 ÷ 관광 지출"]),
    ("m_sea", 2, "계절 편차", ["가장 많은 달 ÷ 가장 적은 달"]),
    ("m_spc", 2, "1명당 지출", ["관광 지출 ÷ 외지인 방문자"]),
    ("m_res", 2, "자원 대비 지출", ["지출 몫 ÷ 웰니스 자원 몫"]),
    ("m_med", 2, "의료 접근성", ["자원 → 종합병원 거리, 의사 수"]),

    ("h_a", 3, "A 체류", ["자고 가나?  숙박 5% 미만이면 약함"]),
    ("h_b", 3, "B 비수기", ["계절을 타나?  2.5배 이상이면 의존"]),
    ("h_c", 3, "C 객단가", ["얼마 쓰나?  강원 평균의 3/4 미만이면 낮음"]),
    ("h_d", 3, "D 미스매치", ["자원만큼 버나?  0.5배 미만이면 미활용"]),
    ("h_e", 3, "E 의료 연계", ["병원이 가까운가?"]),

    ("c1", 4, "자원은 사람을 부르지만, 돈은 숙박을 따라간다",
     ["자원↔방문 ρ 0.52 (뚜렷)", "자원↔지출 ρ 0.39 (뚜렷하지 않음)", "숙박↔지출 ρ 0.67 (뚜렷)"]),
    ("c2", 4, "시·군마다 막힌 곳이 달라 처방도 다르다",
     ["자원이 돈이 되지 못하는 곳 6 · 계절에 쏠리는 곳 3", "당일치기 도시 3 · 체류·소비 거점 5", "관찰 대상 1 (정선)"]),
    ("c3", 4, "의료웰니스 거점은 원주뿐",
     ["클러스터 중 양양·정선·평창은 종합병원 0곳", "의사 수↔외국인 의료관광 ρ 0.85"]),
]

EDGES = [
    ("vis", "k_sigun"), ("spend", "k_sigun"), ("res", "k_sigun"), ("stay", "k_sigun"), ("hosp", "k_sigun"),
    ("vis", "k_month"), ("spend", "k_month"),
    ("res", "k_geo"), ("hosp", "k_geo"),
    ("k_sigun", "m_lodg"), ("k_month", "m_sea"), ("k_sigun", "m_spc"), ("k_sigun", "m_res"), ("k_geo", "m_med"),
    ("m_lodg", "h_a"), ("m_sea", "h_b"), ("m_spc", "h_c"), ("m_res", "h_d"), ("m_med", "h_e"),
    ("h_d", "c1"), ("h_a", "c1"),
    ("h_a", "c2"), ("h_b", "c2"), ("h_c", "c2"), ("h_d", "c2"),
    ("h_e", "c3"),
]
# 핵심 메시지로 가는 길 (굵게)
MAIN = {("res", "k_sigun"), ("stay", "k_sigun"), ("spend", "k_sigun"), ("k_sigun", "m_res"), ("k_sigun", "m_lodg"),
        ("m_res", "h_d"), ("m_lodg", "h_a"), ("h_d", "c1"), ("h_a", "c1")}

COL_W = [250, 230, 240, 270, 330]
GAP_X = 70
TOP = 120
ROW_GAP = 16
LINE_H = 19
PAD = 14


def node_height(lines, title_lines=1):
    return PAD * 2 + 22 * title_lines + LINE_H * len(lines)


def wrap_title(text, width_px):
    # 한글 기준 대략 글자당 16px
    per = max(6, int(width_px / 16))
    if len(text) <= per:
        return [text]
    if ", " in text:  # 쉼표에서 먼저 나눈다
        a, b = text.split(", ", 1)
        return [a + ",", b]
    cut = text.rfind(" ", 0, per)
    cut = cut if cut > 0 else per
    return [text[:cut].strip(), text[cut:].strip()]


def main():
    xs, x = [], 40
    for w in COL_W:
        xs.append(x)
        x += w + GAP_X
    width = x - GAP_X + 40

    # 열마다 세로로 쌓고 가운데 정렬
    pos = {}
    heights = {}
    col_nodes = {c: [n for n in NODES if n[1] == c] for c in range(len(COLUMNS))}
    col_total = {}
    for c, nodes in col_nodes.items():
        total = 0
        for nid, _, title, lines in nodes:
            tl = wrap_title(title, COL_W[c] - 2 * PAD)
            h = node_height(lines, len(tl))
            heights[nid] = (h, tl)
            total += h
        total += ROW_GAP * (len(nodes) - 1) if c != 4 else 34 * (len(nodes) - 1)
        col_total[c] = total
    max_total = max(col_total.values())
    height = TOP + max_total + 50
    for c, nodes in col_nodes.items():
        y = TOP + (max_total - col_total[c]) / 2
        for nid, *_ in nodes:
            h = heights[nid][0]
            pos[nid] = (xs[c], y, COL_W[c], h, c)
            y += h + (ROW_GAP if c != 4 else 34)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" height="{height:.0f}" font-family="{FONT}">',
           f'<rect width="100%" height="100%" fill="#FFFFFF"/>',
           f'<text x="40" y="48" font-size="26" font-weight="700" fill="#1F2933">강원 웰니스관광 진단 · 데이터는 이렇게 연결된다</text>',
           f'<text x="40" y="76" font-size="15" fill="#6B7280">다섯 가지 공공데이터를 시·군·기간·좌표로 엮어 지표를 만들고, 가설 다섯 개를 확인해 결론 세 개를 얻었다. 굵은 선은 핵심 메시지로 가는 길.</text>']

    # 열 머리
    for c, (name, sub, color, _) in enumerate(COLUMNS):
        out.append(f'<text x="{xs[c]}" y="{TOP - 22}" font-size="16" font-weight="700" fill="{color}">{escape(name)}'
                   f'<tspan font-weight="400" fill="#9AA3AE" font-size="13">  {escape(sub)}</tspan></text>')

    # 선 (노드 뒤)
    for a, b in EDGES:
        ax, ay, aw, ah, ac = pos[a]
        bx, by, bw, bh, bc = pos[b]
        x1, y1 = ax + aw, ay + ah / 2
        x2, y2 = bx, by + bh / 2
        mx = (x1 + x2) / 2
        main_ = (a, b) in MAIN
        color = "#C8553D" if main_ else "#C3C9D1"
        out.append(f'<path d="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2 - 6:.1f},{y2:.1f}" fill="none" '
                   f'stroke="{color}" stroke-width="{2.6 if main_ else 1.4}" stroke-opacity="{0.9 if main_ else 1}"/>')
        out.append(f'<path d="M{x2 - 7:.1f},{y2 - 4:.1f} L{x2:.1f},{y2:.1f} L{x2 - 7:.1f},{y2 + 4:.1f}" fill="none" stroke="{color}" stroke-width="{2 if main_ else 1.4}"/>')

    # 노드
    for nid, c, title, lines in NODES:
        x, y, w, h, _ = pos[nid]
        _, _, stroke, fill = COLUMNS[c]
        sw = 2.2 if c == 4 else 1.4
        rx = 22 if c == 1 else 10
        out.append(f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        tl = heights[nid][1]
        ty = y + PAD + 17
        for t in tl:
            out.append(f'<text x="{x + PAD}" y="{ty:.1f}" font-size="{17 if c == 4 else 16}" font-weight="700" fill="#1F2933">{escape(t)}</text>')
            ty += 22
        ty += 2
        for ln in lines:
            out.append(f'<text x="{x + PAD}" y="{ty:.1f}" font-size="13.5" fill="#4B5563">{escape(ln)}</text>')
            ty += LINE_H

    out.append(f'<text x="40" y="{height - 18:.0f}" font-size="12.5" fill="#9AA3AE">ρ: 18개 시·군 순위 상관(Spearman). 함께 움직이는 정도이며 원인을 증명하지 않는다. '
               f'출처: 한국관광 데이터랩, 한국관광공사·건강보험심사평가원 공공데이터 API.</text>')
    out.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
