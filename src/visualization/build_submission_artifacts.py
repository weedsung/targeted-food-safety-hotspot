from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "output" / "pdf"
PLOT_DIR = BASE_DIR / "output" / "eda_plots"
MODEL_DIR = BASE_DIR / "output" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PDF = OUTPUT_DIR / "poster_explanation_report_2page.pdf"
POSTER_PDF = OUTPUT_DIR / "food_safety_a1_poster.pdf"


def register_fonts():
    regular = Path(r"C:\Windows\Fonts\malgun.ttf")
    bold = Path(r"C:\Windows\Fonts\malgunbd.ttf")
    if regular.exists():
        pdfmetrics.registerFont(TTFont("Malgun", str(regular)))
    if bold.exists():
        pdfmetrics.registerFont(TTFont("Malgun-Bold", str(bold)))
    return (
        "Malgun" if regular.exists() else "Helvetica",
        "Malgun-Bold" if bold.exists() else "Helvetica-Bold",
    )


FONT, FONT_BOLD = register_fonts()


def pstyle(name, size, leading=None, color="#111827", bold=False, align=0):
    return ParagraphStyle(
        name,
        fontName=FONT_BOLD if bold else FONT,
        fontSize=size,
        leading=leading or size * 1.35,
        textColor=colors.HexColor(color),
        alignment=align,
        spaceAfter=0,
    )


def para(c, text, x, y, w, h, size=9, color="#111827", bold=False, leading=None, align=0):
    p = Paragraph(text, pstyle("p", size, leading, color, bold, align))
    _, used_h = p.wrap(w, h)
    p.drawOn(c, x, y + h - used_h)
    return used_h


def section_title(c, title, x, y, w):
    c.setFillColor(colors.HexColor("#0F172A"))
    c.setFont(FONT_BOLD, 11.5)
    c.drawString(x, y, title)
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.line(x, y - 4, x + w, y - 4)


def table(c, data, x, y, col_widths, row_heights=None, font_size=7.4):
    t = Table(data, colWidths=col_widths, rowHeights=row_heights)
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), FONT),
                ("FONT", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    t.wrapOn(c, sum(col_widths), 800)
    t.drawOn(c, x, y)
    return t


def card(c, x, y, w, h, title=None, fill="#FFFFFF"):
    c.setFillColor(colors.HexColor(fill))
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    if title:
        c.setFillColor(colors.HexColor("#0F172A"))
        c.setFont(FONT_BOLD, 13)
        c.drawString(x + 10, y + h - 20, title)


def image(c, path, x, y, w, h):
    if path.exists():
        c.drawImage(str(path), x, y, width=w, height=h, preserveAspectRatio=True, anchor="c")
    else:
        para(c, f"이미지 없음: {path.name}", x, y, w, h, size=9, color="#DC2626")


def load_stats():
    summary = pd.read_csv(MODEL_DIR / "model_training_summary.csv").iloc[0]
    odds = pd.read_csv(MODEL_DIR / "odds_ratios.csv").set_index("Feature")
    priority = pd.read_csv(MODEL_DIR / "inspection_priority_index.csv")
    sens = pd.read_csv(MODEL_DIR / "priority_weight_sensitivity.csv")
    return summary, odds, priority, sens


def build_report_pdf():
    summary, odds, _, sens = load_stats()
    c = canvas.Canvas(str(REPORT_PDF), pagesize=A4)
    width, height = A4
    margin_x = 18 * mm
    margin_top = 16 * mm
    usable_w = width - 2 * margin_x

    def footer(page):
        c.setFont(FONT, 7)
        c.setFillColor(colors.HexColor("#64748B"))
        c.drawRightString(width - margin_x, 10 * mm, f"{page} / 2")

    # Page 1
    y = height - margin_top
    c.setFillColor(colors.HexColor("#0F172A"))
    c.setFont(FONT_BOLD, 16)
    c.drawString(margin_x, y, "물가가 오르면 동네 식당의 위생도 흔들릴까?")
    y -= 15
    para(
        c,
        "경제·기후·상권 데이터를 융합한 식품접객업 행정처분 취약 신호 탐지 및 점검 타겟팅 전략",
        margin_x,
        y - 12,
        usable_w,
        15,
        size=8.8,
        color="#475569",
    )
    y -= 38

    section_title(c, "1. 배경", margin_x, y, usable_w)
    y -= 18
    para(
        c,
        "<b>주제 선정.</b> 최근 식재료비, 인건비, 임대료 등 운영비 부담이 커지면서 음식점은 위생관리 비용과 인력을 동시에 유지해야 하는 압박을 받는다. 본 연구는 물가 상승 등 경제적 압박이 커지는 시기에 식품접객업 행정처분 취약 신호가 함께 높아지는지 탐색하였다.",
        margin_x,
        y - 48,
        usable_w,
        48,
        size=8.5,
    )
    y -= 55
    para(
        c,
        "<b>분석 필요성.</b> 기존 단속은 절대 행정처분 건수가 많은 지역에 집중되기 쉽다. 그러나 음식점 수가 많은 대도시는 구조적으로 건수도 커질 수 있으므로, 식당 1,000개당 행정처분 발생률로 보정하고 경제·기후·상권 조건을 함께 검토할 필요가 있다.",
        margin_x,
        y - 48,
        usable_w,
        48,
        size=8.5,
    )
    y -= 60

    section_title(c, "2. 데이터 분석", margin_x, y, usable_w)
    y -= 88
    table(
        c,
        [
            ["영역", "자료", "기간", "활용 변수"],
            ["위생", "식품안전나라 행정처분", "2024.05-2026.04", "처분일자, 업종, 주소"],
            ["경제", "KOSIS 소비자물가지수", "2024.01-2026.03", "총 CPI, 농축수산 CPI"],
            ["기후", "기상청 ASOS", "2024.01-2026.05", "기온, 습도, 강수량, THI"],
            ["상권", "MDIS 전국사업체조사", "2023", "시도별 음식점업 사업체 수"],
        ],
        margin_x,
        y,
        [38, 125, 78, 170],
        font_size=7.1,
    )
    y -= 14
    para(
        c,
        f"전처리 후 식품접객업 행정처분 2,901건을 확보했고, 모델링 데이터는 {int(summary['Sidos_Used'])}개 시도 x {int(summary['Months_Used'])}개월, 총 {int(summary['Rows_Used'])}개 시도-월 관측치로 구성하였다. 종속변수는 식당 1,000개당 행정처분 발생률 상위 20% 핫스팟 여부이며, 임계값은 {summary['Hotspot_Threshold_Per_1000_Restaurants']:.4f}건이다.",
        margin_x,
        y - 42,
        usable_w,
        42,
        size=8.3,
    )
    y -= 52

    table(
        c,
        [
            ["분석 단계", "처리 내용"],
            ["1", "식품접객업 행정처분 필터링 및 시도명 정제"],
            ["2", "기상·경제·상권 자료를 시도-월 단위로 결합"],
            ["3", "식당 1,000개당 행정처분 발생률과 Top 20% 핫스팟 산출"],
            ["4", "상관분석, 로지스틱 회귀, 민감도 분석, FSIPI 구성"],
        ],
        margin_x,
        y - 78,
        [55, 356],
        font_size=7.2,
    )
    footer(1)
    c.showPage()

    # Page 2
    y = height - margin_top
    section_title(c, "2. 데이터 분석 - 결과 및 해석", margin_x, y, usable_w)
    y -= 118
    table(
        c,
        [
            ["변수", "OR", "p-value", "해석"],
            ["총 CPI", f"{odds.loc['CPI_TOTAL', 'OddsRatio']:.2f}", f"{odds.loc['CPI_TOTAL', 'P_Value']:.3f}", "취약 신호와 양의 방향"],
            ["THI", f"{odds.loc['THI', 'OddsRatio']:.2f}", f"{odds.loc['THI', 'P_Value']:.3f}", "보정 발생률 기준 음의 방향"],
            ["음식점업 사업체 수", f"{odds.loc['RESTAURANT_DENSITY', 'OddsRatio']:.2f}", f"{odds.loc['RESTAURANT_DENSITY', 'P_Value']:.3f}", "직접 효과 거의 없음"],
            ["농축수산 CPI", f"{odds.loc['CPI_AGRI', 'OddsRatio']:.2f}", f"{odds.loc['CPI_AGRI', 'P_Value']:.3f}", "총 CPI와 공선성 주의"],
        ],
        margin_x,
        y,
        [95, 50, 58, 208],
        font_size=7.2,
    )
    y -= 22
    para(
        c,
        "총 소비자물가지수 1표준편차 증가 시 핫스팟 오즈는 약 39% 증가하는 방향을 보였다. 다만 p-value가 5% 유의수준을 충족하지 못하므로 확정적 인과관계가 아니라 경제적 압박과 행정처분 취약성 간 연관 가능성을 시사하는 탐색적 정책 신호로 해석하였다. LightGBM은 AUC 0.537로 고성능 예측모형이라고 주장하지 않고 보조 실험으로만 활용하였다.",
        margin_x,
        y - 54,
        usable_w,
        54,
        size=8.3,
    )
    y -= 70

    section_title(c, "3. 분석 활용 전략", margin_x, y, usable_w)
    y -= 19
    para(
        c,
        "본 연구는 예측모형 대신 설명 가능한 식품위생 점검 우선순위 지수(FSIPI)를 제안한다. 각 지표를 0-100점 백분위로 표준화한 뒤 행정처분 발생률, 최근 증가추세, 경제압박, 반복 핫스팟을 가중합해 점검 후보를 정렬한다.",
        margin_x,
        y - 38,
        usable_w,
        38,
        size=8.3,
    )
    y -= 130
    table(
        c,
        [
            ["구성", "의미", "가중치"],
            ["R", "음식점 1,000개당 행정처분 발생률", "55%"],
            ["T", "최근 3개월 대비 증가추세", "20%"],
            ["E", "소비자물가 기반 경제압박", "15%"],
            ["H", "최근 3개월 반복 핫스팟", "10%"],
        ],
        margin_x,
        y,
        [45, 290, 76],
        font_size=7.4,
    )
    y -= 18
    overlap = ", ".join(
        f"{row.Weight_Set}: {int(row.Top10_Overlap_With_Base)}개"
        for row in sens.itertuples()
    )
    para(
        c,
        f"가중치 민감도 분석에서는 기본안 대비 상위 10개 후보 중 8-9개가 유지되어 운영지수의 순위 안정성을 확인하였다. ({overlap})",
        margin_x,
        y - 32,
        usable_w,
        32,
        size=8.1,
    )
    y -= 48
    para(
        c,
        "<b>기대효과.</b> 절대 건수 중심 단속의 대도시 규모 착시를 줄이고, 물가 상승기에 취약 신호를 조기 탐색하며, 단속과 위생교육·시설관리 지원을 함께 설계할 수 있다. 기후 조건은 FSIPI 핵심 점수에 직접 넣지 않고 현장 확인용 기후주의 플래그로 활용한다.",
        margin_x,
        y - 52,
        usable_w,
        52,
        size=8.3,
    )
    y -= 64
    para(
        c,
        "<b>결론.</b> 경제적 압박은 식품접객업 행정처분 취약성과 연결될 가능성을 보였으며, 식품위생 점검은 절대 위반 건수가 아니라 FSIPI로 정렬한 고위험 시도-월을 중심으로 설계할 필요가 있다. 본 지수는 위생위험 확정모형이 아니라 제한된 점검 인력을 설명 가능하게 배분하기 위한 운영형 의사결정 지원지표다.",
        margin_x,
        y - 52,
        usable_w,
        52,
        size=8.3,
    )
    y -= 56
    section_title(c, "참고문헌", margin_x, y, usable_w)
    refs = (
        "식품의약품안전처(2026). 식품위생 행정처분 공개자료. 식품안전나라.<br/>"
        "통계청(2026). 소비자물가지수. KOSIS 국가통계포털. / 통계청(2023). 전국사업체조사 마이크로데이터. MDIS.<br/>"
        "기상청(2026). 종관기상관측(ASOS) 일별 관측자료. 기상자료개방포털.<br/>"
        "Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression. Hoboken, NJ, Wiley."
    )
    para(c, refs, margin_x, y - 55, usable_w, 48, size=6.7, color="#374151")
    footer(2)
    c.save()
    print(REPORT_PDF)


def draw_metric(c, x, y, w, h, label, value, note, accent):
    card(c, x, y, w, h, fill="#F8FAFC")
    c.setFillColor(colors.HexColor(accent))
    c.setFont(FONT_BOLD, 30)
    c.drawString(x + 14, y + h - 38, value)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.setFont(FONT_BOLD, 12)
    c.drawString(x + 14, y + 24, label)
    c.setFont(FONT, 9)
    c.setFillColor(colors.HexColor("#64748B"))
    c.drawString(x + 14, y + 10, note)


def build_a1_poster_pdf():
    summary, odds, priority, sens = load_stats()
    width, height = (594 * mm, 841 * mm)
    c = canvas.Canvas(str(POSTER_PDF), pagesize=(width, height))
    margin = 18 * mm

    c.setFillColor(colors.HexColor("#FFFFFF"))
    c.rect(0, 0, width, height, fill=1, stroke=0)

    red = colors.HexColor("#B91C1C")
    dark = colors.HexColor("#1F2937")
    muted = colors.HexColor("#6B7280")
    line = colors.HexColor("#9CA3AF")

    # Header
    header_h = 92 * mm
    c.setFillColor(colors.HexColor("#0F172A"))
    c.setFont(FONT_BOLD, 37)
    c.drawString(margin, height - margin - 8, "물가가 오르면 동네 식당의")
    c.setFillColor(red)
    c.setFont(FONT_BOLD, 45)
    c.drawString(margin, height - margin - 48, "위생도 흔들릴까?")
    c.setFillColor(dark)
    c.setFont(FONT_BOLD, 19)
    c.drawString(margin, height - margin - 78, ": 경제·기후·상권 기반 식품위생 점검 우선순위 분석")

    banner_w = 150 * mm
    c.setFillColor(red)
    c.rect(width - margin - banner_w, height - header_h, banner_w + margin, header_h, fill=1, stroke=0)
    c.setFillColor(colors.white)
    for i, (label, symbol) in enumerate([("운영비 압박", "₩"), ("사전점검", "!")]):
        cx = width - margin - banner_w + 42 * mm + i * 58 * mm
        cy = height - 38 * mm
        c.circle(cx, cy, 17 * mm, fill=1, stroke=0)
        c.setFillColor(red)
        c.setFont(FONT_BOLD, 25)
        c.drawCentredString(cx, cy - 8, symbol)
        c.setFillColor(colors.white)
        c.setFont(FONT_BOLD, 12)
        c.drawCentredString(cx, cy - 29 * mm, label)
    c.setStrokeColor(colors.HexColor("#E5E7EB"))
    c.setLineWidth(2)
    c.line(0, height - header_h, width, height - header_h)

    # Metrics
    metric_y = height - header_h - 40 * mm
    metric_w = (width - 2 * margin - 3 * 8 * mm) / 4
    metrics = [
        ("정제 행정처분", "2,901건", "식품접객업 필터링", "#B91C1C"),
        ("분석 격자", f"{int(summary['Rows_Used'])}개", "17시도 x 23개월", "#2563EB"),
        ("핫스팟", "Top 20%", "식당 1,000개당 발생률", "#7C3AED"),
        ("FSIPI", "4요소", "발생률·추세·경제·반복", "#059669"),
    ]
    for i, m in enumerate(metrics):
        draw_metric(c, margin + i * (metric_w + 8 * mm), metric_y, metric_w, 28 * mm, *m)

    def label_bar(y, title):
        c.setFillColor(red)
        c.setFont(FONT_BOLD, 22)
        c.drawString(margin, y, title)
        c.setStrokeColor(line)
        c.setLineWidth(1.2)
        c.line(margin, y - 7, width - margin, y - 7)

    # Section 1: background and data
    y1 = metric_y - 30 * mm
    label_bar(y1, "분석 배경 및 데이터")
    panel_y = y1 - 128 * mm
    left_w = (width - 2 * margin) * 0.47
    right_w = width - 2 * margin - left_w - 12 * mm
    card(c, margin, panel_y, left_w, 112 * mm, None, "#F9FAFB")
    c.setFont(FONT_BOLD, 16)
    c.setFillColor(dark)
    c.drawString(margin + 10, panel_y + 96 * mm, "외식업 경영 부담 증가에 따른 식품위생 취약 가능성")
    para(
        c,
        "최근 식재료비, 인건비, 임대료, 공공요금 상승으로 음식점의 운영비 부담이 지속적으로 증가하고 있다. 제한된 매출 안에서 식재료 구매, 인력 유지, 시설 관리, 위생관리 비용을 동시에 부담해야 하므로 경제적 압박과 행정처분 취약 신호를 함께 볼 필요가 있다.<br/><br/>기존 단속은 절대 행정처분 건수가 많은 지역에 집중되기 쉽다. 그러나 음식점 수가 많은 대도시는 건수도 구조적으로 커지므로, 식당 1,000개당 행정처분 발생률로 보정하였다.",
        margin + 10,
        panel_y + 16 * mm,
        left_w - 20,
        72 * mm,
        size=12.5,
        leading=17,
    )
    table(
        c,
        [
            ["영역", "데이터", "역할"],
            ["위생", "식품안전나라 행정처분", "종속변수"],
            ["경제", "KOSIS 소비자물가지수", "운영비 압박"],
            ["기후", "기상청 ASOS", "기후주의 플래그"],
            ["상권", "MDIS 전국사업체조사", "음식점 수 보정"],
        ],
        margin + left_w + 12 * mm,
        panel_y + 42 * mm,
        [38 * mm, 78 * mm, right_w - 116 * mm],
        font_size=10.5,
    )
    para(c, "17개 시도 x 23개월 = 391개 시도-월 관측치, 식품접객업 행정처분 2,901건 분석", margin + left_w + 12 * mm, panel_y + 22 * mm, right_w, 18 * mm, size=12, color="#374151")

    # Section 2: analysis
    y2 = panel_y - 28 * mm
    label_bar(y2, "데이터 분석")
    row_y = y2 - 138 * mm
    col_gap = 8 * mm
    col_w = (width - 2 * margin - 2 * col_gap) / 3
    card(c, margin, row_y, col_w, 120 * mm, "1. 규모 착시 확인", "#FFFFFF")
    para(
        c,
        "음식점업 사업체 수는 절대 행정처분 건수와 상관이 있었지만, 보정 발생률과는 거의 관련이 없었다.",
        margin + 12,
        row_y + 78 * mm,
        col_w - 24,
        30 * mm,
        size=11.5,
        leading=15,
    )
    table(
        c,
        [
            ["확인 항목", "결과"],
            ["사업체 수 vs 절대 건수", "r=0.422"],
            ["사업체 수 vs 보정 발생률", "r=-0.022"],
            ["사업체 수 효과", "OR=1.00"],
        ],
        margin + 12,
        row_y + 22 * mm,
        [70 * mm, col_w - 70 * mm - 24],
        font_size=10,
    )

    card(c, margin + col_w + col_gap, row_y, col_w, 120 * mm, "2. 물가는 정책 신호", "#FFFFFF")
    image(c, PLOT_DIR / "advanced_01_dual_axis_trend.png", margin + col_w + col_gap + 8, row_y + 25 * mm, col_w - 16, 62 * mm)
    para(
        c,
        f"총 CPI OR={odds.loc['CPI_TOTAL', 'OddsRatio']:.2f}, p={odds.loc['CPI_TOTAL', 'P_Value']:.3f}. 5% 확정 인과가 아닌 탐색적 정책 신호로 해석한다.",
        margin + col_w + col_gap + 12,
        row_y + 88 * mm,
        col_w - 24,
        22 * mm,
        size=10.6,
        leading=14,
    )

    card(c, margin + 2 * (col_w + col_gap), row_y, col_w, 120 * mm, "3. 로지스틱 회귀", "#FFFFFF")
    image(c, PLOT_DIR / "advanced_05_model_forest_plot.png", margin + 2 * (col_w + col_gap) + 8, row_y + 14 * mm, col_w - 16, 80 * mm)
    table(
        c,
        [
            ["변수", "OR", "p"],
            ["총 CPI", f"{odds.loc['CPI_TOTAL', 'OddsRatio']:.2f}", f"{odds.loc['CPI_TOTAL', 'P_Value']:.3f}"],
            ["THI", f"{odds.loc['THI', 'OddsRatio']:.2f}", f"{odds.loc['THI', 'P_Value']:.3f}"],
            ["음식점 수", f"{odds.loc['RESTAURANT_DENSITY', 'OddsRatio']:.2f}", "0.980"],
        ],
        margin + 2 * (col_w + col_gap) + 12,
        row_y + 86 * mm,
        [42 * mm, 25 * mm, 25 * mm],
        font_size=9.3,
    )

    # Section 3: FSIPI and conclusion
    y3 = row_y - 26 * mm
    label_bar(y3, "분석 활용 전략")
    bottom_y = y3 - 137 * mm
    left_w = (width - 2 * margin) * 0.52
    right_w = width - 2 * margin - left_w - 12 * mm
    card(c, margin, bottom_y, left_w, 120 * mm, "식품위생 점검 우선순위 지수(FSIPI)", "#FFFFFF")
    para(c, "<b>FSIPI = 0.55R + 0.20T + 0.15E + 0.10H</b>", margin + 14, bottom_y + 94 * mm, left_w - 28, 18 * mm, size=20, color="#7C3AED", bold=True)
    image(c, PLOT_DIR / "stat_04_priority_index_components.png", margin + 14, bottom_y + 16 * mm, left_w - 28, 72 * mm)
    para(c, "민감도 분석: 발생률 중심안 9개, 균형안 8개가 기본안 Top10 후보와 중복되어 순위 안정성을 확인하였다.", margin + 14, bottom_y + 5 * mm, left_w - 28, 10 * mm, size=10.2, color="#374151")

    card(c, margin + left_w + 12 * mm, bottom_y, right_w, 120 * mm, "결론 및 기대효과", "#FFFFFF")
    para(
        c,
        "경제적 압박은 식품접객업 행정처분 취약성과 연결될 가능성을 보였다. 식품위생 점검은 절대 위반 건수가 아니라 FSIPI로 정렬한 고위험 시도-월을 중심으로 설계할 필요가 있다.<br/><br/>FSIPI 80점 이상은 최우선 현장점검 후보로 검토하고, 단속과 함께 위생교육·시설관리 안내·자가점검 도구 제공을 병행한다. 기후 조건은 현장 확인용 주의 플래그로 활용한다.",
        margin + left_w + 12 * mm + 14,
        bottom_y + 38 * mm,
        right_w - 28,
        68 * mm,
        size=13.5,
        leading=18,
    )

    c.setFont(FONT, 8)
    c.setFillColor(muted)
    c.drawString(margin, 14 * mm, "자료: 식품안전나라 행정처분, KOSIS 소비자물가지수, 기상청 ASOS, MDIS 전국사업체조사. 행정처분 자료는 실제 위생상태가 아니라 행정처분 기반 취약 신호로 해석함.")
    c.save()
    print(POSTER_PDF)


if __name__ == "__main__":
    build_report_pdf()
    build_a1_poster_pdf()
