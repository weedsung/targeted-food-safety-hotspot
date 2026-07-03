from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
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
    a1_landscape = landscape((594 * mm, 841 * mm))
    width, height = a1_landscape
    c = canvas.Canvas(str(POSTER_PDF), pagesize=(width, height))
    margin = 24 * mm

    c.setFillColor(colors.HexColor("#F8FAFC"))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.setFont(FONT_BOLD, 46)
    c.drawString(margin, height - margin - 12, "물가가 오르면 동네 식당의 위생도 흔들릴까?")
    c.setFont(FONT, 18)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(margin, height - margin - 42, "경제·기후·상권 데이터 기반 식품접객업 행정처분 취약 신호 탐지와 점검 우선순위 지수")

    metric_y = height - margin - 135
    metric_w = (width - 2 * margin - 3 * 18) / 4
    metrics = [
        ("정제 행정처분", "2,901건", "식품접객업 필터링", "#DC2626"),
        ("분석 격자", f"{int(summary['Rows_Used'])}개", "17시도 x 23개월", "#2563EB"),
        ("핫스팟 기준", "Top 20%", f"임계값 {summary['Hotspot_Threshold_Per_1000_Restaurants']:.3f}건", "#7C3AED"),
        ("LightGBM AUC", f"{summary['LightGBM_AUC']:.3f}", "핵심 주장은 FSIPI", "#475569"),
    ]
    for i, m in enumerate(metrics):
        draw_metric(c, margin + i * (metric_w + 18), metric_y, metric_w, 76, *m)

    gap = 18
    col_w = (width - 2 * margin - 2 * gap) / 3
    top_y = 820
    bottom_y = 150
    section_h = 600

    card(c, margin, top_y, col_w, section_h, "1. 왜 절대 건수가 아닌가")
    para(
        c,
        "음식점 수가 많은 지역은 행정처분 건수도 구조적으로 커질 수 있다. 따라서 본 분석은 식당 1,000개당 행정처분 발생률로 보정하고, 상위 20% 시도-월을 점검 우선 후보로 정의하였다.",
        margin + 18,
        top_y + 390,
        col_w - 36,
        72,
        size=17,
        leading=23,
    )
    table(
        c,
        [
            ["확인 항목", "결과"],
            ["사업체 수 vs 절대 건수", "r = 0.422"],
            ["사업체 수 vs 보정 발생률", "r = -0.022"],
            ["사업체 수 로지스틱 효과", "OR = 1.00, p = 0.980"],
        ],
        margin + 18,
        top_y + 210,
        [150, col_w - 186],
        font_size=12,
    )

    card(c, margin + col_w + gap, top_y, col_w, section_h, "2. 물가는 원인이 아니라 정책 신호")
    image(c, PLOT_DIR / "advanced_01_dual_axis_trend.png", margin + col_w + gap + 16, top_y + 88, col_w - 32, 300)
    para(
        c,
        f"총 CPI 1표준편차 증가 시 핫스팟 오즈는 약 39% 증가 방향을 보였다(OR={odds.loc['CPI_TOTAL', 'OddsRatio']:.2f}, p={odds.loc['CPI_TOTAL', 'P_Value']:.3f}). 5% 유의수준의 확정 인과가 아니라 탐색적 정책 신호로 해석하였다.",
        margin + col_w + gap + 18,
        top_y + 410,
        col_w - 36,
        48,
        size=13.5,
        leading=18,
    )

    card(c, margin + 2 * (col_w + gap), top_y, col_w, section_h, "3. 설명 가능한 FSIPI 제안")
    para(c, "<b>FSIPI = 0.55R + 0.20T + 0.15E + 0.10H</b>", margin + 2 * (col_w + gap) + 18, top_y + 485, col_w - 36, 34, size=22, color="#7C3AED", bold=True)
    table(
        c,
        [
            ["구성", "의미", "비중"],
            ["R", "행정처분 발생률", "55%"],
            ["T", "최근 증가추세", "20%"],
            ["E", "경제압박", "15%"],
            ["H", "반복 핫스팟", "10%"],
        ],
        margin + 2 * (col_w + gap) + 18,
        top_y + 260,
        [60, col_w - 160, 70],
        font_size=12,
    )
    para(
        c,
        "기후 조건은 회귀 방향이 불안정하므로 핵심 점수에 직접 넣지 않고 현장 확인용 기후주의 플래그로 분리하였다.",
        margin + 2 * (col_w + gap) + 18,
        top_y + 170,
        col_w - 36,
        42,
        size=13.2,
        leading=17,
    )

    card(c, margin, bottom_y, col_w, section_h, "4. 로지스틱 회귀 핵심 결과")
    image(c, PLOT_DIR / "advanced_05_model_forest_plot.png", margin + 16, bottom_y + 62, col_w - 32, 340)
    table(
        c,
        [
            ["변수", "OR", "p"],
            ["총 CPI", f"{odds.loc['CPI_TOTAL', 'OddsRatio']:.2f}", f"{odds.loc['CPI_TOTAL', 'P_Value']:.3f}"],
            ["THI", f"{odds.loc['THI', 'OddsRatio']:.2f}", f"{odds.loc['THI', 'P_Value']:.3f}"],
            ["음식점 수", f"{odds.loc['RESTAURANT_DENSITY', 'OddsRatio']:.2f}", f"{odds.loc['RESTAURANT_DENSITY', 'P_Value']:.3f}"],
        ],
        margin + 18,
        bottom_y + 420,
        [125, 75, 75],
        font_size=11.5,
    )

    card(c, margin + col_w + gap, bottom_y, col_w, section_h, "5. 점검 우선순위 산출")
    image(c, PLOT_DIR / "stat_04_priority_index_components.png", margin + col_w + gap + 18, bottom_y + 108, col_w - 36, 330)
    sens_text = "가중치 민감도: 발생률 중심안 9개, 균형안 8개가 기본안 Top10 후보와 중복"
    para(c, sens_text, margin + col_w + gap + 18, bottom_y + 50, col_w - 36, 36, size=15, leading=19)

    card(c, margin + 2 * (col_w + gap), bottom_y, col_w, section_h, "6. 결론 및 활용")
    para(
        c,
        "<b>결론.</b> 경제적 압박은 식품접객업 행정처분 취약성과 연결될 가능성을 보였다. 식품위생 점검은 절대 위반 건수가 아니라 FSIPI로 정렬한 고위험 시도-월을 중심으로 설계할 필요가 있다.<br/><br/><b>활용.</b> 지자체는 FSIPI 80점 이상을 최우선 현장점검 후보로 검토하고, 단속과 함께 위생교육·시설관리 안내·자가점검 도구 제공을 병행한다.",
        margin + 2 * (col_w + gap) + 18,
        bottom_y + 250,
        col_w - 36,
        150,
        size=17,
        leading=23,
    )
    para(
        c,
        "자료: 식품안전나라 행정처분, KOSIS 소비자물가지수, 기상청 ASOS, MDIS 전국사업체조사. 행정처분 자료는 실제 위생상태 그 자체가 아니라 행정처분 기반 취약 신호로 해석함.",
        margin + 2 * (col_w + gap) + 18,
        bottom_y + 62,
        col_w - 36,
        36,
        size=10,
        color="#64748B",
        leading=13,
    )

    c.save()
    print(POSTER_PDF)


if __name__ == "__main__":
    build_report_pdf()
    build_a1_poster_pdf()
