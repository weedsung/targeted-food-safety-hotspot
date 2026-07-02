import os
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / 'output' / 'pdf'
PLOT_DIR = BASE_DIR / 'output' / 'eda_plots'
MODEL_DIR = BASE_DIR / 'output' / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / 'food_safety_statistical_poster.pdf'


def register_fonts():
    font_path = Path(r'C:\Windows\Fonts\malgun.ttf')
    bold_path = Path(r'C:\Windows\Fonts\malgunbd.ttf')
    if font_path.exists():
        pdfmetrics.registerFont(TTFont('Malgun', str(font_path)))
    if bold_path.exists():
        pdfmetrics.registerFont(TTFont('Malgun-Bold', str(bold_path)))
    return 'Malgun' if font_path.exists() else 'Helvetica', 'Malgun-Bold' if bold_path.exists() else 'Helvetica-Bold'


FONT, FONT_BOLD = register_fonts()


def draw_paragraph(c, text, x, y, w, h, size=10, color=colors.HexColor('#1F2937'), leading=None, bold=False):
    style = ParagraphStyle(
        'poster',
        fontName=FONT_BOLD if bold else FONT,
        fontSize=size,
        leading=leading or size * 1.35,
        textColor=color,
        spaceAfter=0,
    )
    p = Paragraph(text, style)
    _, used_h = p.wrap(w, h)
    p.drawOn(c, x, y + h - used_h)
    return used_h


def draw_card(c, x, y, w, h, title=None):
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor('#CBD5E1'))
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
    if title:
        c.setFillColor(colors.HexColor('#0F172A'))
        c.setFont(FONT_BOLD, 13)
        c.drawString(x + 10, y + h - 20, title)


def draw_metric(c, x, y, label, value, note, accent):
    c.setFillColor(colors.HexColor('#F8FAFC'))
    c.setStrokeColor(colors.HexColor('#CBD5E1'))
    c.roundRect(x, y, 155, 58, 4, fill=1, stroke=1)
    c.setFillColor(accent)
    c.setFont(FONT_BOLD, 23)
    c.drawString(x + 12, y + 30, value)
    c.setFillColor(colors.HexColor('#334155'))
    c.setFont(FONT_BOLD, 10.5)
    c.drawString(x + 12, y + 15, label)
    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont(FONT, 8.2)
    c.drawString(x + 12, y + 5, note)


def draw_image_fit(c, path, x, y, w, h):
    if not path.exists():
        draw_paragraph(c, f'이미지 없음: {path.name}', x, y, w, h, size=9, color=colors.red)
        return
    c.drawImage(str(path), x, y, width=w, height=h, preserveAspectRatio=True, anchor='c')


def build_poster():
    width, height = landscape(A3)
    margin = 14 * mm
    c = canvas.Canvas(str(PDF_PATH), pagesize=(width, height))

    c.setFillColor(colors.HexColor('#F8FAFC'))
    c.rect(0, 0, width, height, fill=1, stroke=0)

    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont(FONT_BOLD, 25)
    c.drawString(margin, height - margin - 8, '식당이 많은 곳이 아니라, 위반율이 흔들리는 시공간을 찾다')
    c.setFont(FONT, 10.5)
    c.setFillColor(colors.HexColor('#475569'))
    c.drawString(margin, height - margin - 27, '식품접객업 행정처분 x 소비자물가지수 x 기상지표 x 전국사업체조사 기반 위생 핫스팟 타겟팅')

    summary = pd.read_csv(MODEL_DIR / 'model_training_summary.csv').iloc[0]
    odds = pd.read_csv(MODEL_DIR / 'odds_ratios.csv').set_index('Feature')
    comp = pd.read_csv(MODEL_DIR / 'comprehensive_statistics.csv').set_index('Feature')

    metric_y = height - margin - 88
    metric_w = 155
    metric_gap = 14
    draw_metric(c, margin, metric_y, '정제 위반건', '2,901', '이상 주소 2건 분리', colors.HexColor('#DC2626'))
    draw_metric(c, margin + (metric_w + metric_gap), metric_y, '모델 격자', f"{int(summary['Rows_Used'])}", '17개 시도 x 23개월', colors.HexColor('#2563EB'))
    draw_metric(c, margin + 2 * (metric_w + metric_gap), metric_y, '핫스팟 기준', 'Top 20%', '식당 1,000개당 위반율', colors.HexColor('#7C3AED'))
    draw_metric(c, margin + 3 * (metric_w + metric_gap), metric_y, 'AUC', f"{summary['LightGBM_AUC']:.3f}", '보조 타겟팅 지표', colors.HexColor('#475569'))

    left_x = margin
    left_w = 305
    mid_x = left_x + left_w + 22
    mid_w = 375
    right_x = mid_x + mid_w + 22
    right_w = width - margin - right_x
    top_y = height - margin - 355
    card_h = 245
    bottom_y = margin + 60

    draw_card(c, left_x, top_y, left_w, card_h, '분석 설계')
    design_text = (
        '절대 위반 건수는 식당 수가 많은 대도시에 유리하게 커진다. '
        '따라서 본 분석은 <b>식당 1,000개당 위반율</b>을 만들고, '
        '그 값의 상위 20% 시도-월을 핫스팟으로 정의했다.<br/><br/>'
        '지역명은 유효 시도명 기준으로 재정제했고, 전북 명칭을 통일했다. '
        '2026-04는 CPI 결측으로 모델 학습에서는 제외되었다.'
    )
    draw_paragraph(c, design_text, left_x + 16, top_y + 20, left_w - 32, card_h - 58, size=13)

    draw_card(c, left_x, bottom_y, left_w, card_h, '핵심 해석')
    insight_text = (
        '<b>식당 수가 많은 지역</b>은 절대 위반 건수와 관련이 있지만, '
        '식당 수 대비 핫스팟 여부에는 직접 효과가 거의 없었다.<br/><br/>'
        '반면 <b>총 소비자물가지수</b>는 밀집도 통제 후 위험 증가 방향을 보였다. '
        '이는 확정적 인과 증명보다 정책 신호로 해석하는 것이 안전하다.'
    )
    draw_paragraph(c, insight_text, left_x + 16, bottom_y + 20, left_w - 32, card_h - 58, size=13)

    draw_card(c, mid_x, top_y, mid_w, card_h, '로지스틱 회귀 결과')
    table_data = [['변수', 'OR', 'p-value', '안전한 해석']]
    rows = [
        ('CPI_TOTAL', '총 CPI', '물가 위험 신호'),
        ('THI', '불쾌지수', '보정 위험률에서 음의 관계'),
        ('RESTAURANT_DENSITY', '식당 밀집도', '직접 효과 거의 없음'),
        ('CPI_AGRI', '농축수산 CPI', '다중공선성 주의'),
    ]
    for feature, display_name, label in rows:
        table_data.append([
            display_name,
            f"{odds.loc[feature, 'OddsRatio']:.2f}",
            f"{odds.loc[feature, 'P_Value']:.3f}",
            label,
        ])
    table = Table(table_data, colWidths=[92, 55, 65, 142], rowHeights=30)
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), FONT),
        ('FONT', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 10.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    table.wrapOn(c, mid_w - 22, card_h - 70)
    table.drawOn(c, mid_x + 11, top_y + 52)

    draw_card(c, mid_x, bottom_y, mid_w, card_h, '모델 비교: 정제 전 -> 정제 후')
    compare_text = (
        f"CPI_TOTAL OR은 1.36에서 <b>{odds.loc['CPI_TOTAL', 'OddsRatio']:.2f}</b>로 상승했고, "
        f"p-value는 0.126에서 <b>{odds.loc['CPI_TOTAL', 'P_Value']:.3f}</b>로 개선되었다.<br/><br/>"
        f"식당 밀집도 OR은 <b>{odds.loc['RESTAURANT_DENSITY', 'OddsRatio']:.2f}</b>, "
        f"p-value는 <b>{odds.loc['RESTAURANT_DENSITY', 'P_Value']:.3f}</b>로, "
        '위반율 기준 직접 효과가 거의 없다는 결론이 더 명확해졌다.'
    )
    draw_paragraph(c, compare_text, mid_x + 16, bottom_y + 20, mid_w - 32, card_h - 58, size=13)

    draw_card(c, right_x, top_y, right_w, card_h, '주요 시각화')
    img_w = (right_w - 42) / 2
    img_h = 86
    draw_image_fit(c, PLOT_DIR / 'advanced_01_dual_axis_trend.png', right_x + 14, top_y + 130, img_w, img_h)
    draw_image_fit(c, PLOT_DIR / 'advanced_05_model_forest_plot.png', right_x + 28 + img_w, top_y + 130, img_w, img_h)
    draw_image_fit(c, PLOT_DIR / 'advanced_02_feature_importance.png', right_x + 14, top_y + 35, img_w, img_h)
    draw_image_fit(c, PLOT_DIR / 'advanced_03_correlation_heatmap.png', right_x + 28 + img_w, top_y + 35, img_w, img_h)

    draw_card(c, right_x, bottom_y, right_w, card_h, '정책 제안')
    policy_text = (
        '단속 대상을 자동 결정하기보다, 위험 신호를 정렬해 현장 판단을 돕는 '
        '<b>보조 대시보드</b>로 활용한다.<br/><br/>'
        '1. 식당 수 대비 위반율로 대도시 착시를 줄인다.<br/>'
        '2. 물가 상승 구간을 조기 경보로 본다.<br/>'
        '3. 기후와 공간 정보를 교차해 점검 우선순위를 만든다.'
    )
    draw_paragraph(c, policy_text, right_x + 16, bottom_y + 20, right_w - 32, card_h - 58, size=13)

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont(FONT, 7.2)
    footnote = (
        '자료: 식약처 행정처분, KOSIS 소비자물가지수, 기상청 ASOS, MDIS 전국사업체조사. '
        '수치 기준: output/models/odds_ratios.csv, model_training_summary.csv. '
        'CPI_TOTAL은 10% 유의수준의 정책 신호이며 5% 기준 확정 인과로 표현하지 않음. '
        f"LightGBM AUC={summary['LightGBM_AUC']:.3f}."
    )
    c.drawString(margin, margin + 14, footnote[:185])
    c.drawString(margin, margin + 4, footnote[185:])

    c.save()
    print(PDF_PATH)


if __name__ == '__main__':
    build_poster()
