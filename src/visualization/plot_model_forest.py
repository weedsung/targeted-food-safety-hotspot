import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), 'scratch', 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OR_PATH = os.path.join(BASE_DIR, 'output', 'models', 'odds_ratios.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_model_forest():
    print("5. 모델 핵심 계수(Odds Ratio) 포레스트 플롯 생성 중...")
    
    # 오즈비 데이터 준비
    if os.path.exists(OR_PATH):
        or_df = pd.read_csv(OR_PATH)
    else:
        # 데이터가 없으면 직접 입력 (analysis_results.md 기준)
        data = {
            'Feature': ['CPI_TOTAL', 'RESTAURANT_DENSITY', 'THI', 'CPI_AGRI'],
            'OddsRatio': [1.33, 1.01, 0.74, 0.56]
        }
        or_df = pd.DataFrame(data)
    
    or_df = or_df.sort_values('OddsRatio', ascending=True).reset_index(drop=True)
    
    # 한글 매핑
    label_map = {
        'CPI_TOTAL': '총 소비자물가지수 (경제)',
        'RESTAURANT_DENSITY': '식당 밀집도 (환경)',
        'THI': '기후 불쾌지수 (날씨)',
        'CPI_AGRI': '농축수산물 물가 (식재료)'
    }
    or_df['Label'] = or_df['Feature'].map(lambda x: label_map.get(x, x))

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 1.0 기준선
    ax.axvline(1.0, color='crimson', linestyle='--', linewidth=1.5, label='기준점 (영향 없음)', zorder=1)
    
    # 연결선 + 점 + 수치 라벨 (enumerate로 y좌표 정확히 지정)
    labels = or_df['Label'].tolist()
    for idx, row in or_df.iterrows():
        color = '#C0392B' if row['OddsRatio'] >= 1.0 else '#2980B9'  # OR>1=빨강(위험), OR<1=파랑(보호)
        ax.hlines(idx, 1.0, row['OddsRatio'], colors='gray', alpha=0.5, linewidth=2, zorder=2)
        ax.scatter(row['OddsRatio'], idx, color=color, s=120, zorder=3)
        # 수치 라벨: OR>1이면 오른쪽, OR<1이면 왼쪽에 배치
        offset = 0.04 if row['OddsRatio'] >= 1.0 else -0.04
        ha = 'left' if row['OddsRatio'] >= 1.0 else 'right'
        ax.text(row['OddsRatio'] + offset, idx, f"OR {row['OddsRatio']:.2f}",
                va='center', ha=ha, fontsize=12, fontweight='bold', color=color)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_title("위생 위반 핫스팟 결정 요인 분석 (로지스틱 회귀 오즈비)\n(OR > 1.0: 위반 확률 증가 요인 / OR < 1.0: 위반 확률 감소 요인)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('오즈비 (Odds Ratio: 1표준편차 증가 시 위험 배수)', fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, max(or_df['OddsRatio']) + 0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'advanced_05_model_forest_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"완료: {save_path}")

if __name__ == "__main__":
    plot_model_forest()
