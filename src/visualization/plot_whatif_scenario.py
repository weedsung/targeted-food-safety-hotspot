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

def plot_whatif_scenario():
    print("4. 'What-if' 물가 상승 시나리오 분석 중...")
    
    # 오즈비 로드 (없으면 기본값 사용)
    if os.path.exists(OR_PATH):
        or_df = pd.read_csv(OR_PATH)
        cpi_or = or_df[or_df['Feature'] == 'CPI_TOTAL']['OddsRatio'].values[0]
    else:
        # analysis_results.md에 기록된 값
        cpi_or = 1.33 
    
    # 시나리오 설정: CPI가 현재 대비 0% ~ 10% 상승할 때
    # 오즈비는 1 표준편차 상승 시의 배수이므로, 선형적으로 확률 변화를 시뮬레이션 (단순화)
    rise_pct = np.linspace(0, 10, 11)
    
    # 베이스라인 확률이 20%(상위 20% 핫스팟 정의 기준)라고 가정할 때의 변화
    # P = odds / (1 + odds)
    base_prob = 0.20
    base_odds = base_prob / (1 - base_prob)
    
    # CPI 1 표준편차(약 2-3% 상승 가정)당 cpi_or배 증가
    # 여기서는 시각적 이해를 돕기 위해 1%당 일정 비율 증가로 매핑
    risks = []
    for r in rise_pct:
        # 가상의 영향력 계산 (1% 상승 시 오즈비의 일정 비율 적용)
        current_odds = base_odds * (cpi_or ** (r / 2.0)) # 2% 상승을 1 표준편차로 가정
        current_prob = current_odds / (1 + current_odds)
        risks.append(current_prob * 100) # 퍼센트화

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rise_pct, risks, marker='o', color='#8B0000', linewidth=3, markersize=9, zorder=5)
    ax.fill_between(rise_pct, 20, risks, color='red', alpha=0.12)
    
    ax.axhline(20, color='grey', linestyle='--', linewidth=1.5, label='현재 평균 위험 수준 (Top 20%)')
    
    ax.set_title("물가 상승 시나리오별 '위생 핫스팟' 발생 확률 예측", fontsize=16, fontweight='bold', pad=25)
    ax.set_xlabel('물가 추가 상승률 (%)', fontsize=12)
    ax.set_ylabel('고위험군(핫스팟) 편입 확률 (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 정보 상자 (annotate 대신 텍스트 박스 사용 → 제목 가림 방지)
    final_risk = risks[-1]
    ax.annotate(
        f'CPI +10% 상승 시\n위험 확률 {final_risk:.1f}%로 급증',
        xy=(10, final_risk),            # 화살표 끝에 (x=10, 끝점)
        xytext=(6.5, final_risk - 10),  # 텍스트 위치: 아래으로 내려서 제목 가림 방지
        fontsize=12,
        fontweight='bold',
        color='#8B0000',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF5F5', edgecolor='#C0392B', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2),
    )
    
    # 시작점 (x=0) 주석
    ax.annotate(
        f'현재 {risks[0]:.0f}%',
        xy=(0, risks[0]),
        xytext=(0.3, risks[0] + 2.5),
        fontsize=10, color='#555',
    )
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'advanced_04_whatif_scenario.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"완료: {save_path}")

if __name__ == "__main__":
    plot_whatif_scenario()
